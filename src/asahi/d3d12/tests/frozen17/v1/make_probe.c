/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Probe-shader generator: NIR in, DXIL out.
 *
 * `dxc` does not run on Asahi/aarch64, so the only DXIL this project had
 * for tests was the vkd3d-proton corpus -- LGPL-derived, and therefore
 * usable as a reference/oracle but never vendored into this MIT tree.
 * That left GPU suites either depending on a corpus the repository cannot
 * carry, or asserting nothing about shaders it could not author.
 *
 * Mesa already contains the other direction: src/microsoft/compiler is
 * NIR -> DXIL, MIT, and in tree. Building probe shaders with nir_builder
 * and running them through it gives clean-room, committable DXIL with no
 * dxc dependency at all.
 *
 * This is a BUILD-TIME tool. Nothing in libd3d12agx links it, and no
 * application shader passes through it.
 *
 * The generated .dxil are committed next to the sources that produced
 * them, so replicating the suites needs neither this tool nor dxc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "nir_builder.h"
#include "nir_to_dxil.h"
#include "util/blob.h"

/*
 * gs_input_vs: a vertex shader whose only input is the vertex ID and
 * whose VAR0 is a full float4.
 *
 * This exists because the geometry test needs one and nothing else
 * provides it: of the corpus vertex shaders that write VAR0, the ones
 * with an xyzw write read vertex ATTRIBUTES (which need an
 * attribute-fetch prolog this driver does not have), and the
 * attribute-free ones write only .x. A geometry shader that copies
 * VAR0.xyzw to POSITION therefore had an undefined position, and a test
 * cannot assert on undefined behaviour.
 *
 * Four vertex IDs give four points, one per quadrant, at +-0.5:
 *
 *    id 0 -> (-0.5, -0.5)      id 1 -> (+0.5, -0.5)
 *    id 2 -> (-0.5, +0.5)      id 3 -> (+0.5, +0.5)
 *
 * w is 1.0 so the position is usable as-is after the geometry shader
 * copies it. POSITION is written too, so the same shader is meaningful
 * with and without a geometry shader bound -- which is what makes the
 * control comparison in g6_geometry mean something.
 */
/* One vec4 slot per variable, which is what the DXIL signature expects. */
static int
probe_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static nir_shader *
build_gs_input_vs(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, opts,
                                                  "gs_input_vs");

   /* nir_to_dxil creates the SV_VertexID input variable itself, but only
    * when system_values_read says the shader wants it, and it only
    * lowers the ZERO_BASE form. Miss either and it asserts on a null
    * variable deep inside dxil_nir.
    */
   BITSET_SET(b.shader->info.system_values_read,
              SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   nir_def *id = nir_load_vertex_id_zero_base(&b);

   /* x = (id & 1) ? +0.5 : -0.5 ; y = (id & 2) ? +0.5 : -0.5 */
   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *neg_half = nir_imm_float(&b, -0.5f);

   nir_def *x = nir_bcsel(
      &b, nir_ine_imm(&b, nir_iand_imm(&b, id, 1), 0), half, neg_half);
   nir_def *y = nir_bcsel(
      &b, nir_ine_imm(&b, nir_iand_imm(&b, id, 2), 0), half, neg_half);

   nir_def *pos =
      nir_vec4(&b, x, y, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f));

   nir_variable *var0 = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "var0");
   var0->data.location = VARYING_SLOT_VAR0;
   var0->data.driver_location = 0;
   nir_store_var(&b, var0, pos, 0xf);

   nir_variable *posv = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   posv->data.location = VARYING_SLOT_POS;
   posv->data.driver_location = 1;
   nir_store_var(&b, posv, pos, 0xf);

   b.shader->info.outputs_written =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_POS);
   return b.shader;
}

/*
 * gs_point_passthrough: points in, points out, VAR0 copied to POSITION.
 *
 * Deliberately the cheapest geometry shader that is still a geometry
 * shader. Static topology and no side effects, so the lowering's main
 * body compiles to almost nothing and the shape is
 * POLY_GS_SHAPE_STATIC_PER_INSTANCE -- the one the runtime dispatches
 * with a plain non-indexed draw and no output index buffer.
 */
static nir_shader *
build_gs_point_passthrough(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY, opts,
                                                  "gs_point_passthrough");

   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;

   nir_variable *in = nir_variable_create(
      b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 1, 0), "in_var0");
   in->data.location = VARYING_SLOT_VAR0;
   in->data.driver_location = 0;

   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_vec4_type(), "out_pos");
   out->data.location = VARYING_SLOT_POS;
   out->data.driver_location = 0;

   nir_deref_instr *elem =
      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, in), 0);
   nir_store_var(&b, out, nir_load_deref(&b, elem), 0xf);

   nir_emit_vertex(&b, .stream_id = 0);
   nir_end_primitive(&b, .stream_id = 0);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_POS);
   return b.shader;
}

/*
 * tess_hs / tess_ds: the minimum that is still a tessellation pipeline.
 *
 * Quad domain, integer spacing. The hull shader is a pass-through with
 * constant tessellation factors, so the tessellated output is a fixed,
 * hand-checkable grid rather than something that depends on the input
 * patch. The domain shader places each domain point by bilinear
 * interpolation of the patch corners, which is the identity for a patch
 * covering the viewport -- so the result is a solid filled quad, and any
 * mis-tessellation shows up as missing coverage.
 */
static nir_shader *
build_tess_hs(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_CTRL, opts,
                                                  "tess_hs");
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_QUADS;
   b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   b.shader->info.tess.tcs_vertices_out = 4;
   b.shader->info.tess.ccw = false;
   b.shader->info.tess.point_mode = false;

   nir_variable *in = nir_variable_create(
      b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 4, 0), "in_pos");
   in->data.location = VARYING_SLOT_VAR0;
   in->data.driver_location = 0;

   nir_variable *out = nir_variable_create(
      b.shader, nir_var_shader_out,
      glsl_array_type(glsl_vec4_type(), 4, 0), "out_pos");
   out->data.location = VARYING_SLOT_VAR0;
   out->data.driver_location = 0;

   /* Pass the control point through. */
   nir_def *id = nir_load_invocation_id(&b);
   nir_deref_instr *src =
      nir_build_deref_array(&b, nir_build_deref_var(&b, in), id);
   nir_deref_instr *dst =
      nir_build_deref_array(&b, nir_build_deref_var(&b, out), id);
   nir_store_deref(&b, dst, nir_load_deref(&b, src), 0xf);

   /* Constant tessellation factors: a 2x2 grid of quads. */
   nir_variable *outer = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_array_type(glsl_float_type(), 4, 0),
      "outer");
   outer->data.location = VARYING_SLOT_TESS_LEVEL_OUTER;
   outer->data.driver_location = 4;
   outer->data.patch = true;
   /* The tess factors are COMPACT arrays -- floats packed into one slot's
    * components rather than one slot each. dxil_signature asserts on it.
    */
   outer->data.compact = true;

   nir_variable *inner = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_array_type(glsl_float_type(), 2, 0),
      "inner");
   inner->data.location = VARYING_SLOT_TESS_LEVEL_INNER;
   inner->data.driver_location = 8;
   inner->data.patch = true;
   inner->data.compact = true;

   for (unsigned i = 0; i < 4; i++) {
      nir_store_deref(&b,
                      nir_build_deref_array_imm(
                         &b, nir_build_deref_var(&b, outer), i),
                      nir_imm_float(&b, 2.0f), 0x1);
   }
   for (unsigned i = 0; i < 2; i++) {
      nir_store_deref(&b,
                      nir_build_deref_array_imm(
                         &b, nir_build_deref_var(&b, inner), i),
                      nir_imm_float(&b, 2.0f), 0x1);
   }

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.patch_outputs_written =
      BITFIELD_BIT(VARYING_SLOT_TESS_LEVEL_OUTER - VARYING_SLOT_PATCH0) |
      BITFIELD_BIT(VARYING_SLOT_TESS_LEVEL_INNER - VARYING_SLOT_PATCH0);
   return b.shader;
}

static nir_shader *
build_tess_ds(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_EVAL, opts,
                                                  "tess_ds");
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_QUADS;
   b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   b.shader->info.tess.ccw = false;
   b.shader->info.tess.point_mode = false;

   nir_variable *in = nir_variable_create(
      b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 4, 0), "in_pos");
   in->data.location = VARYING_SLOT_VAR0;
   in->data.driver_location = 0;

   nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_vec4_type(), "gl_Position");
   pos->data.location = VARYING_SLOT_POS;
   pos->data.driver_location = 0;

   nir_def *uv = nir_load_tess_coord(&b);
   nir_def *u = nir_channel(&b, uv, 0);
   nir_def *v = nir_channel(&b, uv, 1);

   nir_def *c[4];
   for (unsigned i = 0; i < 4; i++) {
      c[i] = nir_load_deref(
         &b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, in), i));
   }

   /* Bilinear over a GRID-ordered patch, matching gs_input_vs, which
    * places control point i at x = (i & 1), y = (i & 2). Ring order
    * (c0,c1,c3,c2) would twist the quad into a bowtie and cover half the
    * area, which is exactly the kind of thing this test should catch
    * rather than encode.
    */
   nir_def *top = nir_flrp(&b, c[0], c[1], u);
   nir_def *bot = nir_flrp(&b, c[2], c[3], u);
   nir_def *p = nir_flrp(&b, top, bot, v);

   nir_store_var(&b, pos,
                 nir_vec4(&b, nir_channel(&b, p, 0), nir_channel(&b, p, 1),
                          nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
                 0xf);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_POS);
   return b.shader;
}

/*
 * mesh_quad: the smallest thing that is still a mesh shader.
 *
 * One workgroup of four threads. Thread i writes control point i of a
 * full-screen quad at (+-1, +-1); threads 0 and 1 each emit one
 * triangle. Four vertices, two primitives -- the same shape the corpus
 * ms_simple produces, but authored here so it can be committed.
 *
 * Emitting this at all needed mesh support in nir_to_dxil, which had
 * none: the shader kind, the MSState entry property (whose operand 0 is
 * the numthreads node -- mesh shaders do not use the compute NumThreads
 * tag), and the setMeshOutputCounts / storeVertexOutput / emitIndices
 * intrinsics.
 */
static nir_shader *
build_mesh_quad(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH, opts,
                                                  "mesh_quad");
   b.shader->info.workgroup_size[0] = 4;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 4;
   b.shader->info.mesh.max_primitives_out = 2;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;

   /* The store below is emitted as an already-lowered intrinsic, but the
    * output signature is built by walking VARIABLES -- without this
    * declaration the container carries an empty OSG1 and no consumer can
    * see that the shader writes a position. */
   nir_variable *pos_var = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "position");
   pos_var->data.location = VARYING_SLOT_POS;
   pos_var->data.driver_location = 0;

   nir_def *id = nir_channel(&b, nir_load_local_invocation_id(&b), 0);

   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 4),
                                      nir_imm_int(&b, 2), nir_imm_int(&b, 0));

   /* x = (id & 1) ? 1 : -1 ; y = (id & 2) ? 1 : -1 */
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *neg = nir_imm_float(&b, -1.0f);
   nir_def *x =
      nir_bcsel(&b, nir_ine_imm(&b, nir_iand_imm(&b, id, 1), 0), one, neg);
   nir_def *y =
      nir_bcsel(&b, nir_ine_imm(&b, nir_iand_imm(&b, id, 2), 0), one, neg);

   nir_def *pos = nir_vec4(&b, x, y, nir_imm_float(&b, 0.0f), one);

   nir_store_per_vertex_output(&b, pos, id, nir_imm_int(&b, 0), .base = 0,
                               .write_mask = 0xf,
                               .src_type = nir_type_float32,
                               .io_semantics.location = VARYING_SLOT_POS,
                               .io_semantics.num_slots = 1);

   /* Threads 0 and 1 emit the two triangles: (0,1,2) and (1,3,2). */
   nir_push_if(&b, nir_ult_imm(&b, id, 2));
   {
      nir_def *i0 = nir_bcsel(&b, nir_ieq_imm(&b, id, 0),
                              nir_imm_int(&b, 0), nir_imm_int(&b, 1));
      nir_def *i1 = nir_bcsel(&b, nir_ieq_imm(&b, id, 0),
                              nir_imm_int(&b, 1), nir_imm_int(&b, 3));
      nir_def *i2 = nir_imm_int(&b, 2);

      nir_store_per_primitive_output(
         &b, nir_vec3(&b, i0, i1, i2), id, nir_imm_int(&b, 0), .base = 0,
         .write_mask = 0x7, .src_type = nir_type_uint32,
         .io_semantics.location = VARYING_SLOT_PRIMITIVE_INDICES,
         .io_semantics.num_slots = 1);
   }
   nir_pop_if(&b, NULL);

   b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_POS);
   b.shader->info.per_primitive_outputs =
      BITFIELD64_BIT(VARYING_SLOT_PRIMITIVE_INDICES);
   return b.shader;
}

/*
 * as_dispatch: one amplification thread group that launches one mesh
 * group. The payload is declared (D3D12 requires one) and never read --
 * mesh_quad ignores it -- so nothing here depends on payload plumbing.
 */
static nir_shader *
build_as_dispatch(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TASK, opts,
                                                  "as_dispatch");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = 4;

   nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 1, 1, 1));
   return b.shader;
}

/*
 * vb_vs / vb_ps: the G10 pair. Unlike every other probe in this tree,
 * vb_vs derives NOTHING from SV_VertexID -- both of its inputs are real
 * vertex attributes, so if attribute fetch does not work this shader can
 * draw nothing at all.
 *
 * Two attributes on purpose, INTERLEAVED in one buffer:
 *
 *   TEXCOORD0  float2  offset 0   position
 *   TEXCOORD1  float4  offset 8   colour
 *
 * A driver that fetched only from the start of the buffer would get the
 * position right and the colour wrong, which the second attribute is
 * there to catch. Generic inputs are named TEXCOORD<driver_location> by
 * nir_to_dxil, so the D3D12 input layout in g10_vertex_buffers.c must use
 * those exact semantic names.
 */
static nir_shader *
build_vb_vs(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, opts,
                                                  "vb_vs");

   nir_variable *in_pos = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec_type(2), "pos");
   in_pos->data.location = VERT_ATTRIB_GENERIC0;
   in_pos->data.driver_location = 0;

   nir_variable *in_col = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "col");
   in_col->data.location = VERT_ATTRIB_GENERIC0 + 1;
   in_col->data.driver_location = 1;

   nir_def *xy = nir_load_var(&b, in_pos);
   nir_def *col = nir_load_var(&b, in_col);

   nir_def *pos = nir_vec4(&b, nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
                           nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f));

   nir_variable *out_col = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "var0");
   out_col->data.location = VARYING_SLOT_VAR0;
   out_col->data.driver_location = 0;
   nir_store_var(&b, out_col, col, 0xf);

   nir_variable *out_pos = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   out_pos->data.location = VARYING_SLOT_POS;
   out_pos->data.driver_location = 1;
   nir_store_var(&b, out_pos, pos, 0xf);

   b.shader->info.inputs_read =
      BITFIELD64_BIT(VERT_ATTRIB_GENERIC0) | BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + 1);
   b.shader->info.outputs_written =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_POS);
   return b.shader;
}

static nir_shader *
build_vb_ps(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, opts,
                                                  "vb_ps");

   nir_variable *in_col = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "var0");
   in_col->data.location = VARYING_SLOT_VAR0;
   in_col->data.driver_location = 0;

   nir_variable *out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "SV_Target");
   out->data.location = FRAG_RESULT_DATA0;
   out->data.driver_location = 0;

   nir_store_var(&b, out, nir_load_var(&b, in_col), 0xf);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   return b.shader;
}

/*
 * vb_inst_vs: three attributes, one of them PER-INSTANCE.
 *
 *   TEXCOORD0  float2  per-vertex    position
 *   TEXCOORD1  float4  per-vertex    colour
 *   TEXCOORD2  float2  per-INSTANCE  offset, added to the position
 *
 * The offset comes from a different vertex buffer slot stepped once per
 * instance rather than once per vertex, so a driver that treated it as
 * per-vertex data would put every instance in the same place -- which is
 * exactly what g11 checks for.
 */
static nir_shader *
build_vb_inst_vs(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, opts,
                                                  "vb_inst_vs");

   nir_variable *in_pos = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec_type(2), "pos");
   in_pos->data.location = VERT_ATTRIB_GENERIC0;
   in_pos->data.driver_location = 0;

   nir_variable *in_col = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "col");
   in_col->data.location = VERT_ATTRIB_GENERIC0 + 1;
   in_col->data.driver_location = 1;

   nir_variable *in_off = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec_type(2), "off");
   in_off->data.location = VERT_ATTRIB_GENERIC0 + 2;
   in_off->data.driver_location = 2;

   nir_def *xy = nir_fadd(&b, nir_load_var(&b, in_pos),
                          nir_load_var(&b, in_off));
   nir_def *col = nir_load_var(&b, in_col);

   nir_def *pos = nir_vec4(&b, nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
                           nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f));

   nir_variable *out_col = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "var0");
   out_col->data.location = VARYING_SLOT_VAR0;
   out_col->data.driver_location = 0;
   nir_store_var(&b, out_col, col, 0xf);

   nir_variable *out_pos = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   out_pos->data.location = VARYING_SLOT_POS;
   out_pos->data.driver_location = 1;
   nir_store_var(&b, out_pos, pos, 0xf);

   b.shader->info.inputs_read = BITFIELD64_BIT(VERT_ATTRIB_GENERIC0) |
                                BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + 1) |
                                BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + 2);
   b.shader->info.outputs_written =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_POS);
   return b.shader;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr,
              "usage: %s <gs_input_vs|gs_point|tess_hs|tess_ds|mesh_quad|"
              "as_dispatch|vb_vs|vb_ps|vb_inst_vs> out.dxil\n",
              argv[0]);
      return 2;
   }

   glsl_type_singleton_init_or_ref();

   bool want_gs = !strcmp(argv[1], "gs_point");
   bool want_hs = !strcmp(argv[1], "tess_hs");
   bool want_ds = !strcmp(argv[1], "tess_ds");
   bool want_mesh = !strcmp(argv[1], "mesh_quad");
   bool want_amp = !strcmp(argv[1], "as_dispatch");
   bool want_vbvs = !strcmp(argv[1], "vb_vs");
   bool want_vbps = !strcmp(argv[1], "vb_ps");
   bool want_vbinst = !strcmp(argv[1], "vb_inst_vs");

   /* Mesh shaders do not exist below shader model 6.5; everything else
    * here is expressible in 6.2, which is what the corpus mostly uses. */
   enum dxil_shader_model sm =
      (want_mesh || want_amp) ? SHADER_MODEL_6_5 : SHADER_MODEL_6_2;

   nir_shader_compiler_options options;
   dxil_get_nir_compiler_options(&options, sm, 1 << 5, 1 << 5);
   nir_shader *nir = want_gs   ? build_gs_point_passthrough(&options)
                     : want_hs ? build_tess_hs(&options)
                     : want_ds ? build_tess_ds(&options)
                     : want_mesh ? build_mesh_quad(&options)
                     : want_amp ? build_as_dispatch(&options)
                     : want_vbvs ? build_vb_vs(&options)
                     : want_vbps ? build_vb_ps(&options)
                     : want_vbinst ? build_vb_inst_vs(&options)
                               : build_gs_input_vs(&options);

   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_all);
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_io,
            nir_var_shader_out |
               ((want_gs || want_hs || want_ds || want_vbvs || want_vbps ||
                 want_vbinst)
                   ? nir_var_shader_in
                   : 0),
            probe_type_size, 0);
   NIR_PASS(_, nir, nir_opt_dce);

   struct nir_to_dxil_options opts = {
      .environment = DXIL_ENVIRONMENT_VULKAN,
      .shader_model_max = sm,
      .validator_version_max = DXIL_VALIDATOR_1_6,
   };

   struct blob blob;
   blob_init(&blob);
   if (!nir_to_dxil(nir, &opts, NULL, &blob)) {
      fprintf(stderr, "nir_to_dxil failed\n");
      return 1;
   }

   FILE *f = fopen(argv[2], "wb");
   if (!f) {
      perror(argv[2]);
      return 1;
   }
   fwrite(blob.data, 1, blob.size, f);
   fclose(f);

   printf("wrote %s, %zu bytes\n", argv[2], blob.size);
   blob_finish(&blob);
   glsl_type_singleton_decref();
   return 0;
}
