/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G4c acceptance driver: DXIL in, AGX machine code out, with the whole
 * path visible — bitstream, module, NIR, and the final disassembly. No
 * SPIR-V, no dxc, no external compiler is involved at run time.
 */

#include "d12_dxil.h"
#include "../d12_native_descriptor_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "poly/nir/poly_nir.h"
#include "agx_tilebuffer.h"
#include "agx_uvs.h"
#include "disasm.h"
#include "compiler/glsl_types.h"
#include "nir.h"

static const char *
stage_name(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX: return "vertex";
   case MESA_SHADER_FRAGMENT: return "fragment";
   case MESA_SHADER_COMPUTE: return "compute";
   case MESA_SHADER_GEOMETRY: return "geometry";
   case MESA_SHADER_TESS_CTRL: return "hull";
   case MESA_SHADER_TESS_EVAL: return "domain";
   default: return "?";
   }
}

/*
 * The shader kind lives in the DXBC container's DXIL part header (u16 at
 * offset 2) and mirrors DXC's -T profile. Map it to a Mesa stage.
 */
static bool
stage_from_option(const char *opt, mesa_shader_stage *out)
{
   if (!strcmp(opt, "ps") || !strcmp(opt, "fragment")) {
      *out = MESA_SHADER_FRAGMENT;
      return true;
   }
   if (!strcmp(opt, "vs") || !strcmp(opt, "vertex")) {
      *out = MESA_SHADER_VERTEX;
      return true;
   }
   if (!strcmp(opt, "cs") || !strcmp(opt, "compute")) {
      *out = MESA_SHADER_COMPUTE;
      return true;
   }
   return false;
}

int
main(int argc, char **argv)
{
   mesa_shader_stage stage = MESA_SHADER_FRAGMENT;
   bool stage_forced = false;
   const char *path = NULL;
   bool print_nir = false;
   /* Compile a vertex shader the way the driver does when a geometry
    * shader follows it: as the COMPUTE program that fills the vertex
    * output buffer.
    */
   bool sw_vs = false;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-s") && i + 1 < argc) {
         if (!stage_from_option(argv[++i], &stage)) {
            fprintf(stderr, "unknown stage %s\n", argv[i]);
            return 2;
         }
         stage_forced = true;
      } else if (!strcmp(argv[i], "-n")) {
         print_nir = true;
      } else if (!strcmp(argv[i], "-swvs")) {
         sw_vs = true;
      } else {
         path = argv[i];
      }
   }

   if (!path) {
      fprintf(stderr, "usage: %s [-s ps|vs|cs] [-n] shader.dxil\n", argv[0]);
      return 2;
   }

   /* The driver gets this from agx_open_device; a tool with no device
    * must do it itself before anything touches glsl types (scratch and
    * shared-memory lowering both do).
    */
   glsl_type_singleton_init_or_ref();

   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "cannot open %s\n", path);
      return 2;
   }

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);

   uint8_t *data = malloc(size);
   if (!data || fread(data, 1, size, f) != (size_t)size) {
      fprintf(stderr, "cannot read %s\n", path);
      return 2;
   }
   fclose(f);

   /* 1. container -> bitcode (and the stage it declares) */
   const uint8_t *bc;
   size_t bc_size;
   const char *why = NULL;
   if (!d12_dxil_find_bitcode(data, size, &bc, &bc_size, &why)) {
      /*
       * No DXIL: Shader Model 5.x, which the DXBC front end handles.
       * Doing it here as well as in the PSO path matters more than it
       * looks -- this tool needs NO GPU, so it is the only way to test
       * shader translation while the hardware is unavailable.
       */
      uint32_t ver = 0;
      const uint8_t *chunk;
      size_t chunk_size;
      if (d12_dxbc_find_shader(data, size, &chunk, &chunk_size, &ver)) {
         static const mesa_shader_stage kind_to_stage[] = {
            MESA_SHADER_FRAGMENT, MESA_SHADER_VERTEX,   MESA_SHADER_GEOMETRY,
            MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL, MESA_SHADER_COMPUTE,
         };
         unsigned kind = ver >> 16;
         mesa_shader_stage st =
            kind < ARRAY_SIZE(kind_to_stage) ? kind_to_stage[kind] : stage;

         printf("%s: DXBC sm%u.%u, stage %s\n", path, (ver >> 4) & 0xf,
                ver & 0xf, _mesa_shader_stage_to_string(st));

         const char *dxbc_err = "unknown";
         struct d12_binding_map dmap;
         memset(&dmap, 0, sizeof(dmap));
         nir_shader *dn =
            d12_dxbc_to_nir(data, size, st, &dmap,
                            D12_NATIVE_DESCRIPTOR_GEN7, NULL, &dxbc_err);
         if (!dn) {
            printf("DXBC->NIR FAILED: %s\n", dxbc_err);
            return 1;
         }
         printf("DXBC->NIR OK\n");
         if (getenv("D3D12AGX_PRINT_NIR"))
            nir_print_shader(dn, stdout);
         return 0;
      }

      fprintf(stderr, "%s: %s\n", path, why);
      return 1;
   }

   /* The container names its stage; honour it or refuse. Defaulting on
    * an unmappable kind is how a mesh shader once "passed" as a fragment
    * shader.
    */
   mesa_shader_stage container_stage;
   if (!stage_forced) {
      if (d12_dxil_container_stage(data, size, &container_stage)) {
         stage = container_stage;
      } else {
         uint32_t kind;
         if (d12_dxil_container_kind(data, size, &kind)) {
            fprintf(stderr, "%s: unsupported shader kind %u (%s)\n", path,
                    kind, d12_dxil_kind_name(kind));
            return 1;
         }
         fprintf(stderr, "%s: container does not declare a stage\n", path);
         return 1;
      }
   }

   if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_VERTEX &&
       stage != MESA_SHADER_COMPUTE && stage != MESA_SHADER_GEOMETRY &&
       stage != MESA_SHADER_TESS_CTRL && stage != MESA_SHADER_TESS_EVAL) {
      fprintf(stderr, "%s: stage %s has no AGX backing here\n", path,
              _mesa_shader_stage_to_string(stage));
      return 1;
   }

   /* 2. bitcode -> module */
   struct d12_module mod;
   if (!d12_dxil_module_parse(&mod, bc, bc_size)) {
      fprintf(stderr, "module parse failed: %s\n",
              mod.error ? mod.error : "unknown");
      d12_dxil_module_finish(&mod);
      return 1;
   }

   printf("%s: %s shader, %u types, %u values, %u functions\n", path,
          stage_name(stage), mod.num_types, mod.num_values,
          mod.num_functions);

   for (unsigned i = 0; print_nir && i < mod.num_functions; i++) {
      printf("  fn %u: %-28s %s val=%u insts=%u%s\n", i,
             mod.functions[i].name ? mod.functions[i].name : "(none)",
             mod.functions[i].is_declaration ? "decl" : "DEF ",
             mod.functions[i].value_index, mod.functions[i].num_insts,
             (mod.has_patch_constant && i == mod.patch_constant_func)
                ? "  <-- patch constant" : "");
   }

   /* 3. module -> NIR (no SPIR-V anywhere in here) */
   const char *error = NULL;
   /* No root signature here, so the map comes from what the module itself
    * declares. The driver never does this: there, both sides read the
    * assignment stored on the root signature.
    */
   struct d12_binding_map map;
   d12_binding_map_from_module(&mod, &map);

   nir_shader *nir =
      d12_dxil_to_nir(&mod, stage, &map, D12_NATIVE_DESCRIPTOR_GEN7, &error);
   if (!nir) {
      fprintf(stderr, "DXIL->NIR failed: %s\n", error);
      d12_dxil_module_finish(&mod);
      return 1;
   }

   printf("DXIL->NIR OK\n");
   if (print_nir)
      nir_print_shader(nir, stdout);

   /* 4. Lowering, then NIR -> AGX.
    *
    * The AGX backend does not consume store_output for fragment shaders
    * directly: colour writes become tile-memory stores, which is what
    * agx_nir_lower_tilebuffer does (the same pass Honeykrisp and the
    * background/end-of-tile builders run). Skipping this is exactly the
    * failure mode where the backend reports "Unhandled intrinsic".
    */
   NIR_PASS(_, nir, agx_nir_lower_texture);
   /* Multisampled image stores lower the same way in every stage; the
    * FS-only placement left compute MS stores to hit the backend raw.
    */
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   /*
    * Geometry and tessellation have no hardware stages on AGX: they run
    * as compute, through the same lowering Honeykrisp uses
    * (src/poly/nir). The lowering REWRITES info.stage to COMPUTE, so it
    * has to happen before anything reads the stage back.
    *
    * A geometry shader expands into four programs -- count, the main
    * body, a copy shader and a pre-GS pass. Only the main body is
    * compiled here; running the whole set is the runtime's job.
    */
   struct poly_gs_info gs_info = {0};
   nir_shader *gs_count = NULL, *gs_copy = NULL, *pre_gs = NULL;

   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      NIR_PASS(_, nir, poly_nir_lower_gs, &gs_count, &gs_copy, &pre_gs,
               &gs_info);
   } else if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, poly_nir_lower_tcs);
   } else if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      /* to_hw_vs: the domain shader becomes the hardware vertex shader
       * when nothing follows it.
       */
      NIR_PASS(_, nir, poly_nir_lower_tes, true);
   }

   if (sw_vs && nir->info.stage == MESA_SHADER_VERTEX) {
      printf("sw_vs: outputs_written before lowering = 0x%llx\n",
             (unsigned long long)nir->info.outputs_written);
      NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->xfb_info = NULL;
      NIR_PASS(_, nir, d12_dxil_lower_sw_vs);
   }

   agx_preprocess_nir(nir);
   NIR_PASS(_, nir, d12_dxil_lower_root);

   if (sw_vs) {
      printf("\n=== software VS (compute) ===\n");
      nir_print_shader(nir, stdout);
   }

   struct agx_tilebuffer_layout tib = {0};
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      enum pipe_format formats[8] = {PIPE_FORMAT_R8G8B8A8_UNORM};
      tib = agx_build_tilebuffer_layout(formats, 1, 1, false);

      unsigned bindless_base = 0;
      NIR_PASS(_, nir, agx_nir_lower_tilebuffer, &tib, NULL, &bindless_base,
               NULL, NULL, false, false);
      /* Depth/stencil-ref writes become zs_emit, a coverage write
       * becomes demote_samples, and discards become sample_mask writes.
       * hk orders these the same way (hk_shader.c): zs_emit before the
       * sample-mask lowering, everything before MSAA lowering.
       */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
      NIR_PASS(_, nir, agx_nir_lower_sample_mask);
      NIR_PASS(_, nir, agx_nir_lower_monolithic_msaa, 1);
      /* The passes above can introduce driver-owned sysvals
       * (api_sample_mask, shader_part_tests_zs); rerun the lowering that
       * resolves them. No-op when none appeared.
       */
      NIR_PASS(_, nir, d12_dxil_lower_root);
      NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, nir, agx_nir_lower_texture);
   }

   /* Vertex outputs go through the unified vertex store, not plain
    * store_output — the vertex-stage analogue of the tilebuffer lowering.
    */
   struct agx_unlinked_uvs_layout uvs = {0};
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* Real vertex attributes arrive through the prolog ABI: the main
       * shader reads registers the prolog exported, so load_input must
       * become load_exported_agx here (see d12_pipeline.c). A
       * DRAW with a nonempty input layout also needs an attribute-
       * fetching prolog keyed from that layout, which does not exist
       * yet: the static_vi prolog only fixes up vertex/instance IDs.
       *
       * These braces are load-bearing. Without them only the first pass
       * was guarded and every stage got the vertex lowerings -- so this
       * tool, which is what the corpus figure is measured with, was
       * compiling a shader the driver never builds. That is the
       * shader-key trap in docs/findings.md, one level worse: not a
       * different key, different PASSES.
       */
      NIR_PASS(_, nir, agx_nir_lower_vs_input_to_prolog);
      NIR_PASS(_, nir, agx_nir_lower_uvs, &uvs);

      /* Same constant lowering the driver does (see d12_pipeline.c). */
      struct agx_varyings_vs linked = {0};
      agx_assign_uvs(&linked, &uvs, 0, 0);
      NIR_PASS(_, nir, d12_dxil_lower_uvs_index, linked.slots);
   }

   /*
    * This tool has no open device, so the device key is filled in for the
    * hardware the campaign targets rather than gathered. It has to be
    * filled in at all: the AGX backend asserts on an unset coherency
    * tristate the moment a shader uses a global atomic, and the assert
    * fires in the packer, a long way from anything that looks like a
    * missing device property.
    */
   struct agx_shader_key key = {
      .no_stop = false,
      /* Must match d12_pipeline.c. The driver keeps constants in the
       * binary; promoting them here would disassemble a shader the driver
       * never runs, with different uniform registers.
       */
      .promote_constants = false,
      /* The driver reserves the low uniforms for the root table and the
       * texture heap, so the preamble must allocate above them. Without
       * this the tool compiles a DIFFERENT shader from the one the driver
       * runs -- one whose preamble writes the registers the driver
       * publishes bindings in -- and its disassembly stops being evidence
       * about the driver.
       */
      .reserved_preamble = D12_UNIFORM_RESERVED,
      /* Large shaders spill; without this the allocator asserts that
       * internal shaders are unspillable. */
      .has_scratch = true,
      .dev = {
         /* M-series with one cluster per die: no G13X coherency. */
         .needs_g13x_coherency = U_TRISTATE_NO,
         .soft_fault = false,
      },
   };

   if (print_nir) {
      printf("\n=== NIR after AGX lowering ===\n");
      nir_print_shader(nir, stdout);
   }

   /*
    * A geometry shader is not one program. poly_nir_lower_gs splits it
    * into a count pass, a pre-GS pass, the main body and a copy shader,
    * and the driver must run all four. Compiling only the main body
    * hides failures in the other three -- and makes a trivial main body
    * look like a broken shader.
    */
   if (gs_count || gs_copy || pre_gs) {
      /* The lowering already computes everything the runtime has to
       * honour; printing it is what turns "geometry shaders compile"
       * into a statement about how much dispatch machinery each one
       * actually needs. Without XFB and without GS queries the count,
       * prefix-sum and pre-GS programs are never dispatched at all
       * (hk_cmd_draw.c:1535), so `xfb`/`prefix_sum` are the fields that
       * separate a cheap shader from an expensive one.
       */
      static const char *const shape_name[] = {
         [POLY_GS_SHAPE_STATIC_PER_INSTANCE] = "STATIC_PER_INSTANCE",
         [POLY_GS_SHAPE_STATIC_PER_PRIM] = "STATIC_PER_PRIM",
         [POLY_GS_SHAPE_STATIC_INDEXED] = "STATIC_INDEXED",
         [POLY_GS_SHAPE_DYNAMIC_INDEXED] = "DYNAMIC_INDEXED",
      };
      const char *sh = (gs_info.shape < ARRAY_SIZE(shape_name) &&
                        shape_name[gs_info.shape])
                          ? shape_name[gs_info.shape]
                          : "?";

      /* u_prim_name lives in gallium's aux, which this build does not
       * link (-Dgallium-drivers= is empty). Only the modes a GS can
       * output are reachable here.
       */
      static const char *const prim_name[] = {
         [MESA_PRIM_POINTS] = "POINTS",
         [MESA_PRIM_LINES] = "LINES",
         [MESA_PRIM_LINE_STRIP] = "LINE_STRIP",
         [MESA_PRIM_TRIANGLES] = "TRIANGLES",
         [MESA_PRIM_TRIANGLE_STRIP] = "TRIANGLE_STRIP",
      };
      const char *pm = (gs_info.mode < ARRAY_SIZE(prim_name) &&
                        prim_name[gs_info.mode])
                          ? prim_name[gs_info.mode]
                          : "?";

      printf("gs_info: mode=%s shape=%s max_indices=%u count_words=%u "
             "xfb=%d prefix_sum=%d multistream=%d\n",
             pm, sh, gs_info.max_indices, gs_info.count_words, gs_info.xfb,
             gs_info.prefix_sum, gs_info.multistream);

      struct {
         const char *name;
         nir_shader *s;
      } extra[] = {
         {"gs count", gs_count},
         {"gs pre", pre_gs},
         {"gs rast", gs_copy},
      };

      for (unsigned i = 0; i < ARRAY_SIZE(extra); i++) {
         if (!extra[i].s)
            continue;

         /* The third program is the RASTERIZATION shader: it reads the
          * emitted vertex buffer and feeds the rasterizer, so it is a
          * hardware vertex shader and needs the vertex output lowering,
          * exactly as hk_lower_hw_vs does for it (hk_shader.c).
          */
         if (extra[i].s->info.stage == MESA_SHADER_VERTEX ||
             !strcmp(extra[i].name, "gs rast")) {
            struct agx_unlinked_uvs_layout euvs = {0};
            extra[i].s->info.stage = MESA_SHADER_VERTEX;
            NIR_PASS(_, extra[i].s, agx_nir_lower_uvs, &euvs);

            struct agx_varyings_vs elinked = {0};
            agx_assign_uvs(&elinked, &euvs, 0, 0);
            NIR_PASS(_, extra[i].s, d12_dxil_lower_uvs_index, elinked.slots);
         }

         agx_preprocess_nir(extra[i].s);
         NIR_PASS(_, extra[i].s, d12_dxil_lower_root);

         if (print_nir) {
            printf("\n=== %s program ===\n", extra[i].name);
            nir_print_shader(extra[i].s, stdout);
         }

         struct agx_shader_key ekey = key;
         struct agx_shader_part epart;
         memset(&epart, 0, sizeof(epart));
         agx_compile_shader_nir(extra[i].s, &ekey, &epart);

         printf("  %s: %u bytes, %u GPRs\n", extra[i].name,
                epart.info.binary_size, epart.info.nr_gprs);
         free(epart.binary);
      }
   }

   struct agx_shader_part part;
   memset(&part, 0, sizeof(part));
   agx_compile_shader_nir(nir, &key, &part);

   if (!part.binary || !part.info.binary_size) {
      fprintf(stderr, "AGX compile produced no code\n");
      d12_dxil_module_finish(&mod);
      return 1;
   }

   printf("NIR->AGX OK: %u bytes total, %u GPRs, main at 0x%x%s\n",
          part.info.binary_size, part.info.nr_gprs, part.info.main_offset,
          part.info.has_preamble ? " (has preamble)" : "");

   /* The binary is rodata + optional preamble + main; disassemble the code
    * sections rather than decoding the constant pool as instructions.
    */
   if (part.info.has_preamble) {
      printf("\n; AGX preamble (offset 0x%x)\n", part.info.preamble_offset);
      agx2_disassemble(part.binary + part.info.preamble_offset,
                       part.info.main_offset > part.info.preamble_offset
                          ? part.info.main_offset - part.info.preamble_offset
                          : part.info.binary_size - part.info.preamble_offset,
                       stdout);
   }

   printf("\n; AGX main (offset 0x%x) — from %s\n", part.info.main_offset,
          path);
   agx2_disassemble(part.binary + part.info.main_offset,
                    part.info.binary_size - part.info.main_offset, stdout);

   free(part.binary);
   d12_dxil_module_finish(&mod);
   free(data);
   return 0;
}
