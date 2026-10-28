/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "agx_nir_m5_mesh.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/u_math.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define M5_REFUSE() do { if (getenv("ASAHI_MESH_CONTRACT_DEBUG")) fprintf(stderr, "m5 mesh refuse: %s:%d\n", __func__, __LINE__); return false; } while (0)

struct mesh_layout {
   unsigned output_base, primitive_count, vertex_count, position;
   unsigned prim_vertices;
   unsigned plane_stride, planes, bytes;
   bool storage_probe;
   uint64_t primitive_audit_va;
   const struct agx_m5_mesh_varyings *varyings;
};

static uint64_t
primitive_audit_va(void)
{
   const char *enable = getenv("ASAHI_MESH_PP_AUDIT");
   const char *address = getenv("ASAHI_MESH_MARKER_VA");
   return enable && !strcmp(enable, "1") && address
      ? strtoull(address, NULL, 0) : 0;
}

static bool
primitive_audit_fan(void)
{
   const char *fan = getenv("ASAHI_MESH_PP_AUDIT_FAN");
   return fan && !strcmp(fan, "1");
}

static unsigned
primitive_audit_primitives(void)
{
   return primitive_audit_fan() ? 4 : 2;
}

static unsigned
primitive_audit_lanes(void)
{
   return primitive_audit_fan() ? 8 : 4;
}

static unsigned
primitive_audit_lane_first(void)
{
   return 16 * 4 * primitive_audit_primitives();
}

static unsigned
primitive_audit_fragment_first(void)
{
   return primitive_audit_lane_first() + 8 * 4 * primitive_audit_lanes();
}

static bool
native_primitive_address(void)
{
   const char *enable = getenv("ASAHI_MESH_NATIVE_PRIMITIVE_ADDRESS_DIAGNOSTIC");
   return enable && !strcmp(enable, "1");
}

static bool
native_page_carrier(void)
{
   const char *enable = getenv("ASAHI_MESH_NATIVE_PAGE_CARRIER");
   return enable && !strcmp(enable, "1");
}

static unsigned
page_carrier_words(void)
{
   return native_page_carrier() ? 1 : 2;
}

static void
primitive_capture(nir_builder *b, uint64_t va, unsigned first,
                  nir_def *record, unsigned words, nir_def **values)
{
   nir_def *address = nir_iadd(b, nir_imm_int64(b, va + 32 + 4 * first),
      nir_imul_imm(b, nir_u2u64(b, record), 4 * words));
   for (unsigned c = 0; c < words; ++c) {
      if (values[c])
         nir_global_atomic(b, 32, nir_iadd_imm(b, address, 4 * c),
            nir_u2u32(b, values[c]), .atomic_op = nir_atomic_op_xchg);
   }
}

static void
primitive_order(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_WORKGROUP,
      .memory_semantics = NIR_MEMORY_ACQ_REL,
      .memory_modes = nir_var_mem_global | nir_var_mem_shared);
}

static unsigned
prim_vertices(const nir_shader *s)
{
   switch (s->info.mesh.primitive_type) {
   case MESA_PRIM_POINTS:
      return 1;
   case MESA_PRIM_LINES:
      return 2;
   case MESA_PRIM_TRIANGLES:
      return 3;
   default:
      return 0;
   }
}

static struct mesh_layout
mesh_layout(const nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   const unsigned base = ALIGN_POT(s->info.shared_size, 64);
   




   const unsigned count = ALIGN_POT(prim_vertices(s) * s->info.mesh.max_primitives_out, 16);
   const unsigned position = count + 16;
   const unsigned stride = ALIGN_POT(s->info.mesh.max_vertices_out, 4) * 4;
   return (struct mesh_layout) {
      .primitive_audit_va = primitive_audit_va(),
      .prim_vertices = prim_vertices(s),
      .output_base = base, .primitive_count = base + count,
      .vertex_count = base + count + 4, .position = base + position,
      .plane_stride = stride, .planes = varyings->size,
      .bytes = ALIGN_POT(position + varyings->size * stride, 64),
      .storage_probe = getenv("ASAHI_MESH_PP_PROBE") != NULL &&
                       !strcmp(getenv("ASAHI_MESH_PP_PROBE"), "1"),
      .varyings = varyings,
   };
}

static unsigned
vertex_output_slot(nir_intrinsic_instr *i)
{
   return nir_intrinsic_io_semantics(i).location + nir_src_as_uint(i->src[2]);
}


static unsigned
first_plane(const struct agx_m5_mesh_varyings *varyings, unsigned slot)
{
   return slot == VARYING_SLOT_POS ? 0 : varyings->first[slot];
}

static nir_def *
widen_output(nir_builder *b, nir_intrinsic_instr *i, nir_def *value)
{
   


   if (value->bit_size == 32)
      return value;
   nir_alu_type type = nir_alu_type_get_base_type(nir_intrinsic_src_type(i));
   return nir_convert_to_bit_size(b, value, type, 32);
}

static bool
lower(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   const struct mesh_layout *layout = data;
   const unsigned output_base = layout->output_base;
   switch(i->intrinsic) {
   case nir_intrinsic_store_per_vertex_output: {
      b->cursor=nir_before_instr(&i->instr);
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,4);
      const unsigned slot=vertex_output_slot(i);
      if (slot != VARYING_SLOT_POS && !layout->varyings->components[slot])
         break;
      const unsigned first=first_plane(layout->varyings,slot);
      unsigned component=nir_intrinsic_component(i);
      nir_def *value=widen_output(b,i,i->src[0].ssa), *probe=NULL;
      if (slot == VARYING_SLOT_VAR0 && layout->storage_probe) {
         



         nir_def *storage=nir_load_mesh_primitive_storage_agx(b);
         probe=nir_vec4(b,nir_channel(b,storage,0),nir_channel(b,storage,1),
                        nir_channel(b,nir_load_workgroup_id(b),0),
                        nir_load_local_invocation_index(b));
      }
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *address=nir_iadd_imm(b,element,
            layout->position+layout->plane_stride*(first+component+c));
         nir_store_shared(b,probe ? nir_channel(b,probe,component+c)
                                  : nir_channel(b,value,c),address,
                            .write_mask=1,.align_mul=4);
         if (slot == VARYING_SLOT_CLIP_DIST0 || slot == VARYING_SLOT_CLIP_DIST1) {
            const unsigned clip = 4 * (slot - VARYING_SLOT_CLIP_DIST0) + component + c;
            if (clip < layout->varyings->clip_count) {
               nir_store_shared(b, nir_channel(b, value, c),
                  nir_iadd_imm(b, element, layout->position + layout->plane_stride *
                                              (layout->varyings->clip_first + clip)),
                  .write_mask = 1, .align_mul = 4);
            }
         }
      }
      break;
   }
   case nir_intrinsic_store_per_primitive_output: {
      b->cursor=nir_before_instr(&i->instr);
      const unsigned slot = vertex_output_slot(i);
      if (slot != VARYING_SLOT_PRIMITIVE_INDICES) {
         nir_def *storage = nir_pack_64_2x32(b, nir_load_mesh_primitive_storage_agx(b));
         nir_def *offset = nir_imul_imm(b, nir_u2u64(b, i->src[1].ssa),
                                      layout->varyings->primitive_stride_B);
         nir_def *value = widen_output(b, i, i->src[0].ssa);
         u_foreach_bit(c, nir_intrinsic_write_mask(i)) {
            nir_def *address = nir_iadd_imm(b, nir_iadd(b, storage, offset),
               4 * (layout->varyings->primitive_first[slot] +
                    nir_intrinsic_component(i) + c));
            nir_store_global(b, nir_channel(b, value, c), address,
                             .write_mask = 1, .align_mul = 4);
            if (layout->primitive_audit_va && c == 0) {
               nir_def *group = nir_channel(b, nir_load_workgroup_id(b), 0);
               const unsigned primitives = primitive_audit_primitives();
               nir_push_if(b, nir_iand(b, nir_ult_imm(b, group, 4),
                                      nir_ult_imm(b, i->src[1].ssa, primitives)));
               primitive_order(b);
               nir_def *post = nir_load_global(b, 1, 32, address,
                  .align_mul = 4, .access = ACCESS_VOLATILE);
               nir_def *fields[16] = {nir_imm_int(b, 0x50510001), group,
                  i->src[1].ssa, nir_load_local_invocation_index(b),
                  nir_u2u32(b, storage), nir_u2u32(b, nir_ushr_imm(b, storage, 32)),
                  nir_u2u32(b, address), nir_u2u32(b, nir_ushr_imm(b, address, 32)),
                  nir_imm_int(b, nir_intrinsic_component(i) + c), nir_channel(b, value, c), post,
                  nir_imm_int(b, 0x50510002)};
               primitive_capture(b, layout->primitive_audit_va, 0,
                  nir_iadd(b, nir_imul_imm(b, group, primitives), i->src[1].ssa), 16, fields);
               nir_pop_if(b, NULL);
            }
         }
         break;
      }
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,layout->prim_vertices);
      unsigned component=nir_intrinsic_component(i);
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *index=nir_channel(b,i->src[0].ssa,c);
         


         nir_def *packed=nir_u2u8(b,index);
         nir_store_shared(b,packed,nir_iadd_imm(b,element,output_base+component+c),
                            .write_mask=1,.align_mul=1);
      }
      break;
   }
   case nir_intrinsic_set_vertex_and_primitive_count: {
      b->cursor=nir_before_instr(&i->instr);
      nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
      nir_store_shared(b,nir_u2u16(b,i->src[1].ssa),nir_imm_int(b,layout->primitive_count),
                         .write_mask=1,.align_mul=2);
      nir_store_shared(b,nir_u2u16(b,i->src[0].ssa),nir_imm_int(b,layout->vertex_count),
                         .write_mask=1,.align_mul=2);
      nir_pop_if(b,NULL);
      break;
   }
   default:M5_REFUSE();
   }
   nir_instr_remove(&i->instr);
   return true;
}

#define USER_VARYINGS BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)
#define MESH_PSIZ_PLANE 4
#define MESH_CLIP_SLOTS (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1)
#define PRIMITIVE_SLOTS (USER_VARYINGS | VARYING_BIT_PRIMITIVE_ID | \
                         VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)

bool
agx_gen1_mesh_gather_varyings(const nir_shader *s,
                            struct agx_m5_mesh_varyings *varyings)
{
   memset(varyings, 0, sizeof(*varyings));
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_store_per_vertex_output &&
                i->intrinsic != nir_intrinsic_store_per_primitive_output)
               continue;
            if (!nir_src_is_const(i->src[2]))
               M5_REFUSE();
            const unsigned slot = vertex_output_slot(i);
            if (i->intrinsic == nir_intrinsic_store_per_primitive_output) {
               if (slot == VARYING_SLOT_PRIMITIVE_INDICES)
                  continue;
               const unsigned end = nir_intrinsic_component(i) +
                                    util_last_bit(nir_intrinsic_write_mask(i));
               if (slot >= 64 || !(PRIMITIVE_SLOTS & BITFIELD64_BIT(slot)) ||
                   end > 4 ||
                   (i->src[0].ssa->bit_size != 32 && i->src[0].ssa->bit_size != 16))
                  M5_REFUSE();
               varyings->primitive_comps[slot] =
                  MAX2(varyings->primitive_comps[slot], end);
               continue;
            }
            if (slot == VARYING_SLOT_POS)
               continue;
            if (slot == VARYING_SLOT_PSIZ &&
                s->info.mesh.primitive_type != MESA_PRIM_POINTS)
               continue;
            if (slot >= 64 || !((USER_VARYINGS | VARYING_BIT_PSIZ | MESH_CLIP_SLOTS) &
                                BITFIELD64_BIT(slot))) {
               if (getenv("ASAHI_MESH_CONTRACT_DEBUG"))
                  fprintf(stderr, "m5 mesh vertex slot %u\n", slot);
               M5_REFUSE();
            }
            const unsigned end = nir_intrinsic_component(i) +
                                 util_last_bit(nir_intrinsic_write_mask(i));
            if (end > 4)
               M5_REFUSE();
            varyings->components[slot] = MAX2(varyings->components[slot], end);
         }
      }
   }
   


   unsigned record = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!varyings->primitive_comps[slot])
         continue;
      varyings->primitive_first[slot] = record;
      record += varyings->primitive_comps[slot];
   }
   varyings->primitive_components = varyings->primitive_comps[VARYING_SLOT_VAR0];
   varyings->primitive_stride_B = 4 * record;
   if ((uint64_t)varyings->primitive_stride_B * s->info.mesh.max_primitives_out >
       UINT64_C(0x10000))
      M5_REFUSE();

   if (s->info.cull_distance_array_size)
      M5_REFUSE();
   const bool clip_written = varyings->components[VARYING_SLOT_CLIP_DIST0] ||
                             varyings->components[VARYING_SLOT_CLIP_DIST1];
   if (clip_written != (s->info.clip_distance_array_size != 0))
      M5_REFUSE();

   unsigned next = 4, slots = 0;
   if (varyings->components[VARYING_SLOT_PSIZ]) {
      varyings->first[VARYING_SLOT_PSIZ] = MESH_PSIZ_PLANE;
      next++;
   }
   if (varyings->primitive_comps[VARYING_SLOT_LAYER] ||
       varyings->primitive_comps[VARYING_SLOT_VIEWPORT]) {
      varyings->lv = true;
      varyings->lv_first = next++;
   }
   if (clip_written) {
      varyings->clip_first = next;
      varyings->clip_count = s->info.clip_distance_array_size;
      next += varyings->clip_count;
   }
   if (varyings->primitive_stride_B)
      varyings->components[AGX_MESH_PRIMITIVE_PAGE_SLOT] = page_carrier_words();
   for (unsigned slot = VARYING_SLOT_CLIP_DIST0; slot < 64; ++slot) {
      if (!varyings->components[slot])
         continue;
      varyings->first[slot] = next;
      next += varyings->components[slot];
      slots++;
   }
   varyings->size = next;
   varyings->dynamic = slots > 1 || varyings->lv;
   return true;
}


static bool
varyings_valid(const struct agx_m5_mesh_varyings *v)
{
   if (getenv("ASAHI_MESH_CONTRACT_DEBUG")) {
      fprintf(stderr, "m5 mesh layout size=%u clip=%u@%u lv=%u@%u:", v->size, v->clip_count, v->clip_first, v->lv, v->lv_first);
      for (unsigned slot = 0; slot < 64; ++slot)
         if (v->components[slot])
            fprintf(stderr, " %u:%u@%u", slot, v->components[slot], v->first[slot]);
      fprintf(stderr, "\n");
   }
   if (v->size < 4 || v->size > 4 + 128)
      M5_REFUSE();
   uint8_t used[4 + 128] = {0};
   unsigned user = 0;
   if (v->components[VARYING_SLOT_PSIZ]) {
      if (v->components[VARYING_SLOT_PSIZ] != 1 ||
          v->first[VARYING_SLOT_PSIZ] != MESH_PSIZ_PLANE)
         M5_REFUSE();
      used[MESH_PSIZ_PLANE] = 1;
      user++;
   }
   if (v->lv) {
      if (v->lv_first < 4 || v->lv_first >= v->size || used[v->lv_first]++)
         M5_REFUSE();
      user++;
   }
   if (v->clip_count) {
      if (v->clip_count > 8 || v->clip_first < 4 ||
          v->clip_first + v->clip_count > v->size)
         M5_REFUSE();
      for (unsigned c = 0; c < v->clip_count; ++c) {
         if (used[v->clip_first + c]++)
            M5_REFUSE();
      }
      user += v->clip_count;
   }
   unsigned hidden = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!v->components[slot])
         continue;
      if (slot == VARYING_SLOT_PSIZ)
         continue;
      if ((MESH_CLIP_SLOTS & BITFIELD64_BIT(slot)) && v->clip_count &&
          v->first[slot] >= v->clip_first &&
          v->first[slot] + v->components[slot] <= v->clip_first + v->clip_count) {
         hidden += v->components[slot];
         continue;
      }
      if ((!((USER_VARYINGS | MESH_CLIP_SLOTS) & BITFIELD64_BIT(slot)) &&
           !(slot == AGX_MESH_PRIMITIVE_PAGE_SLOT && v->primitive_stride_B != 0 &&
             v->components[slot] == page_carrier_words())) || v->components[slot] > 4 ||
          v->first[slot] < 4 || v->first[slot] + v->components[slot] > v->size)
         M5_REFUSE();
      for (unsigned c = 0; c < v->components[slot]; ++c) {
         if (used[v->first[slot] + c]++)
            M5_REFUSE();
      }
      user += v->components[slot];
   }
   return user + hidden + 4 == v->size;
}

bool
agx_nir_lower_gen1_mesh_triangle_probe(nir_shader *s)
{
   struct agx_m5_mesh_varyings varyings;
   if (!agx_gen1_mesh_gather_varyings(s, &varyings))
      M5_REFUSE();
   return agx_nir_lower_gen1_mesh(s, &varyings);
}

bool
agx_nir_lower_gen1_mesh(nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   unsigned threads=s->info.workgroup_size[0]*s->info.workgroup_size[1]*
                    s->info.workgroup_size[2];
   if(s->info.stage!=MESA_SHADER_MESH || s->info.shared_size>32768 ||
      s->info.workgroup_size_variable || threads==0 || threads>128 ||
      s->info.mesh.max_vertices_out==0 || s->info.mesh.max_vertices_out>256 ||
      s->info.mesh.max_primitives_out==0 || s->info.mesh.max_primitives_out>256 ||
      prim_vertices(s)==0 ||
      (s->info.per_primitive_outputs & ~(VARYING_BIT_PRIMITIVE_INDICES | PRIMITIVE_SLOTS)) ||
      (s->info.outputs_written & ~s->info.per_primitive_outputs) & ~(VARYING_BIT_POS|VARYING_BIT_PSIZ|MESH_CLIP_SLOTS|USER_VARYINGS) ||
      !varyings_valid(varyings))
      M5_REFUSE();
   if (s->info.mesh.primitive_type == MESA_PRIM_POINTS &&
       !varyings->components[VARYING_SLOT_PSIZ])
      M5_REFUSE();
   struct mesh_layout layout = mesh_layout(s, varyings);
   if (getenv("ASAHI_MESH_PP_AUDIT") &&
       !strcmp(getenv("ASAHI_MESH_PP_AUDIT"), "1") &&
       (!layout.primitive_audit_va || (layout.primitive_audit_va & 15) ||
        s->info.mesh.max_vertices_out > primitive_audit_lanes() ||
        s->info.mesh.max_primitives_out > primitive_audit_primitives() ||
        threads > primitive_audit_lanes()))
      M5_REFUSE();
   if (layout.output_base + layout.bytes > 32768)
      M5_REFUSE();
   unsigned physical_threads=MAX2(threads,MIN2(s->info.mesh.max_vertices_out,32));
   bool counts=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if(in->type!=nir_instr_type_intrinsic)continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            switch(i->intrinsic) {
            case nir_intrinsic_set_vertex_and_primitive_count:counts=true;break;
            case nir_intrinsic_barrier:
               if (physical_threads != threads) {
                  if (nir_intrinsic_execution_scope(i) == SCOPE_WORKGROUP)
                     nir_intrinsic_set_execution_scope(i, SCOPE_SUBGROUP);
                  if (nir_intrinsic_memory_scope(i) == SCOPE_WORKGROUP)
                     nir_intrinsic_set_memory_scope(i, SCOPE_SUBGROUP);
               }
               break;
            case nir_intrinsic_store_per_vertex_output: {
               if (!nir_src_is_const(i->src[2]))
                  M5_REFUSE();
               const unsigned slot=vertex_output_slot(i);
               const unsigned bits=i->src[0].ssa->bit_size;
               const unsigned end=nir_intrinsic_component(i)+
                                  util_last_bit(nir_intrinsic_write_mask(i));
               if(slot>=64 || (bits!=32 && bits!=16) || end>4)
                  M5_REFUSE();
               if(slot==VARYING_SLOT_POS) {
                  if(bits!=32)M5_REFUSE();
               } else if(slot==VARYING_SLOT_PSIZ) {
                  if(varyings->components[slot] && (bits!=32 || end>1))M5_REFUSE();
               } else if(!varyings->components[slot] ||
                         end>varyings->components[slot]) {
                  M5_REFUSE();
               }
               break;
            }
            case nir_intrinsic_store_per_primitive_output: {
               if (!nir_src_is_const(i->src[2]))
                  M5_REFUSE();
               const unsigned slot = vertex_output_slot(i);
               const unsigned end = nir_intrinsic_component(i) +
                                    util_last_bit(nir_intrinsic_write_mask(i));
               if (slot == VARYING_SLOT_PRIMITIVE_INDICES) {
                  if (end > prim_vertices(s) || nir_src_as_uint(i->src[2]) != 0)
                     M5_REFUSE();
               } else if (slot >= 64 || !varyings->primitive_stride_B ||
                          end > varyings->primitive_comps[slot] ||
                          (i->src[0].ssa->bit_size != 32 && i->src[0].ssa->bit_size != 16)) {
                  M5_REFUSE();
               }
               break;
            }
            case nir_intrinsic_load_per_vertex_output:
            case nir_intrinsic_load_per_primitive_output:
            case nir_intrinsic_launch_mesh_workgroups:M5_REFUSE();
            default:break;
            }
         }
      }
   }
   if(!counts)M5_REFUSE();
   if (physical_threads != threads) {
      


      nir_function_impl *entry=nir_shader_get_entrypoint(s);
      nir_foreach_block(block,entry) {
         nir_foreach_instr_safe(in,block) {
            if (in->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            nir_builder b=nir_builder_at(nir_before_instr(in));
            nir_def *replacement=NULL;
            if (i->intrinsic == nir_intrinsic_load_workgroup_size) {
               replacement=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                  s->info.workgroup_size[1],s->info.workgroup_size[2]);
            } else if (i->intrinsic == nir_intrinsic_load_local_invocation_id ||
                       i->intrinsic == nir_intrinsic_load_global_invocation_id) {
               nir_def *index=nir_load_local_invocation_index(&b);
               unsigned x=s->info.workgroup_size[0], y=s->info.workgroup_size[1];
               replacement=nir_vec3(&b,nir_umod_imm(&b,index,x),
                  nir_umod_imm(&b,nir_udiv_imm(&b,index,x),y),
                  nir_udiv_imm(&b,index,x*y));
               if (i->intrinsic == nir_intrinsic_load_global_invocation_id) {
                  

                  nir_def *size=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                     s->info.workgroup_size[1],s->info.workgroup_size[2]);
                  replacement=nir_iadd(&b,nir_imul(&b,nir_load_workgroup_id(&b),size),
                                       replacement);
                  replacement=nir_u2uN(&b,replacement,i->def.bit_size);
               }
            }
            if (replacement) {
               nir_def_rewrite_uses(&i->def,replacement);
               nir_instr_remove(in);
            }
         }
      }
      nir_cf_list body;
      nir_cf_extract(&body,nir_before_cf_list(&entry->body),
                            nir_after_cf_list(&entry->body));
      nir_builder b=nir_builder_at(nir_before_impl(entry));
      nir_if *active=nir_push_if(&b,nir_ult_imm(&b,
                                  nir_load_local_invocation_index(&b),threads));
      b.cursor=nir_cf_reinsert(&body,b.cursor);
      nir_pop_if(&b,active);
      s->info.workgroup_size[0]=physical_threads;
      s->info.workgroup_size[1]=s->info.workgroup_size[2]=1;
   }
   


   nir_shader_intrinsics_pass(s,lower,nir_metadata_none,&layout);
   nir_function_impl *impl=nir_shader_get_entrypoint(s);
   nir_builder b=nir_builder_at(nir_after_cf_list(&impl->body));
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
                  .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
   
   nir_push_if(&b,nir_ieq_imm(&b,nir_load_subgroup_id(&b),0));
   nir_def *p=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.primitive_count),.align_mul=2);
   nir_def *v=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.vertex_count),.align_mul=2);
   if (varyings->lv) {
      nir_def *storage = nir_pack_64_2x32(&b, nir_load_mesh_primitive_storage_agx(&b));
      nir_def *lane = nir_load_subgroup_invocation(&b);
      for (unsigned prim = 0; prim < s->info.mesh.max_primitives_out; prim += 32) {
         nir_def *index = nir_iadd_imm(&b, lane, prim);
         nir_push_if(&b, nir_ult(&b, index, nir_u2u32(&b, p)));
         nir_def *record = nir_iadd(&b, storage,
            nir_u2u64(&b, nir_imul_imm(&b, index, varyings->primitive_stride_B)));
         nir_def *layer = nir_imm_int(&b, 0), *viewport = nir_imm_int(&b, 0);
         if (varyings->primitive_comps[VARYING_SLOT_LAYER])
            layer = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, record,
               4 * varyings->primitive_first[VARYING_SLOT_LAYER]), .align_mul = 4);
         if (varyings->primitive_comps[VARYING_SLOT_VIEWPORT])
            viewport = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, record,
               4 * varyings->primitive_first[VARYING_SLOT_VIEWPORT]), .align_mul = 4);
         nir_def *packed = nir_pack_32_2x16_split(&b, nir_u2u16(&b, layer),
                                                  nir_u2u16(&b, viewport));
         for (unsigned k = 0; k < layout.prim_vertices; ++k) {
            nir_def *vertex = nir_u2u32(&b, nir_load_shared(&b, 1, 8,
               nir_iadd_imm(&b, nir_imul_imm(&b, index, layout.prim_vertices),
                            layout.output_base + k), .align_mul = 1));
            nir_store_shared(&b, packed,
               nir_iadd_imm(&b, nir_imul_imm(&b, vertex, 4),
                            layout.position + layout.plane_stride * varyings->lv_first),
               .write_mask = 1, .align_mul = 4);
         }
         nir_pop_if(&b, NULL);
      }
      nir_barrier(&b, .execution_scope = SCOPE_SUBGROUP, .memory_scope = SCOPE_SUBGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   }
   if (varyings->primitive_stride_B) {
      nir_def *storage = nir_load_mesh_primitive_storage_agx(&b);
      nir_def *lane = nir_load_subgroup_invocation(&b);
      for (unsigned vertex = 0; vertex < s->info.mesh.max_vertices_out; vertex += 32) {
         nir_def *index = nir_iadd_imm(&b, lane, vertex);
         nir_push_if(&b, nir_ult(&b, index, nir_u2u32(&b, v)));
         if (native_page_carrier()) {
            nir_def *page = nir_unpack_64_2x32_split_x(&b,
               nir_ushr_imm(&b, nir_pack_64_2x32(&b, storage), 16));
            nir_store_shared(&b, page,
               nir_iadd_imm(&b, nir_imul_imm(&b, index, 4),
                  layout.position + layout.plane_stride *
                     varyings->first[AGX_MESH_PRIMITIVE_PAGE_SLOT]),
               .write_mask = 1, .align_mul = 4);
         } else {
            for (unsigned c = 0; c < 2; ++c) {
               nir_store_shared(&b, nir_channel(&b, storage, c),
                  nir_iadd_imm(&b, nir_imul_imm(&b, index, 4),
                     layout.position + layout.plane_stride *
                        (varyings->first[AGX_MESH_PRIMITIVE_PAGE_SLOT] + c)),
                  .write_mask = 1, .align_mul = 4);
            }
         }
         if (layout.primitive_audit_va) {
            nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
            nir_push_if(&b, nir_ult_imm(&b, group, 4));
            primitive_order(&b);
            nir_def *shared = nir_iadd_imm(&b, nir_imul_imm(&b, index, 4),
               layout.position + layout.plane_stride *
                  varyings->first[AGX_MESH_PRIMITIVE_PAGE_SLOT]);
            nir_def *fields[8] = {nir_imm_int(&b, 0x50510010), index,
               nir_channel(&b, storage, 0), nir_channel(&b, storage, 1),
               nir_load_shared(&b, 1, 32, shared,
                  .align_mul = 4, .access = ACCESS_VOLATILE),
               nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, shared, layout.plane_stride),
                  .align_mul = 4, .access = ACCESS_VOLATILE), group,
               nir_imm_int(&b, 0x50510011)};
            primitive_capture(&b, layout.primitive_audit_va, primitive_audit_lane_first(),
               nir_iadd(&b, nir_imul_imm(&b, group, primitive_audit_lanes()), index), 8, fields);
            nir_pop_if(&b, NULL);
         }
         nir_pop_if(&b, NULL);
      }
      nir_barrier(&b, .execution_scope = SCOPE_SUBGROUP,
         .memory_scope = SCOPE_SUBGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
         .memory_modes = nir_var_mem_shared);
   }
   nir_allocate_mesh_uvb_agx(&b,p,v);
   nir_def *lane=nir_load_subgroup_invocation(&b);
   nir_def *primitive_words=nir_udiv_imm(&b,
      nir_iadd_imm(&b,nir_imul_imm(&b,nir_u2u32(&b,p),layout.prim_vertices),3),4);
   for (unsigned word=0;
        word<DIV_ROUND_UP(layout.prim_vertices*s->info.mesh.max_primitives_out,4);
        word+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,word),primitive_words));
      

      nir_store_mesh_uvb_agx(&b,nir_imm_ivec2(&b,word<<16,
         layout.output_base+4*word),.flags=1);
      nir_pop_if(&b,NULL);
   }
   


   nir_def *selector[4 + 128] = {NULL};
   if (varyings->dynamic) {
      for (unsigned slot = 0; slot < 64; ++slot) {
         if (!varyings->components[slot])
            continue;
         nir_def *base = nir_u2u32(&b, nir_load_uvs_index_agx(&b,
            .io_semantics = {.location = slot, .num_slots = 1}));
         for (unsigned c = 0; c < varyings->components[slot]; ++c)
            selector[varyings->first[slot] + c] = nir_iadd_imm(&b, base, c);
      }
   }
   for(unsigned vertex=0;vertex<s->info.mesh.max_vertices_out;vertex+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,vertex),nir_u2u32(&b,v)));
      for(unsigned plane=0;plane<layout.planes;plane++) {
         const unsigned address=layout.position+layout.plane_stride*plane+4*vertex;
         

         nir_def *descriptor = selector[plane]
            ? nir_vec2(&b, nir_ior_imm(&b, selector[plane], vertex << 16),
                       nir_imm_int(&b, address))
            : nir_imm_ivec2(&b, (vertex << 16) | plane, address);
         nir_store_mesh_uvb_agx(&b,descriptor,.flags=0);
      }
      nir_pop_if(&b,NULL);
   }
   nir_pop_if(&b,NULL);
   if (layout.primitive_audit_va)
      nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
   s->info.shared_size=layout.output_base+layout.bytes;
   s->info.outputs_written=0;s->info.per_primitive_outputs=0;
   nir_progress(true,impl,nir_metadata_none);
   return true;
}




static bool
lower_payload(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   (void)data;
   if (i->intrinsic==nir_intrinsic_barrier &&
       (nir_intrinsic_memory_modes(i)&nir_var_mem_task_payload)) {
      nir_intrinsic_set_memory_modes(i,
         (nir_intrinsic_memory_modes(i)&~nir_var_mem_task_payload)|nir_var_mem_global);
      return true;
   }
   bool store=i->intrinsic==nir_intrinsic_store_task_payload;
   if (!store && i->intrinsic!=nir_intrinsic_load_task_payload) M5_REFUSE();
   b->cursor=nir_before_instr(&i->instr);
   nir_def *pointer=nir_pack_64_2x32(b,nir_load_mesh_payload_address_agx(b));
   nir_def *offset=i->src[store ? 1 : 0].ssa;
   nir_def *addr=nir_iadd(b,pointer,nir_u2u64(b,
                            nir_iadd_imm(b,offset,nir_intrinsic_base(i))));
   if (store) {
      nir_store_global(b,i->src[0].ssa,addr,
         .write_mask=nir_intrinsic_write_mask(i),
         .align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i));
      nir_instr_remove(&i->instr);
   } else {
      nir_def_replace(&i->def,nir_load_global(b,i->def.num_components,
         i->def.bit_size,addr,.align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i)));
   }
   return true;
}

bool
agx_nir_lower_gen1_task_payload_probe(nir_shader *s)
{
   if (s->info.task_payload_size>4) M5_REFUSE();
   bool reads=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if (in->type!=nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            reads |= i->intrinsic==nir_intrinsic_load_task_payload;
            if (i->intrinsic==nir_intrinsic_task_payload_atomic ||
                i->intrinsic==nir_intrinsic_task_payload_atomic_swap ||
                i->intrinsic==nir_intrinsic_launch_mesh_workgroups_with_payload_deref)
               M5_REFUSE();
         }
      }
   }
   if (reads && s->info.stage==MESA_SHADER_MESH)
      M5_REFUSE();
   nir_shader_intrinsics_pass(s,lower_payload,nir_metadata_control_flow,NULL);
   return true;
}

static bool
lower_launch(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   if (i->intrinsic!=nir_intrinsic_launch_mesh_workgroups) M5_REFUSE();
   const unsigned grid=*(unsigned *)data;
   b->cursor=nir_before_instr(&i->instr);
   nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
   nir_store_shared(b,i->src[0].ssa,nir_imm_int(b,grid),
                       .write_mask=7,.align_mul=4);
   nir_pop_if(b,NULL);
   nir_instr_remove(&i->instr);
   return true;
}

bool
agx_nir_lower_gen1_task_probe(nir_shader *s)
{
   unsigned threads=s->info.workgroup_size[0]*s->info.workgroup_size[1]*
                    s->info.workgroup_size[2];
   if (s->info.stage!=MESA_SHADER_TASK || s->info.workgroup_size_variable ||
       threads<3 || threads>128 || s->info.shared_size>48 ||
       !agx_nir_lower_gen1_task_payload_probe(s)) M5_REFUSE();
   unsigned launches=0;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if (in->type==nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(in)->intrinsic==nir_intrinsic_launch_mesh_workgroups)
               launches++;
         }
      }
   }
   if (!launches) M5_REFUSE();
   unsigned grid=ALIGN_POT(s->info.shared_size,16);
   nir_shader_intrinsics_pass(s,lower_launch,nir_metadata_none,&grid);
   nir_builder b=nir_builder_at(nir_after_cf_list(&nir_shader_get_entrypoint(s)->body));
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
      .memory_semantics=NIR_MEMORY_ACQ_REL,
      .memory_modes=nir_var_mem_shared);

   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_push_if(&b,nir_ult_imm(&b,lane,3));
   nir_def *value=nir_load_shared(&b,1,32,
      nir_iadd_imm(&b,nir_imul_imm(&b,lane,4),grid),.align_mul=4);
   nir_store_object_childcount_agx(&b,value);
   nir_pop_if(&b,NULL);
   s->info.shared_size=64;
   nir_progress(true,b.impl,nir_metadata_none);
   return true;
}

static bool
lower_primitive_input(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_load_per_primitive_input)
      M5_REFUSE();
   b->cursor = nir_before_instr(&i->instr);
   const struct agx_m5_mesh_varyings *mesh = data;
   const unsigned location = nir_intrinsic_io_semantics(i).location;
   if (mesh && (location >= 64 || !mesh->primitive_comps[location])) {
      nir_def_rewrite_uses(&i->def, nir_imm_zero(b, i->def.num_components,
                                                 i->def.bit_size));
      nir_instr_remove(&i->instr);
      return true;
   }
   const unsigned record_first = mesh ? mesh->primitive_first[location] : 0;
   nir_def *primitive_cf = nir_load_coefficients_agx(b, nir_imm_int(b, 0),
      .component = 0, .interp_mode = INTERP_MODE_FLAT,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
   nir_def *base_low_cf = nir_load_coefficients_agx(b, nir_imm_int(b, 0),
      .component = 0, .interp_mode = INTERP_MODE_FLAT,
      .io_semantics = {.location = AGX_MESH_PRIMITIVE_PAGE_SLOT, .num_slots = 1});
   nir_def *base_high_cf = native_page_carrier() ? base_low_cf :
      nir_load_coefficients_agx(b, nir_imm_int(b, 0),
      .component = 1, .interp_mode = INTERP_MODE_FLAT,
      .io_semantics = {.location = AGX_MESH_PRIMITIVE_PAGE_SLOT, .num_slots = 1});
   nir_def *low = nir_channel(b, primitive_cf, 2);
   nir_def *base_low = nir_channel(b, base_low_cf, 2);
   nir_def *base_high = nir_channel(b, base_high_cf, 2);
   nir_def *base = native_page_carrier()
      ? nir_ishl_imm(b, nir_u2u64(b, base_low), 16)
      : nir_pack_64_2x32(b, nir_vec2(b, base_low, base_high));
   nir_def *address = native_primitive_address()
      ? nir_ior(b, nir_iand_imm(b, base, ~UINT64_C(0xffff)),
                nir_u2u64(b, nir_iand_imm(b, nir_ishl_imm(b, low, 2), 0xffff)))
      : nir_iadd(b, base, nir_u2u64(b, nir_iand_imm(b, low, 0xffff)));
   address = nir_iadd_imm(b, address,
                          4 * (record_first + nir_intrinsic_component(i)));
   nir_def *value = nir_load_global(b, i->def.num_components, 32, address,
                                  .align_mul = 4);
   const uint64_t va = primitive_audit_va();
   if (va) {
      nir_def *position = nir_load_frag_coord(b);
      nir_def *x = nir_f2u32(b, nir_channel(b, position, 0));
      nir_def *y = nir_f2u32(b, nir_channel(b, position, 1));
      nir_def *group = nir_ushr_imm(b, x, 4);
      nir_def *bottom = nir_ieq_imm(b, y, 60);
      nir_def *sample, *right = NULL, *below = NULL, *left = NULL;
      const bool fan = primitive_audit_fan();
      if (fan) {
         nir_def *column = nir_iand_imm(b, x, 15);
         nir_def *centre = nir_ieq_imm(b, column, 8);
         nir_def *middle = nir_ieq_imm(b, y, 32);
         nir_def *top = nir_iand(b, centre, nir_ieq_imm(b, y, 4));
         right = nir_iand(b, nir_ieq_imm(b, column, 14), middle);
         below = nir_iand(b, centre, bottom);
         left = nir_iand(b, nir_ieq_imm(b, column, 2), middle);
         sample = nir_ior(b, nir_ior(b, top, right), nir_ior(b, below, left));
      } else {
         sample = nir_iand(b, nir_ieq_imm(b, nir_iand_imm(b, x, 15), 8),
            nir_ior(b, nir_ieq_imm(b, y, 4), bottom));
      }
      nir_push_if(b, nir_iand(b, nir_ult_imm(b, group, 4), sample));
      nir_def *fields[32] = {nir_imm_int(b, 0x50510030), x, y, group,
         nir_channel(b, primitive_cf, 0), nir_channel(b, primitive_cf, 1), low,
         nir_channel(b, base_low_cf, 0), nir_channel(b, base_low_cf, 1), base_low,
         nir_channel(b, base_high_cf, 0), nir_channel(b, base_high_cf, 1), base_high,
         nir_u2u32(b, address), nir_u2u32(b, nir_ushr_imm(b, address, 32)),
         nir_channel(b, value, 0), nir_imm_int(b, nir_intrinsic_component(i)),
         nir_imm_int(b, 0x50510031)};
      nir_def *record = fan
         ? nir_iadd(b, nir_imul_imm(b, group, 4),
              nir_bcsel(b, right, nir_imm_int(b, 1),
                 nir_bcsel(b, below, nir_imm_int(b, 2),
                    nir_bcsel(b, left, nir_imm_int(b, 3), nir_imm_int(b, 0)))))
         : nir_iadd(b, nir_imul_imm(b, group, 2), nir_b2i32(b, bottom));
      primitive_capture(b, va, primitive_audit_fragment_first(), record, 32, fields);
      nir_pop_if(b, NULL);
   }
   if (i->def.bit_size != 32)
      value = nir_convert_to_bit_size(b, value,
         nir_alu_type_get_base_type(nir_intrinsic_dest_type(i)), i->def.bit_size);
   nir_def_rewrite_uses(&i->def, value);
   nir_instr_remove(&i->instr);
   return true;
}

static bool
fragment_inputs_supported(nir_shader *s, const struct agx_m5_mesh_varyings *mesh)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_load_per_primitive_input)
               continue;
            const unsigned location = nir_intrinsic_io_semantics(i).location;
            if (!nir_src_is_const(i->src[0]) || nir_src_as_uint(i->src[0]) != 0 ||
                nir_intrinsic_component(i) + i->def.num_components > 4 ||
                (i->def.bit_size != 32 && i->def.bit_size != 16))
               M5_REFUSE();
            if (location >= 64 || !(PRIMITIVE_SLOTS & BITFIELD64_BIT(location)))
               M5_REFUSE();
            if (!mesh && location != VARYING_SLOT_VAR0)
               M5_REFUSE();
         }
      }
   }
   return true;
}

bool
agx_nir_lower_gen1_mesh_fragment_layout(nir_shader *s,
                                      const struct agx_m5_mesh_varyings *mesh)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT || !mesh ||
       !fragment_inputs_supported(s, mesh))
      M5_REFUSE();
   nir_shader_intrinsics_pass(s, lower_primitive_input,
      primitive_audit_va() ? nir_metadata_none : nir_metadata_control_flow,
      (void *)mesh);
   if (primitive_audit_va())
      nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
   return true;
}

bool
agx_nir_lower_gen1_mesh_fragment(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      M5_REFUSE();
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_load_per_primitive_input)
               continue;
            if (nir_intrinsic_io_semantics(i).location != VARYING_SLOT_VAR0 ||
                !nir_src_is_const(i->src[0]) || nir_src_as_uint(i->src[0]) != 0 ||
                nir_intrinsic_component(i) + i->def.num_components > 4 ||
                (i->def.bit_size != 32 && i->def.bit_size != 16))
               M5_REFUSE();
         }
      }
   }
   nir_shader_intrinsics_pass(s, lower_primitive_input,
      primitive_audit_va() ? nir_metadata_none : nir_metadata_control_flow, NULL);
   if (primitive_audit_va())
      nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
   return true;
}
