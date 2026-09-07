/* SPDX-License-Identifier: MIT */
/* Exercise the production allocator's shuffle transaction. The independent
 * oracle applies its parallel copies to unique half-register payloads. */
#include "../agx_register_allocate.c"
#include <stdio.h>

struct fixture {
   agx_context shader;
   struct agx_shader_key key;
   struct ra_ctx ra;
   uint8_t widths[512];
   enum agx_size sizes[512];
   enum ra_class classes[512];
   uint16_t assignments[512], origins[512], values[AGX_NUM_REGS];
   BITSET_DECLARE(used, AGX_NUM_REGS);
   unsigned variables, high_water, memory_count;
};

static void
init(struct fixture *f, unsigned bound, unsigned reserved)
{
   memset(f, 0, sizeof(*f));
   f->key.dev.encoding =
      reserved == 2 ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3B;
   f->shader.key = &f->key;
   f->shader.any_cf = reserved > 0;
   f->shader.any_quad_divergent_shuffle = reserved == 2 || reserved == 4;
   f->shader.has_spill_pcopy_reserved = reserved == 8;
   f->ra = (struct ra_ctx){
      .shader = &f->shader, .ssa_to_reg = f->assignments,
      .ncomps = f->widths, .sizes = f->sizes, .classes = f->classes,
      .used_regs[RA_GPR] = f->used, .bound[RA_GPR] = bound,
      .count[RA_GPR] = &f->high_water, .count[RA_MEM] = &f->memory_count,
   };
   BITSET_SET_COUNT(f->used, 0, reserved);
}

static void
value(struct fixture *f, unsigned reg, unsigned width, enum agx_size size)
{
   unsigned ssa = f->variables++;
   assert(ssa < 511 && width && reg % width == 0);
   assert(!BITSET_TEST_COUNT(f->used, reg, width));
   f->widths[ssa] = width;
   f->sizes[ssa] = size;
   f->assignments[ssa] = f->origins[ssa] = reg;
   f->ra.reg_to_ssa[reg] = ssa;
   f->high_water = MAX2(f->high_water, reg + width);
   BITSET_SET_COUNT(f->used, reg, width);
   for (unsigned h = 0; h < width; ++h)
      f->values[reg + h] = 1 + ssa * 67 + h;
}

static void
check(struct fixture *f, unsigned count)
{
   f->widths[511] = count;
   f->sizes[511] = count > 1 ? AGX_SIZE_32 : AGX_SIZE_16;
   agx_index dest = {.type = AGX_INDEX_NORMAL, .value = 511,
                    .size = f->sizes[511]};
   struct util_dynarray copies = UTIL_DYNARRAY_INIT;
   unsigned result = assign_regs_by_copying(&f->ra, dest, NULL, &copies, 128);
   assert(result + count <= 128 && result % count == 0);
   uint16_t after[AGX_NUM_REGS];
   memcpy(after, f->values, sizeof(after));
   util_dynarray_foreach(&copies, struct agx_copy, c) {
      unsigned width = agx_size_align_16(c->src.size);
      assert(c->src.type == AGX_INDEX_REGISTER);
      assert(c->dest + width <= f->ra.bound[RA_GPR]);
      assert(c->src.value + width <= f->ra.bound[RA_GPR]);
      memcpy(after + c->dest, f->values + c->src.value, width * sizeof(uint16_t));
   }
   BITSET_DECLARE(assigned, AGX_NUM_REGS) = {0};
   BITSET_SET_COUNT(assigned, 0, reserved_size(&f->shader));
   BITSET_SET_COUNT(assigned, result, count);
   for (unsigned ssa = 0; ssa < f->variables; ++ssa) {
      unsigned reg = f->assignments[ssa], width = f->widths[ssa];
      assert(reg + width <= f->ra.bound[RA_GPR] && reg % width == 0);
      assert(!BITSET_TEST_COUNT(assigned, reg, width));
      BITSET_SET_COUNT(assigned, reg, width);
      assert(f->ra.reg_to_ssa[reg] == ssa);
      assert(!memcmp(after + reg, f->values + f->origins[ssa], width * sizeof(uint16_t)));
   }
   assert(!memcmp(assigned, f->used, sizeof(assigned)));
   assert(f->high_water > 128);
   util_dynarray_fini(&copies);
}

int
main(void)
{
   struct fixture f;
   init(&f, 256, 0);
   for (unsigned reg = 0; reg < 128; reg += 2)
      value(&f, reg, 2, AGX_SIZE_32);
   check(&f, 8);

   init(&f, 256, 0);
   for (unsigned reg = 0; reg < 128; reg += 16)
      value(&f, reg, 16, AGX_SIZE_32);
   check(&f, 8);

   /* Only eight free halves. A wide vector costs fewer scalar copies, but
    * evicting it would strand all free space in two unevictable blocks. */
   init(&f, 256, 1);
   value(&f, 8, 1, AGX_SIZE_16);
   for (unsigned reg = 10; reg < 16; ++reg)
      value(&f, reg, 1, AGX_SIZE_16);
   for (unsigned reg = 16; reg < 256; reg += 16)
      value(&f, reg, 16, AGX_SIZE_64);
   check(&f, 8);
   assert(f.assignments[511] == 8);

   unsigned rng = 0x1234567;
   for (unsigned run = 0; run < 256; ++run) {
      unsigned reserved[] = {0, 1, 2, 4, 8};
      unsigned count = 1u << (run % 4);
      unsigned bound = run & 1 ? 192 : 256;
      init(&f, bound, reserved[(run / 4) % ARRAY_SIZE(reserved)]);
      /* Fill the whole low bank and all but count halves of the high bank.
       * Power-of-two vectors independently vary in width and scalar type. */
      for (unsigned reg = reserved_size(&f.shader); reg < bound - count;) {
         rng = rng * 1664525u + 1013904223u;
         unsigned width = 1u << ((rng >> 16) % 5);
         while (reg % width || reg + width > bound - count)
            width /= 2;
         enum agx_size size = width >= 4 && (rng & 1) ? AGX_SIZE_64 :
                              width >= 2 ? AGX_SIZE_32 : AGX_SIZE_16;
         value(&f, reg, width, size);
         reg += width;
      }
      check(&f, count);
   }
   puts("TEX register bank: 259 pressure/permutation cases PASS");
   return 0;
}
