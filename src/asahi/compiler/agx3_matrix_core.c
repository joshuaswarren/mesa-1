#include "agx3_matrix_core.h"
#include "agx3_binder.h"

bool
agx3_matrix_encode_mma8(enum agx_isa_encoding encoding,
                        const struct agx3_matrix_mma8_desc *d,
                        uint8_t out[14], uint8_t *length)
{
   if (!d)
      return false;
   unsigned type = d->input_type ? d->input_type : AGX3_MATRIX_INPUT_F16;
   unsigned align_mask = type == AGX3_MATRIX_INPUT_F16 ? 3 : 1;
   unsigned source_max = type == AGX3_MATRIX_INPUT_F32 ? 252 : 254;
   if (encoding != AGX_ISA_ENCODING_AGX3C || !out || !length ||
       (type != AGX3_MATRIX_INPUT_F16 && type != AGX3_MATRIX_INPUT_F32 &&
        type != AGX3_MATRIX_INPUT_BF16) ||
       (d->dest & align_mask) || d->dest > 252 ||
       (d->a & 1) || d->a > source_max || (d->b & 1) || d->b > source_max ||
       (!d->c_zero && ((d->c & align_mask) || d->c > 252)) ||
       (d->c_zero && (d->c || d->cache[2] || d->discard[2])))
      return false;

   for (unsigned i = 0; i < 3; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 1}, {42, 2}, {43, 1},
      {60, 2}, {67, 3}, {68, 61},
   };
   struct agx3_field_kv fields[23];
   unsigned count = 0;
   fields[count++] = (struct agx3_field_kv){26, type};
   fields[count++] = (struct agx3_field_kv){43, type};
   fields[count++] = (struct agx3_field_kv){2, d->dest};
   fields[count++] = (struct agx3_field_kv){11, d->dest_cache};
   fields[count++] = (struct agx3_field_kv){18, d->a};
   fields[count++] = (struct agx3_field_kv){35, d->b};
   fields[count++] = (struct agx3_field_kv){52, d->c};
   fields[count++] = (struct agx3_field_kv){59, d->c_zero ? 1 : 2};
   fields[count++] = (struct agx3_field_kv){27, d->cache[0]};
   fields[count++] = (struct agx3_field_kv){29, d->discard[0]};
   fields[count++] = (struct agx3_field_kv){44, d->cache[1]};
   fields[count++] = (struct agx3_field_kv){46, d->discard[1]};
   fields[count++] = (struct agx3_field_kv){61, d->cache[2]};
   fields[count++] = (struct agx3_field_kv){63, d->discard[2]};
   for (unsigned i = 0; i < 8; ++i)
      fields[count++] = (struct agx3_field_kv){69 + i,
                                             (d->wait_mask >> i) & 1};

   return agx3_bind_compose(encoding, "f32matmad8x8", 0, base,
                            sizeof(base) / sizeof(base[0]), fields, count,
                            out, length);
}

bool
agx3_matrix_encode_gemm16(enum agx_isa_encoding encoding,
                         const struct agx3_matrix_gemm16_desc *d,
                         uint8_t out[10], uint8_t *length)
{
   if (!d)
      return false;
   enum agx3_matrix_input_type b_type = d->b_input_type ? d->b_input_type : d->input_type;
   if (encoding != AGX_ISA_ENCODING_AGX3C || !out || !length ||
       (b_type != AGX3_MATRIX_INPUT_F16 && b_type != AGX3_MATRIX_INPUT_BF16) ||
       (d->dest & 15) || d->dest > 240 ||
       (d->a & 7) || d->a > 248 || (d->b & 7) || d->b > 248 ||
       (d->input_type != AGX3_MATRIX_INPUT_F16 &&
        d->input_type != AGX3_MATRIX_INPUT_BF16))
      return false;

   const struct agx3_field_kv base[] = {
      {1, 2}, {2, 2}, {9, 2}, {14, 1}, {20, 2}, {25, 1},
      {30, 1}, {31, 3}, {32, 70},
   };
   struct agx3_field_kv fields[29];
   unsigned count = 0;
   fields[count++] = (struct agx3_field_kv){0, d->dest};
   fields[count++] = (struct agx3_field_kv){3, d->dest_hint};
   fields[count++] = (struct agx3_field_kv){8, d->a};
   fields[count++] = (struct agx3_field_kv){10, d->input_type};
   fields[count++] = (struct agx3_field_kv){19, d->b};
   fields[count++] = (struct agx3_field_kv){21, b_type};
   fields[count++] = (struct agx3_field_kv){7, d->zero_accumulator};
   fields[count++] = (struct agx3_field_kv){15, d->transpose_a};
   fields[count++] = (struct agx3_field_kv){14, !d->preserve_a};
   fields[count++] = (struct agx3_field_kv){25, !d->preserve_b};
   fields[count++] = (struct agx3_field_kv){26, d->transpose_b};
   fields[count++] = (struct agx3_field_kv){33, d->field33};
   for (unsigned i = 0; i < 8; ++i)
      fields[count++] = (struct agx3_field_kv){40 + i,
                                             (d->field40_mask >> i) & 1};
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         fields[count++] = (struct agx3_field_kv){33 + i, 1};
   return agx3_bind_compose(encoding, "fGeMM16x16x16", 0, base,
                            sizeof(base) / sizeof(base[0]), fields, count,
                            out, length);
}
