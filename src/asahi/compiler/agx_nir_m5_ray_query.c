/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_nir_m5_ray_query.h"
#include "agx3_rt_m5_core.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_debug.h"

static bool
m5_candidate_api(void)
{
   return debug_get_bool_option("ASAHI_M5_RQ_CANDIDATE_API", false);
}

static bool
m5_candidate_mode(void)
{
   return m5_candidate_api() ||
          debug_get_bool_option("ASAHI_M5_RQ_CANDIDATE_PROBE", false);
}

/* Keep native geometry opacity so opaque triangles commit in the
 * intersector, while transparent triangles continue through the native
 * candidate queue and Vulkan application any-hit code. This is the
 * configuration the M5 RT gate qualifies (455 cases). */
static bool
m5_native_opaque(void)
{
   return m5_candidate_api();
}

static bool
m5_defer_completion(nir_builder *b)
{
   return m5_candidate_api() &&
          b->shader->info.stage == MESA_SHADER_COMPUTE &&
          b->shader->info.ray_queries == 1 &&
          !getenv("ASAHI_M5_RQ_STATE_ADDRESS") &&
          !getenv("ASAHI_M5_RQ_ZERO_SPAN_DIAG_ADDRESS") &&
          debug_get_bool_option("ASAHI_M5_RQ_DEFER_COMPLETION", true);
}

/* The live raw IFT50 callback stores its final 48-bit instance pointer at
 * driver+8640+6*lane. Its complete per-clique tail is 8832 bytes (276*32),
 * which overlaps the next clique with the 272-entry candidate stride, so the
 * candidate profile uses 276 (and the matching packed SCS layout in
 * hk_cmd_dispatch.c/hk_queue.c). */
static bool
m5_raw50_layout_probe(void)
{
   return m5_candidate_mode();
}

static unsigned
m5_candidate_driver_stride(void)
{
   return m5_raw50_layout_probe() ? 276 : 272;
}

static nir_def *
m5_native_opaque_eligible(nir_builder *b, nir_def *flags)
{
   /* The captured native query policy implements geometry/instance opacity,
    * but these Vulkan policies require our candidate filter or first-hit
    * ordering. Face culling remains encoded in the native ray packet. */
   /* Owner replay counts application candidates. Native opaque pruning can
    * alter that prefix, so qualify this policy only for a lexical single
    * query until the mixed-owner prefix protocol is separately established. */
   /* Native RT-pipeline shaders arrive here after lowering to COMPUTE.
    * Graphics query resources and callback layouts require their own
    * hardware qualification before using this optimization. */
   return m5_native_opaque() && b->shader->info.stage == MESA_SHADER_COMPUTE &&
          b->shader->info.ray_queries == 1
      ? nir_ieq_imm(b, nir_iand_imm(b, flags, 2 | 4 | 0x40 | 0x100), 0)
      : nir_imm_false(b);
}

nir_def *
agx_nir_m5_candidate_count(nir_builder *b, nir_def *driver, nir_def *lane)
{
   assert(driver->bit_size == 32 && lane->bit_size == 32);
   /* RIA3 payload segment14: the four-bit count starts at bit21. */
   /* The captured IFT runtime writes its query count at 1728. The newer
    * query-once compiler witness uses 1600: its payload layout is 128 bytes
    * smaller before the candidate queue. Operand shape is the native +0x6ac
    * two-register selector-2 form. */
   nir_def *control = nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
      .base = 1728, .component = 2);
   return nir_iand_imm(b, nir_ushr_imm(b, control, 21), 15);
}

struct agx_m5_ray_candidate
agx_nir_m5_load_candidate(nir_builder *b, nir_def *driver, nir_def *lane,
                          nir_def *index)
{
   assert(driver->bit_size == 32 && lane->bit_size == 32 &&
          index->bit_size == 32);
   /* The native program multiplies the lane by200 and its candidate cursor
    * by20. Each record is a 48-bit compressed address, 16-bit flags, two
    * barycentrics and distance. Do not decode it as a committed hit packet. */
   nir_def *offset = nir_iadd(b, nir_imul_imm(b, lane, 200),
                                nir_imul_imm(b, index, 20));
   if (debug_get_bool_option("ASAHI_M5_RQ_CANDIDATE_FLAT", false)) {
      nir_def *address = nir_iadd(b, driver, offset);
      nir_def *prefix = nir_load_scs_base_agx(b, 2, 32, address,
         .base = 1856, .format = PIPE_FORMAT_R32_UINT, .component = 4);
      return (struct agx_m5_ray_candidate){
         .compressed_address = nir_iand_imm(b, nir_pack_64_2x32(b, prefix),
                                            UINT64_C(0xffffffffffff)),
         .flags = nir_ushr_imm(b, nir_channel(b, prefix, 1), 16),
         .barycentrics = nir_load_scs_base_agx(b, 2, 32, address,
            .base = 1864, .format = PIPE_FORMAT_R32_UINT, .component = 4),
         .distance = nir_load_scs_base_agx(b, 1, 32, address,
            .base = 1872, .format = PIPE_FORMAT_R32_UINT, .component = 4),
      };
   }
   /* Native +0x7d0..+0x82e uses selector 0 with the driver in f16 and
    * per-lane record offset in f31. Preserve that operand form. */
   nir_def *prefix = nir_load_scs_register_agx(b, 2, 32, offset, driver,
      .base = 1856, .format = PIPE_FORMAT_R32_UINT, .component = 0);
   return (struct agx_m5_ray_candidate){
      .compressed_address = nir_iand_imm(b, nir_pack_64_2x32(b, prefix),
                                         UINT64_C(0xffffffffffff)),
      .flags = nir_ushr_imm(b, nir_channel(b, prefix, 1), 16),
      .barycentrics = nir_load_scs_register_agx(b, 2, 32, offset, driver,
         .base = 1864, .format = PIPE_FORMAT_R32_UINT, .component = 0),
      .distance = nir_load_scs_register_agx(b, 1, 32, offset, driver,
         .base = 1872, .format = PIPE_FORMAT_R32_UINT, .component = 0),
   };
}

/* Private qualification of native zero/FTZ maximum-distance admission.
 * API bounds stay in the query variables; only the traversal packet changes.
 * Root hardware receipts distinguish positive zero-span rays (admitted) from
 * zero/subnormal maxima (no native candidate), with FLT_MIN admitted. */
static nir_def *
m5_native_admission_tmax(nir_builder *b, nir_def *api_tmax)
{
   if (!m5_candidate_api())
      return api_tmax;

   /* Classify the bit representation, avoiding a floating comparison that
    * could itself flush a subnormal. Preserve negative finite values, NaNs,
    * infinity, and every normal maximum bit-for-bit. Both signed zeros are
    * valid nonnegative endpoints. */
   nir_def *zero_or_positive_subnormal = nir_ior(b,
      nir_ult_imm(b, api_tmax, 0x00800000u),
      nir_ieq_imm(b, api_tmax, 0x80000000u));
   return nir_bcsel(b, zero_or_positive_subnormal,
                   nir_imm_int(b, 0x00800000u), api_tmax);
}

/* Request the next candidate batch exactly as the native runtime does. The
 * RIA runtime IR agx.intersection_query.next (work/neo-rt-20261011/native-ria2/
 * native-rt-ir) performs, before every agc.continue.ray: cursor=0, control
 * word &= ~0x1e00001 (count bits 21..24 and the active bit0), both 32-bit words
 * of the per-lane completion record (its payload segment 7; driver+256+8*lane
 * here) stored as zero, and the committed distance published to the ray
 * packet when it changed. The record's bit 6 is only its busy poll; the
 * runtime never derives the resume decision from it. The installed lowering
 * cleared bit 6 of byte 7 alone, leaving any stale committed-hit words from
 * the previous batch in place. */
static void
m5_candidate_request_batch(nir_builder *b, nir_def *id, nir_def *driver,
                           nir_def *lane, nir_def *status_address,
                           nir_def *control, nir_def *committed_t)
{
   nir_store_scs_indexed_agx(b, nir_iand_imm(b, control, ~0x1e00001u),
      driver, lane, .base = 1728, .component = 2);
   if (committed_t) {
      nir_store_scs_uniform_agx(b, m5_native_admission_tmax(b, committed_t),
         nir_load_preamble(b, 1, 64, .base = 152), nir_ishl_imm(b, id, 6),
         .base = 24, .format = PIPE_FORMAT_R32_UINT, .component = 0);
   }
   nir_store_scs_base_agx(b, nir_imm_ivec2(b, 0, 0), status_address,
      .base = 0, .format = PIPE_FORMAT_R32_UINT, .component = 4);
   nir_continue_ray_agx(b, id);
}

void
agx_nir_m5_trace_opaque(nir_builder *b, nir_def *request, nir_def *bvh_state,
                        nir_def *payload_base, nir_def *payload_index,
                        nir_def *status_base, nir_def *launch_words,
                        nir_def **initial_pending,
                        nir_def **continuations)
{
   assert(request->bit_size == 64 && bvh_state->bit_size == 64);
   assert(payload_base->bit_size == 64 && payload_index->bit_size == 32);
   assert(status_base->bit_size == 32);
   nir_variable *steps = NULL;
   if (continuations) {
      steps = nir_local_variable_create(b->impl, glsl_uint_type(), "rt_steps");
      nir_store_var(b, steps, nir_imm_int(b, 0), 1);
   }

   assert(launch_words->bit_size == 32 && launch_words->num_components == 3);
   nir_intersect_ray_agx(b, request, bvh_state,
                         nir_channel(b, launch_words, 0),
                         nir_channel(b, launch_words, 1),
                         nir_channel(b, launch_words, 2));
   nir_def *pending = nir_load_scs_uniform_agx(
      b, 1, 16, payload_base, payload_index, .base = 51,
      .format = PIPE_FORMAT_R8_UINT, .component = 0);
   if (initial_pending)
      *initial_pending = nir_u2u32(b, pending);

   /* Native rt-wide main +0x4f2..+0x596: byte51 selects the continuation
    * path. Its status record is addressed separately, through uniform134.
    * Use the special status-load form only for the read/modify/write pair;
    * the completion check is an ordinary byte load, as in the capture.
    */
   nir_push_if(b, nir_ine_imm(b, pending, 0));
   nir_def *status_address = nir_iadd(b, status_base, payload_index);
   nir_push_loop(b);
   nir_def *status = nir_load_scs_base_agx(
      b, 1, 16, status_address, .base = 7, .format = PIPE_FORMAT_R8_UINT,
      .component = 4, .flags = 5);
   nir_def *resume = nir_ior_imm(b, nir_iand_imm(b, status, ~64u), 16);
   nir_store_scs_base_agx(b, resume, status_address, .base = 7,
                         .format = PIPE_FORMAT_R8_UINT, .component = 4);
   nir_continue_ray_agx(b, nir_ushr_imm(b, payload_index, 6));
   if (steps)
      nir_store_var(b, steps, nir_iadd_imm(b, nir_load_var(b, steps), 1), 1);
   status = nir_load_scs_base_agx(b, 1, 16, status_address, .base = 7,
                                 .format = PIPE_FORMAT_R8_UINT, .component = 4);
   nir_break_if(b, nir_ieq_imm(b, status, 0));
   nir_pop_loop(b, NULL);
   nir_pop_if(b, NULL);
   if (continuations)
      *continuations = nir_load_var(b, steps);
}

/* ScsLayoutGen2 stores the level records before end[6], not at end[6].
 * This entered-query profile has one256-byte record per resident clique. */
static nir_def *
m5_query_level_base(nir_builder *b)
{
   nir_def *packed = nir_load_preamble(b, 1, 32, .base = 52);
   nir_def *cliques = nir_ushr_imm(b, packed, 24);
   return nir_isub(b, nir_load_preamble(b, 1, 32, .base = 48),
                      nir_ishl_imm(b, cliques, 8));
}

/* Native leaf records: the triangle record starts with 0x40|type (type 8
 * opaque, 16 nonopaque) and keeps its geometry/primitive ids at 52/56; the
 * 48-byte bounding-box record starts with marker 0x18 and keeps geometry id
 * (low 30 bits, opaque in bit 31) at 28 and primitive id at 32. The runtime's
 * agx.bounding_box_test reads the box record from the token address plus 4,
 * the same past-the-marker convention the triangle candidate address uses.
 * ASAHI_M5_RQ_AABB_RECORD_BIAS is a diagnostic to A/B that convention on
 * hardware; the production default is 4. */
static unsigned
m5_aabb_record_bias(void)
{
   return debug_get_num_option("ASAHI_M5_RQ_AABB_RECORD_BIAS", 4);
}

/* Caller has established that this queue entry exists. The triangle leaf
 * and instance-node offsets are independently checked by candidate43's
 * nonzero-ID hardware fixture. */
static void
m5_candidate_packet(nir_builder *b, nir_def *driver, nir_def *id,
                    nir_def *index, nir_def *packet[5],
                    nir_def *ray_flags, nir_def **opaque, nir_def **aabb)
{
   nir_def *lane = nir_iand_imm(b, id, 31);
   struct agx_m5_ray_candidate c =
      agx_nir_m5_load_candidate(b, driver, lane, index);
   /* The captured Gen1 extractor identifies triangle type8 as opaque and
    * type16 as nonopaque, and bounding boxes by marker 24 (0x18). The
    * triangle candidate points past this marker. */
   const int bias = m5_aabb_record_bias();
   nir_def *marker = nir_load_global(b, 1, 32,
      nir_iadd_imm(b, c.compressed_address, -4), .align_mul = 4);
   nir_def *is_aabb = NULL;
   if (bias == 4) {
      is_aabb = nir_ieq_imm(b, nir_iand_imm(b, marker, 0xff), 0x18);
   } else {
      nir_def *base_marker = nir_load_global(b, 1, 32,
         nir_iadd_imm(b, c.compressed_address, bias - 4), .align_mul = 4);
      is_aabb = nir_ieq_imm(b, nir_iand_imm(b, base_marker, 0xff), 0x18);
   }
   nir_def *id_offset = nir_bcsel(b, is_aabb,
      nir_imm_int(b, 28 - bias), nir_imm_int(b, 48));
   nir_def *ids = nir_load_global(b, 2, 32,
      nir_iadd(b, c.compressed_address, nir_i2i64(b, id_offset)),
      .align_mul = 4);
   nir_def *level_address = nir_iadd(b, m5_query_level_base(b),
                                    nir_ishl_imm(b, id, 3));
   nir_def *halves[3];
   for (unsigned i = 0; i < 3; ++i)
      halves[i] = nir_u2u32(b, nir_load_scs_base_agx(b, 1, 16,
         level_address, .base = 2 * i, .format = PIPE_FORMAT_R16_UINT,
         .component = 4));
   nir_def *node = nir_pack_64_2x32_split(b,
      nir_ior(b, halves[0], nir_ishl_imm(b, halves[1], 16)), halves[2]);
   nir_def *meta = nir_load_global(b, 2, 32, nir_iadd_imm(b, node, 96),
                                   .align_mul = 4);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *opaque_value = zero;
   if (opaque) {
      /* Native instance records retain Vulkan instance flags at byte8. */
      nir_def *instance_flags = nir_load_global(b, 1, 32,
         nir_iadd_imm(b, node, 8), .align_mul = 4);
      nir_def *geometry_opaque = nir_bcsel(b, is_aabb,
         nir_test_mask(b, nir_channel(b, ids, 0), 0x80000000u),
         nir_ieq_imm(b, nir_iand_imm(b, marker, 31), 8));
      nir_def *instance_opaque = nir_bcsel(b,
         nir_test_mask(b, instance_flags, 8), nir_imm_false(b), geometry_opaque);
      instance_opaque = nir_ior(b, nir_test_mask(b, instance_flags, 4), instance_opaque);
      *opaque = nir_bcsel(b, nir_test_mask(b, ray_flags, 2),
         nir_imm_false(b), nir_ior(b, nir_test_mask(b, ray_flags, 1), instance_opaque));
      opaque_value = nir_b2i32(b, *opaque);
   }
   if (aabb)
      *aabb = is_aabb;
   nir_def *geometry_mask = nir_bcsel(b, is_aabb,
      nir_imm_int(b, 0x3fffffff), nir_imm_int(b, 0xffffff));
   nir_def *face_geom = nir_ior(b,
      nir_iand(b, nir_channel(b, ids, 0), geometry_mask),
      /* Native candidate_vectors +0xf58 shifts the16-bit flags by8. Bit0
       * stays set on both windings and is not the candidate facing bit. */
      nir_bcsel(b, is_aabb, zero,
         nir_ishl_imm(b, nir_iand_imm(b, c.flags, 1u << 8), 23)));
   /* Word0.x keeps the candidate AABB opacity for
    * rayQueryGetIntersectionCandidateAABBOpaqueEXT. */
   packet[0] = nir_vec4(b, opaque_value, zero, zero, zero);
   packet[1] = nir_vec4(b, zero, zero, c.distance, zero);
   /* Type bits 30..31: 1 triangle, 2 bounding box (the native hit record
    * uses the same two-bit kind, see agx.commit_serial.bounding_box). */
   packet[2] = nir_vec4(b, face_geom, zero, zero,
      nir_ior(b, nir_iand(b, nir_channel(b, ids, 1), geometry_mask),
              nir_bcsel(b, is_aabb, nir_imm_int(b, 0x80000000u),
                        nir_imm_int(b, 0x40000000))));
   packet[3] = nir_vec4(b, zero, nir_channel(b, c.barycentrics, 0),
                       nir_channel(b, c.barycentrics, 1), zero);
   packet[4] = nir_vec4(b, nir_unpack_64_2x32_split_x(b, node),
      nir_unpack_64_2x32_split_y(b, node), nir_channel(b, meta, 1),
      nir_channel(b, meta, 0));
}

/* Diagnostic capture of the shared context/root area and driver payloads.
 * Two 64-lane snapshots, before launch and after completion, 64 KiB each.
 * The explicit address is supplied by the private host fixture. */
static void
m5_capture_query_state(nir_builder *b, nir_def *id, nir_def *driver, bool after)
{
   const char *text = getenv("ASAHI_M5_RQ_STATE_ADDRESS");
   if (!text)
      return;
   uint64_t output = strtoull(text, NULL, 0);
   if (!output)
      return;
   nir_def *capture_mask = nir_load_global(b, 1, 32,
      nir_imm_int64(b, output + 131072), .align_mul = 4);
   nir_push_if(b, nir_test_mask(b, capture_mask, after ? 2 : 1));
   nir_variable *cursor = nir_local_variable_create(b->impl, glsl_uint_type(),
                                                    "capture_cursor");
   nir_store_var(b, cursor, nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   nir_def *i = nir_load_var(b, cursor);
   nir_break_if(b, nir_uge_imm(b, i, 32));
   nir_def *offset = nir_iadd(b, nir_ishl_imm(b, id, 9),
                                nir_ishl_imm(b, i, 4));
   for (unsigned region = 0; region < 2; ++region) {
      nir_def *first_base = debug_get_bool_option(
         "ASAHI_M5_RQ_CAPTURE_LEVELS", false)
         ? m5_query_level_base(b) : nir_imm_int(b, 0x6000);
      nir_def *address = nir_iadd(b, offset, region ? driver : first_base);
      nir_def *value = nir_load_scs_base_agx(b, 4, 32, address,
         .base = 0, .format = PIPE_FORMAT_R32_UINT, .component = 4);
      nir_store_global(b, value, nir_iadd_imm(b, nir_u2u64(b, offset),
         output + (after ? 65536 : 0) + region * 32768),
         .align_mul = 16, .write_mask = 15);
   }
   nir_store_var(b, cursor, nir_iadd_imm(b, i, 1), 1);
   nir_pop_loop(b, NULL);
   nir_pop_if(b, NULL);
}

/* Native rays_instance uses a driver record in bank 3, separate from the
 * bank-0 ray. Offsets and strides are from candidate26's live SCS snapshots. */
static void
m5_trace_instance(nir_builder *b, nir_def *selector, nir_def *id,
                  nir_def *opaque, nir_def *query_owner, nir_def *minimum,
                  nir_def *native_opaque_eligible, nir_def **type_prim,
                  nir_def **bary, nir_def **instance, nir_def **front_face,
                  nir_def **distance)
{
   const bool candidate_probe = m5_candidate_mode();
   const bool native_init_parity = candidate_probe &&
      debug_get_bool_option("ASAHI_M5_RQ_NATIVE_INIT_PARITY", false);
   nir_def *lane = nir_iand_imm(b, id, 31);
   nir_def *driver = nir_iadd_imm(b,
      nir_imul_imm(b, nir_iand_imm(b, id, ~31u), 72), 0x5d8000);
   if (candidate_probe)
      driver = nir_iadd(b, nir_load_preamble(b, 1, 32, .base = 56),
         nir_imul_imm(b, nir_iand_imm(b, id, ~31u), m5_candidate_driver_stride()));
   nir_def *zero = nir_imm_int(b, 0);
   if (debug_get_num_option("ASAHI_M5_RQ_ADDRESS_PROBE", 0) == 2) {
      *type_prim = nir_imm_int(b, 0x40000001);
      *bary = nir_vec2(b, nir_u2f32(b, lane), nir_u2f32(b, id));
      *instance = nir_vec4(b, zero, zero, lane, driver);
      *front_face = nir_imul_imm(b, lane, 200);
      return;
   }
   /* RIA3 agx.intersect.ray's payload_segment.41 control uses bit1
    * for forced-opacity=opaque; its feature bits remain 0x120. */
   nir_def *pair = nir_vec2(b, nir_bcsel(b, opaque,
      nir_imm_int(b, 0x122), nir_imm_int(b, 0x120)), zero);
   if (candidate_probe) {
      pair = nir_vec2(b, nir_ior(b, nir_imm_int(b, 0x142),
         nir_b2i32(b, nir_ine_imm(b, query_owner, 0))),
         native_init_parity ? minimum : zero);
   }
   nir_store_scs_indexed_agx(b, pair, driver, lane,
                            .base = 0, .component = 3);
   nir_store_scs_indexed_agx(b, nir_vec2(b, zero, zero), driver, lane,
                            .base = 256, .component = 3);

   /* One per-clique query descriptor, with a fresh active query at +256. */
   nir_def *clique = nir_imul_imm(b, nir_ushr_imm(b, id, 5), 12);
   nir_def *state = nir_load_preamble(b, 1, 64, .base = 140);
   nir_store_scs_uniform_agx(b,
      nir_imm_ivec4(b, 0, 1, 0x600100e8, 0), state, clique,
      .base = 256, .format = PIPE_FORMAT_R32_UINT, .component = 2);
   nir_store_scs_uniform_agx(b, nir_imm_ivec4(b, 0, 0, 0, 0), state, clique,
      .base = 272, .format = PIPE_FORMAT_R32_UINT, .component = 2);
   nir_store_scs_uniform_agx(b,
      nir_load_preamble(b, 2, 32, .base = 80), state, clique,
      .base = 288, .format = PIPE_FORMAT_R32_UINT, .component = 2);
   /* Clear the level's pending bit. The fresh profile has one level. */
   nir_def *level = nir_iadd_imm(b, nir_ishl_imm(b, id, 3), 0x60e000);
   if (candidate_probe)
      level = nir_iadd(b, nir_ishl_imm(b, id, 3), m5_query_level_base(b));
   /* Gen2 native next and the deployed instance callback both clear only
    * bit0 of the compressed instance frame's trailing16-bit field. Its
    * remaining bits belong to the runtime and survive query activation. */
   nir_def *level_flags = nir_imm_intN_t(b, 0, 16);
   if (native_init_parity) {
      level_flags = nir_load_scs_base_agx(b, 1, 16, level,
         .base = 6, .format = PIPE_FORMAT_R16_UINT, .component = 0);
      level_flags = nir_iand_imm(b, level_flags, ~1u);
   }
   nir_store_scs_base_agx(b, level_flags, level,
                         .base = 6, .format = PIPE_FORMAT_R16_UINT, .component = 0);

   /* Native instance publication inserts the high nibble into the live
    * 0x111001 control word. Replacing this with the primitive-only
    * 0x10000000 header makes even the otherwise native shader miss.
    * This constant belongs to the current one-level instance profile. */
   m5_capture_query_state(b, id, driver, false);
   if (candidate_probe) {
      /* Runtime next identifies its resident owner from bits 1..20 of this
       * control word. Native query-model71 masks the entry result to 20 bits
       * and shifts it by1 before publishing, preserving the count/active bits. */
      nir_store_scs_indexed_agx(b,
         nir_ishl_imm(b, nir_iand_imm(b, query_owner, 0xfffff), 1), driver, lane,
         .base = 1728, .component = 2);
      nir_intersect_ray_agx(b, nir_imm_int64(b, 0), selector,
                            nir_imm_int(b, debug_get_bool_option(
                               "ASAHI_M5_RQ_INLINE_PACKET", false)
                                  ? 0x10216001 : 0x00216000), zero, id);

   } else {
      nir_intersect_ray_agx(b, nir_imm_int64(b, 0), selector,
                            nir_imm_int(b, 0x10111001), zero, id);
   }
   if (m5_defer_completion(b)) {
      *type_prim = zero;
      *bary = nir_vec2(b, zero, zero);
      *instance = nir_vec4(b, zero, zero, zero, zero);
      *front_face = zero;
      *distance = zero;
      return;
   }
   nir_def *pending = nir_load_scs_register_agx(b, 1, 16, driver, lane,
      .base = 263, .format = PIPE_FORMAT_R8_UINT, .component = 3, .flags = 9);
   /* Bit 7 also occurs on completed hits. Only bit 6 requests a
    * continuation; testing the entire byte resumes completed traversal. */
   nir_def *resume_needed = nir_ine_imm(b, nir_iand_imm(b, pending, 0x40), 0);
   if (m5_candidate_api())
      resume_needed = nir_iand(b, resume_needed,
         nir_ieq_imm(b, agx_nir_m5_candidate_count(b, driver, lane), 0));
   /* candidate_next imports the native committed packet before clearing
    * completion words on an empty-batch continuation. Eagerly continuing
    * here would discard an opaque hit that has no application candidate. */
   if (m5_native_opaque())
      resume_needed = nir_iand(b, resume_needed,
                                  nir_inot(b, native_opaque_eligible));
   nir_push_if(b, resume_needed);
   nir_def *status_address = nir_iadd_imm(b,
      nir_iadd(b, driver, nir_ishl_imm(b, lane, 3)), 256);
   nir_push_loop(b);
   nir_def *status = nir_load_scs_base_agx(b, 1, 16, status_address,
      .base = 7, .format = PIPE_FORMAT_R8_UINT, .component = 4, .flags = 5);
   if (candidate_probe) {
      /* The queue is empty here (count == 0), so clearing the count bits is
       * a no-op; the active bit and completion words follow the runtime. */
      m5_candidate_request_batch(b, id, driver, lane, status_address,
         nir_load_scs_indexed_agx(b, 1, 32, driver, lane, .base = 1728,
                                  .component = 2), NULL);
   } else {
      nir_def *resume_status = nir_ior_imm(b, nir_iand_imm(b, status, ~64u), 16);
      nir_store_scs_base_agx(b, resume_status, status_address,
         .base = 7, .format = PIPE_FORMAT_R8_UINT, .component = 4);
      nir_continue_ray_agx(b, id);
   }
   status = nir_load_scs_base_agx(b, 1, 16, status_address,
      .base = 7, .format = PIPE_FORMAT_R8_UINT, .component = 4, .flags = 8);
   nir_def *finished = nir_ieq_imm(b, nir_iand_imm(b, status, 0x40), 0);
   if (m5_candidate_api())
      finished = nir_ior(b, finished,
         nir_ine_imm(b, agx_nir_m5_candidate_count(b, driver, lane), 0));
   nir_break_if(b, finished);
   nir_pop_loop(b, NULL);
   nir_pop_if(b, NULL);
   m5_capture_query_state(b, id, driver, true);
   if (candidate_probe) {
      nir_def *count = agx_nir_m5_candidate_count(b, driver, lane);
      struct agx_m5_ray_candidate first =
         agx_nir_m5_load_candidate(b, driver, lane, zero);
      /* Diagnostic transport only: expose queue fields through existing
       * committed-result slots. This is not Vulkan candidate support. */
      *type_prim = nir_ior_imm(b, count, 0x40000000);
      *bary = first.barycentrics;
      *instance = nir_vec4(b, zero, zero, nir_u2u32(b, pending),
                           first.distance);
      *front_face = nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
         .base = 0, .component = 3);
      if (!m5_candidate_api() &&
          debug_get_bool_option("ASAHI_M5_RQ_METADATA_PROBE", false)) {
         nir_def *packet[5];
         nir_def *empty = nir_imm_ivec4(b, 0, 0, 0, 0);
         nir_push_if(b, nir_ine_imm(b, count, 0));
         m5_candidate_packet(b, driver, id, zero, packet, NULL, NULL, NULL);
         nir_push_else(b, NULL);
         nir_pop_if(b, NULL);
         for (unsigned i = 0; i < 5; ++i)
            packet[i] = nir_if_phi(b, packet[i], empty);
         *type_prim = nir_channel(b, packet[2], 3);
         *front_face = nir_channel(b, packet[2], 0);
         *instance = packet[4];
         *bary = nir_channels(b, packet[3], 6);
         *distance = nir_channel(b, packet[1], 2);
      }
      if (debug_get_bool_option("ASAHI_M5_RQ_ADDRESS_PROBE", false)) {
         *bary = nir_vec2(b, nir_u2f32(b, lane), nir_u2f32(b, id));
         *instance = nir_vec4(b, zero, zero, lane, driver);
         *front_face = nir_imul_imm(b, lane, 200);
      }
      return;
   }
   nir_def *hit_words = nir_load_scs_indexed_agx(b, 2, 32, driver, lane,
      .base = 256, .component = 3, .flags = 1);
   *type_prim = nir_channel(b, hit_words, 0);
   /* Fresh G17G triangle_front_facing witness selects1 when the signed
    * second hit word is negative: the native front-face bit is bit31. */
   *front_face = nir_channel(b, hit_words, 1);
   *bary = nir_load_scs_indexed_agx(b, 2, 32, driver, lane,
      .base = 512, .component = 3, .flags = 1);

   /* The captured runtime native-main.bin reads the committed instance
    * pointer at driver + 2112 + 6*lane (selector0). Fresh Metal compiler
    * witnesses use1984 with their own layout; they cannot override this
    * captured depth232/72-byte-driver profile. Do not confuse these with the
    * current traversal instance or the low24 primitive-ID bits. */
   nir_push_if(b, nir_ine_imm(b, nir_ushr_imm(b, *type_prim, 30), 0));
   nir_def *pointer_halves[3];
   for (unsigned i = 0; i < 3; ++i)
      pointer_halves[i] = nir_u2u32(b, nir_load_scs_register_agx(
         b, 1, 16, nir_imul_imm(b, lane, 6), driver,
         .base = 2112 + 2*i, .format = PIPE_FORMAT_R16_UINT, .component = 0));
   nir_def *pointer = nir_pack_64_2x32_split(b,
      nir_ior(b, pointer_halves[0], nir_ishl_imm(b, pointer_halves[1], 16)),
      pointer_halves[2]);
   nir_def *metadata = nir_load_global(b, 2, 32, nir_iadd_imm(b, pointer, 96),
                                       .align_mul = 4);
   nir_def *hit_instance = nir_vec4(b, nir_unpack_64_2x32_split_x(b, pointer),
      nir_unpack_64_2x32_split_y(b, pointer), nir_channel(b, metadata, 1),
      nir_channel(b, metadata, 0));
   nir_push_else(b, NULL);
   nir_def *no_instance = nir_imm_ivec4(b, 0, 0, 0, 0);
   nir_pop_if(b, NULL);
   *instance = nir_if_phi(b, hit_instance, no_instance);
}

static void
m5_init_triangle_packet(nir_builder *b, nir_def *payload_base,
                                nir_def *payload_index, nir_def *origin,
                                nir_def *direction, nir_def *t_min,
                                nir_def *t_max, nir_def *bounds_min,
                                nir_def *bounds_max, nir_def *mask,
                                nir_def *query_flags, nir_def *opaque, nir_def *first_hit,
                                nir_def *triangle_cull,
                                nir_def *native_opaque_eligible)
{
   nir_def *o[3], *d[3], *far[3];
   for (unsigned i = 0; i < 3; ++i) {
      o[i] = nir_channel(b, origin, i);
      d[i] = nir_channel(b, direction, i);
      nir_def *lo = nir_fsub(b, o[i], nir_channel(b, bounds_min, i));
      nir_def *hi = nir_fsub(b, o[i], nir_channel(b, bounds_max, i));
      /* One-ULP outward bias, as in the captured native woop setup. */
      lo = nir_fmul(b, lo, nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), lo),
                                   nir_imm_float(b, 0x1.fffffcp-1f),
                                   nir_imm_float(b, 0x1.000002p0f)));
      hi = nir_fmul(b, hi, nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), hi),
                                   nir_imm_float(b, 0x1.000002p0f),
                                   nir_imm_float(b, 0x1.fffffcp-1f)));
      far[i] = nir_fmax(b, nir_fabs(b, lo), nir_fabs(b, hi));
   }
   nir_def *y_major = nir_flt(b, nir_fabs(b, d[0]), nir_fabs(b, d[1]));
   nir_def *xy_major = nir_bcsel(b, y_major, d[1], d[0]);
   nir_def *z_major = nir_flt(b, nir_fabs(b, xy_major), nir_fabs(b, d[2]));
   nir_def *major = nir_bcsel(b, z_major, nir_imm_int(b, 2),
                             nir_b2i32(b, y_major));
   nir_def *major_dir = nir_bcsel(b, z_major, d[2], xy_major);
   nir_def *major_far = nir_bcsel(b, z_major, far[2],
                                 nir_bcsel(b, y_major, far[1], far[0]));
   nir_def *other_far = nir_bcsel(
      b, z_major, nir_fmax(b, far[0], far[1]),
      nir_fmax(b, far[2], nir_bcsel(b, y_major, far[0], far[1])));
   nir_def *epsilon = nir_fmul_imm(b, nir_fadd(b, major_far, other_far), 0x1.4p-22f);
   nir_def *quantized = nir_iand_imm(
      b, nir_ushr_imm(b, nir_iadd_imm(b, epsilon, 131071), 17), 0x3fff);
   nir_def *woop = nir_ior(b, nir_ishl_imm(b, major, 16),
                         nir_ishl_imm(b, quantized, 18));
   nir_def *u = nir_bcsel(b, nir_ieq_imm(b, major, 2), nir_imm_int(b, 0),
                         nir_iadd_imm(b, major, 1));
   nir_def *v = nir_bcsel(b, nir_ieq_imm(b, major, 0), nir_imm_int(b, 2),
                         nir_iadd_imm(b, major, -1));
   nir_def *negative = nir_flt(b, major_dir, nir_imm_float(b, 0));
   nir_def *sx = nir_bcsel(b, negative, v, u);
   nir_def *sy = nir_bcsel(b, negative, u, v);
   /* Matching RIA3 runtime (03cb21ad.ll): payload_segment's ray
    * control stores opacity in bits0..1 and enables per-geometry opacity
    * with bit14 when the forced opacity is zero. */
   nir_def *control = nir_ior(
      b, nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, mask, 0xff), 18),
                 nir_ior(b, nir_ishl_imm(b, sx, 2), nir_ishl_imm(b, sy, 4))),
      nir_bcsel(b, opaque, nir_imm_int(b, 1), nir_imm_int(b, 1 << 14)));
   if (m5_candidate_mode()) {
      nir_def *nonopaque = nir_ior_imm(b,
         nir_iand_imm(b, control, ~0x4003u), 0x402);
      control = nir_bcsel(b, native_opaque_eligible,
                            nir_ior_imm(b, control, 0x400), nonopaque);
   }
   /* The compiler witness toggles metadata bit11 for accept-any;
    * matching RIA3 IR stores its policy in root packet bit11 (%211).
    * Bit6 changes facing and must not be used for this policy. */
   control = nir_ior(b, control, nir_bcsel(b, first_hit,
      nir_imm_int(b, 1 << 11), nir_imm_int(b, 0)));
   /* RIA3 intersect.ray %224..%238 publishes triangle culling at bits8..9.
    * With the captured winding, Metal back/front map to 1/2 respectively. */
   control = nir_ior(b, control, triangle_cull);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *words[4] = {
      nir_vec4(b, o[0], d[0], o[1], d[1]),
      nir_vec4(b, o[2], d[2], m5_native_admission_tmax(b, t_max), control),
      nir_vec4(b, woop, query_flags, t_min, zero),
      nir_vec4(b, zero, zero, zero, zero),
   };
   for (unsigned i = 0; i < 4; ++i)
      nir_store_scs_uniform_agx(b, words[i], payload_base, payload_index,
                                .base = 16 * i, .format = PIPE_FORMAT_R32_UINT,
                                .component = 0);
}

/* Preserve the compiler fixture/opaque-only packet API. Vulkan candidate
 * lowering supplies its actual ray flags to the private policy-aware helper. */
void
agx_nir_m5_init_triangle_packet(nir_builder *b, nir_def *payload_base,
                              nir_def *payload_index, nir_def *origin,
                              nir_def *direction, nir_def *t_min,
                              nir_def *t_max, nir_def *bounds_min,
                              nir_def *bounds_max, nir_def *mask,
                              nir_def *query_flags, nir_def *opaque,
                              nir_def *first_hit, nir_def *triangle_cull)
{
   m5_init_triangle_packet(b, payload_base, payload_index, origin, direction,
      t_min, t_max, bounds_min, bounds_max, mask, query_flags, opaque,
      first_hit, triangle_cull, nir_imm_false(b));
}

#include "util/u_dynarray.h"

struct m5_rq_state {
   nir_variable *query;
   nir_variable *as, *origin, *direction, *tmin, *tmax, *mask, *active, *flags;
   nir_variable *result[5];
   nir_variable *initial_completion;
   bool queue_first;
   bool native_opaque_possible;
   nir_variable *candidate[5];
   nir_variable *started, *resident_id, *cursor, *owner, *capture_event;
   nir_variable *processed, *replay, *commit_history, *prune_t;
   nir_variable *batch_completed;
   /* Logical query numbers of this variable's elements: first_logical for a
    * scalar query, first_logical..first_logical+logical_count-1 for a query
    * array. Native allocation slots are assigned from the logical number by
    * m5_rq_slot(). */
   unsigned first_logical;
   unsigned logical_count;
   /* Some use of the query array has a non-constant element index. */
   bool dynamic;
};

/* Logical queries beyond the witnessed native allocation indices share a
 * slot. Every use publishes its own owner token and a slot switch between
 * logical queries restarts traversal through the existing owner-mismatch
 * replay, so a shared slot only costs that replay. */
static unsigned
m5_rq_slot(unsigned logical)
{
   return 4 + logical % AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES;
}

/* Logical number of the query element a deref selects, or NULL when the
 * deref is a scalar query or a constant element (then *constant is set). */
static nir_def *
m5_rq_logical(nir_builder *b, const struct m5_rq_state *state,
              nir_deref_instr *query, unsigned *constant)
{
   *constant = state->first_logical;
   if (query->deref_type == nir_deref_type_var)
      return NULL;
   assert(query->deref_type == nir_deref_type_array &&
          nir_deref_instr_parent(query)->deref_type == nir_deref_type_var);
   if (nir_src_is_const(query->arr.index)) {
      *constant += nir_src_as_uint(query->arr.index);
      return NULL;
   }
   return nir_iadd_imm(b, nir_u2u32(b, query->arr.index.ssa),
                       state->first_logical);
}

static nir_def *
m5_rq_enter_reachable(nir_builder *b, nir_def *slot, unsigned first,
                      unsigned i, unsigned slots)
{
   const unsigned candidate =
      (first + i) % AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES;
   if (i + 1 == slots)
      return nir_enter_query_agx(b, .base = 4 + candidate);
   nir_push_if(b, nir_ieq_imm(b, slot, candidate));
   nir_def *then_value = nir_enter_query_agx(b, .base = 4 + candidate);
   nir_push_else(b, NULL);
   nir_def *else_value = m5_rq_enter_reachable(b, slot, first, i + 1, slots);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, then_value, else_value);
}

/* Enter the native query slot of a logical query. A constant logical number
 * enters its slot directly; a dynamically selected element chooses among the
 * slots its elements can map to. */
static nir_def *
m5_rq_enter(nir_builder *b, const struct m5_rq_state *state,
            nir_deref_instr *query)
{
   unsigned constant;
   nir_def *logical = m5_rq_logical(b, state, query, &constant);
   if (!logical)
      return nir_enter_query_agx(b, .base = m5_rq_slot(constant));

   /* The elements reach min(count, slot count) consecutive slots starting at
    * the first element's slot; the last reachable slot needs no test. */
   const unsigned slots = MIN2(state->logical_count,
                               AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES);
   nir_def *slot = nir_umod_imm(b, logical,
                                AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES);
   return m5_rq_enter_reachable(b, slot, m5_rq_slot(state->first_logical) - 4,
                                0, slots);
}

/* Shared-edge/vertex candidate filtering, see m5_boundary_duplicate. On by
 * default (Vulkan requires watertight single hits); ASAHI_M5_RQ_WATERTIGHT=0
 * disables it for diagnostics. */
static bool
m5_rq_watertight(void)
{
   return debug_get_bool_option("ASAHI_M5_RQ_WATERTIGHT", true);
}

/* The candidate cursor indexes a queue of at most 15 entries (a four-bit
 * count). With m5_rq_watertight its upper bits carry the boundary distance
 * key; otherwise they stay zero and the cursor is used unmasked. */
#define M5_CURSOR_INDEX 0xfu
/* Boundary distance key: distance bits 8..31. Partners compute t from their
 * own planes; on hardware they differ by up to one ulp in probes but by more
 * than 31 ulps on the CTS random-height mesh, all inside one 256-ulp step. */
#define M5_BOUNDARY_KEY_LOW 0xffu

static nir_def *
m5_cursor_index(nir_builder *b, nir_deref_instr *cursor)
{
   nir_def *packed = nir_load_deref(b, cursor);
   return m5_rq_watertight() ? nir_iand_imm(b, packed, M5_CURSOR_INDEX)
                             : packed;
}

static void
m5_cursor_set_index(nir_builder *b, nir_deref_instr *cursor, nir_def *index)
{
   if (m5_rq_watertight())
      index = nir_ior(b, index,
         nir_iand_imm(b, nir_load_deref(b, cursor), ~M5_CURSOR_INDEX));
   nir_store_deref(b, cursor, index, 1);
}

/* Private opt-in bounded owner recorder. 64 lanes, 8 lexical queries,
 * 64 events/query, 32 bytes/event = 1 MiB. The host reserves the device-address
 * range and supplies its address before compiling the query shader. No atomics,
 * extra submissions, or global waits. Records are private to a lane/query. */
static uint64_t
m5_model_capture_address(void)
{
   const char *text = getenv("ASAHI_M5_RQ_MODEL_ADDRESS");
   return text ? strtoull(text, NULL, 0) : 0;
}

static void
m5_model_capture(nir_builder *b, struct m5_rq_state *state,
                 nir_deref_instr *capture_event,
                 nir_deref_instr *owner, nir_deref_instr *resident_id,
                 nir_deref_instr *cursor, nir_deref_instr *active,
                 nir_deref_instr *result, unsigned phase)
{
   const uint64_t output = m5_model_capture_address();
   if (!output || !capture_event)
      return;
   nir_def *id = nir_load_deref(b, resident_id);
   nir_def *event = nir_load_deref(b, capture_event);
   nir_push_if(b, nir_iand(b, nir_ult_imm(b, id, 64),
                             nir_ult_imm(b, event, 64)));
   nir_def *lane = nir_iand_imm(b, id, 31);
   nir_def *driver = nir_iadd(b, nir_load_preamble(b, 1, 32, .base = 56),
      nir_imul_imm(b, nir_iand_imm(b, id, ~31u), m5_candidate_driver_stride()));
   nir_def *control = nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
      .base = 1728, .component = 2);
   nir_def *status = nir_load_scs_indexed_agx(b, 2, 32, driver, lane,
      .base = 256, .component = 3);
   nir_def *offset = nir_iadd(b, nir_imul_imm(b, id, 8 * 64 * 32),
      nir_iadd_imm(b, nir_imul_imm(b, event, 32),
         (m5_rq_slot(state->first_logical) - 4) * 64 * 32));
   nir_def *address = nir_iadd_imm(b, nir_u2u64(b, offset), output);
   nir_store_global(b, nir_vec4(b, nir_imm_int(b, phase),
      nir_load_deref(b, owner), id, control), address,
      .align_mul = 16, .write_mask = 15);
   nir_store_global(b, nir_vec4(b, nir_channel(b, status, 0),
      nir_channel(b, status, 1), m5_cursor_index(b, cursor),
      nir_ior(b, nir_iand_imm(b, nir_channel(b, nir_load_deref(b, result), 2),
         ~1u), nir_b2i32(b, nir_load_deref(b, active)))),
      nir_iadd_imm(b, address, 16), .align_mul = 16, .write_mask = 15);
   nir_store_deref(b, capture_event, nir_iadd_imm(b, event, 1), 1);
   nir_pop_if(b, NULL);
}

static const struct glsl_type *
m5_rq_field_type(const struct glsl_type *query, const struct glsl_type *field)
{
   if (!glsl_type_is_array(query))
      return field;
   return glsl_array_type(m5_rq_field_type(glsl_get_array_element(query), field),
                          glsl_get_length(query), 0);
}

static nir_deref_instr *
m5_rq_field_deref(nir_builder *b, nir_deref_instr *query, nir_variable *field)
{
   if (query->deref_type == nir_deref_type_var)
      return nir_build_deref_var(b, field);
   assert(query->deref_type == nir_deref_type_array);
   return nir_build_deref_follower(b,
      m5_rq_field_deref(b, nir_deref_instr_parent(query), field), query);
}

static bool
m5_rq_supported_load(const nir_intrinsic_instr *intr)
{
   switch (nir_intrinsic_ray_query_value(intr)) {
   case nir_ray_query_value_flags:
   case nir_ray_query_value_tmin:
   case nir_ray_query_value_world_ray_origin:
   case nir_ray_query_value_world_ray_direction:
      return true;
   case nir_ray_query_value_intersection_type:
   case nir_ray_query_value_intersection_t:
   case nir_ray_query_value_intersection_primitive_index:
   case nir_ray_query_value_intersection_geometry_index:
   case nir_ray_query_value_intersection_barycentrics:
   case nir_ray_query_value_intersection_instance_id:
   case nir_ray_query_value_intersection_instance_custom_index:
   case nir_ray_query_value_intersection_front_face:
   case nir_ray_query_value_intersection_instance_sbt_index:
   case nir_ray_query_value_intersection_object_ray_direction:
   case nir_ray_query_value_intersection_object_ray_origin:
   case nir_ray_query_value_intersection_object_to_world:
   case nir_ray_query_value_intersection_world_to_object:
      return nir_intrinsic_committed(intr) || m5_candidate_api();
   case nir_ray_query_value_intersection_candidate_aabb_opaque:
      return !nir_intrinsic_committed(intr) && m5_candidate_api();
   default:
      return false;
   }
}

/* Native Gen2 commit records the first32 candidate positions where it
 * publishes a new pruning distance. Replay restores those updates at the
 * same positions instead of restarting with the final bound. */
static void
m5_replay_commit(nir_builder *b, nir_deref_instr *processed,
                 nir_deref_instr *history, nir_deref_instr *prune_t,
                 nir_def *distance)
{
   if (!processed)
      return;
   nir_def *count = nir_load_deref(b, processed);
   nir_push_if(b, nir_ule_imm(b, count, 32));
   nir_def *bit = nir_ishl(b, nir_imm_int(b, 1), nir_iadd_imm(b, count, -1));
   nir_store_deref(b, history, nir_ior(b, nir_load_deref(b, history), bit), 1);
   nir_store_deref(b, prune_t, distance, 1);
   nir_pop_if(b, NULL);
}

/* Consume one candidate at a time. RIA3 intersection_query.next clears both
 * count bits21..24 and the queue's low active bit before requesting another
 * batch. The committed distance is published before continuation so that the
 * intersector can prune work, while the committed API result stays private. */
/* Gen2 intersection_query.next copies segment7 hit words, segment8 bary,
 * segment11 committed instance and the ray-root max-distance into its API
 * result before requesting another batch. The retained gfxrq-both helper8
 * (VA0x100000a800) stores these exact captured-layout offsets: 256/260,
 * 512, 2112/2116 and ray24. Fresh compiler witnesses with a smaller driver
 * layout use1984 instead of2112 and are not this IFT runtime. */
/* Admit the initial upper endpoint, then retain the current winner at ties.
 * A missing committed hit must not turn t==tmax into an application miss. */
nir_def *
agx_nir_m5_should_import_triangle(nir_builder *b, nir_def *native_type,
                                  nir_def *prior_type, nir_def *distance,
                                  nir_def *best_t, nir_def *minimum)
{
   nir_def *before_best = nir_bcsel(b,
      nir_ieq_imm(b, nir_ushr_imm(b, prior_type, 30), 0),
      nir_fge(b, best_t, distance), nir_flt(b, distance, best_t));
   return nir_iand(b, nir_ieq_imm(b, nir_ushr_imm(b, native_type, 30), 1),
      nir_iand(b, before_best, nir_fge(b, distance, minimum)));
}

static void
m5_import_native_opaque(nir_builder *b, nir_def *id, nir_def *driver,
                        nir_def *lane, nir_deref_instr *result[5],
                        nir_deref_instr *minimum, nir_deref_instr *flags,
                        nir_def *completion_status)
{
   nir_push_if(b, m5_native_opaque_eligible(b, nir_load_deref(b, flags)));
   /* Canonical native decoder byte-reencodes REGISTER_INDEXED selector3:
    * type256 scalar marker1; geom260 scalar marker0. */
   nir_def *hit = nir_vec2(b,
      nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
         .base = 256, .component = 3, .flags = 1),
      nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
         .base = 260, .component = 3));
   nir_def *type_prim = nir_channel(b, hit, 0);
   nir_def *distance = nir_load_scs_uniform_agx(b, 1, 32,
      nir_load_preamble(b, 1, 64, .base = 152), nir_ishl_imm(b, id, 6),
      .base = 24, .format = PIPE_FORMAT_R32_UINT, .component = 0);
   nir_def *best = nir_channel(b, nir_load_deref(b, result[1]), 2);
   /* Private benchmark transport: its 64-byte output record leaves words
    * 1..10 unused. Capture the raw operands before interpreting type or t,
    * keeping the public query result and flags unchanged. The caller pins
    * the supplied allocation for the entire dispatch. */
   const char *diag_text = getenv("ASAHI_M5_RQ_NATIVE_OPAQUE_DIAG_ADDRESS");
   uint64_t diag_address = diag_text ? strtoull(diag_text, NULL, 0) : 0;
   if (diag_address && b->shader->info.stage == MESA_SHADER_COMPUTE) {
      nir_def *invocation = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
      nir_def *output = nir_iadd_imm(b,
         nir_ishl_imm(b, nir_u2u64(b, invocation), 6), diag_address);
      if (debug_get_bool_option("ASAHI_M5_RQ_NATIVE_OPAQUE_DIAG_TAIL", false)) {
         /* Raw root bytes32..63 expose both coordinate slots. Offset60 is
          * left uninterpreted: the native union declares an i32 there. */
         nir_def *base = nir_load_preamble(b, 1, 64, .base = 152);
         nir_def *index = nir_ishl_imm(b, id, 6);
         for (unsigned i = 0; i < 2; ++i) {
            nir_def *tail = nir_load_scs_uniform_agx(b, 4, 32, base, index,
               .base = 32 + 16 * i, .format = PIPE_FORMAT_R32_UINT,
               .component = 0);
            nir_store_global(b, tail, nir_iadd_imm(b, output, 4 + 16 * i),
               .align_mul = 4, .write_mask = 15);
         }
         nir_store_global(b, nir_vec2(b, type_prim, distance),
            nir_iadd_imm(b, output, 36), .align_mul = 4, .write_mask = 3);
         nir_store_global(b, nir_u2u32(b, completion_status),
            nir_iadd_imm(b, output, 48), .align_mul = 4, .write_mask = 1);
         nir_def *control = nir_load_scs_uniform_agx(b, 1, 32, base, index,
            .base = 28, .format = PIPE_FORMAT_R32_UINT, .component = 0);
         nir_store_global(b, control, nir_iadd_imm(b, output, 56),
            .align_mul = 4, .write_mask = 1);
      } else if (debug_get_bool_option("ASAHI_M5_RQ_NATIVE_OPAQUE_DIAG_PACKET", false)) {
         nir_def *base = nir_load_preamble(b, 1, 64, .base = 152);
         nir_def *index = nir_ishl_imm(b, id, 6);
         nir_def *ray[3];
         for (unsigned i = 0; i < 3; ++i)
            ray[i] = nir_load_scs_uniform_agx(b, 4, 32, base, index,
               .base = 16 * i, .format = PIPE_FORMAT_R32_UINT, .component = 0);
         nir_store_global(b, ray[0], nir_iadd_imm(b, output, 4),
            .align_mul = 4, .write_mask = 15);
         nir_store_global(b, ray[1], nir_iadd_imm(b, output, 20),
            .align_mul = 4, .write_mask = 15);
         nir_store_global(b, nir_channels(b, ray[2], 3),
            nir_iadd_imm(b, output, 36), .align_mul = 4, .write_mask = 3);
         nir_store_global(b, nir_channel(b, ray[2], 2),
            nir_iadd_imm(b, output, 48), .align_mul = 4, .write_mask = 1);
         nir_store_global(b, nir_channel(b, ray[2], 3),
            nir_iadd_imm(b, output, 56), .align_mul = 4, .write_mask = 1);
      } else {
      nir_store_global(b, nir_vec4(b, type_prim, nir_channel(b, hit, 1),
                                  distance, nir_u2u32(b, completion_status)),
         nir_iadd_imm(b, output, 4), .align_mul = 4, .write_mask = 15);
      nir_def *control = nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
         .base = 1728, .component = 2);
      nir_store_global(b, nir_vec4(b, driver, id, lane, control),
         nir_iadd_imm(b, output, 20), .align_mul = 4, .write_mask = 15);
      nir_store_global(b, nir_vec2(b, best,
         nir_b2i32(b, m5_native_opaque_eligible(b, nir_load_deref(b, flags)))),
         nir_iadd_imm(b, output, 36), .align_mul = 4, .write_mask = 3);
      }
   }
   /* AABB candidates are still returned to the application. Native triangle
    * kind1 is the only packet auto-committed by this qualification path. */
   nir_push_if(b, agx_nir_m5_should_import_triangle(b, type_prim,
      nir_channel(b, nir_load_deref(b, result[2]), 3), distance, best,
      nir_load_deref(b, minimum)));
   nir_def *bary = nir_load_scs_indexed_agx(b, 2, 32, driver, lane,
      .base = 512, .component = 3, .flags = 1);
   nir_def *halves[3];
   for (unsigned i = 0; i < 3; ++i)
      halves[i] = nir_u2u32(b, nir_load_scs_register_agx(b, 1, 16,
         nir_imul_imm(b, lane, 6), driver, .base = (m5_raw50_layout_probe() ? 8640 : 2112) + 2 * i,
         .format = PIPE_FORMAT_R16_UINT, .component = 0));
   nir_def *pointer = nir_pack_64_2x32_split(b,
      nir_ior(b, halves[0], nir_ishl_imm(b, halves[1], 16)), halves[2]);
   nir_def *metadata = nir_load_global(b, 2, 32,
      nir_iadd_imm(b, pointer, 96), .align_mul = 4);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *packet[5] = {
      nir_imm_ivec4(b, 1, 0, 0, 0),
      nir_vec4(b, zero, zero, distance, zero),
      nir_vec4(b, nir_channel(b, hit, 1), zero, zero, type_prim),
      nir_vec4(b, zero, nir_channel(b, bary, 0),
                   nir_channel(b, bary, 1), zero),
      nir_vec4(b, nir_unpack_64_2x32_split_x(b, pointer),
                   nir_unpack_64_2x32_split_y(b, pointer),
                   nir_channel(b, metadata, 1), nir_channel(b, metadata, 0)),
   };
   for (unsigned i = 0; i < 5; ++i)
      nir_store_deref(b, result[i], packet[i], 15);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

/* Native Gen2 copy_committed_data latches query state1 when a completed
 * batch contains candidates, even when control bit0 is clear. It then clears
 * that bit; exhausting the batch resumes from the latched state1. Every
 * activation/continuation resets cursor0 and keeps count until the next
 * request, so count!=0 retains that state without another per-query variable.
 * Preserve existing busy/control admission while qualifying full parity. */
nir_def *
agx_nir_m5_should_resume_batch(nir_builder *b, nir_def *status,
                              nir_def *control, nir_def *count)
{
   return nir_ior(b, nir_ine_imm(b, count, 0),
      nir_ior(b, nir_test_mask(b, status, 64), nir_test_mask(b, control, 1)));
}

/* Vulkan watertightness: a ray through an edge shared by two triangles, or
 * through the shared vertex of a closed fan, must not report a double hit.
 *
 * M5's triangle test is inclusive. When its edge function is exactly zero it
 * returns every triangle at that edge or vertex, each with a zero
 * barycentric there, up to the hardware's barycentric rounding (~2^-24
 * observed; diagonal edges can come back as a tiny nonzero u or v, and
 * u + v as 1 +- an ulp). Its arithmetic is not exact:
 * hardware probes of the closed-fan rays report both spoke neighbours for
 * rays that exact arithmetic places strictly inside one triangle. The shader
 * therefore cannot predict which candidates the hardware returns, and the
 * native runtime's query.next does not filter them either.
 *
 * Observe instead: remember the distance of the last boundary triangle this
 * query let through, and drop a later boundary triangle at the same distance
 * (see M5_BOUNDARY_KEY_LOW). Both are then at the same point of the shared
 * edge or vertex, so the ray still hits there exactly once. A drop only
 * happens after its partner was delivered, so it never turns a hit into a
 * miss.
 *
 * The key lives in the candidate cursor's spare bits: the cursor never
 * exceeds the four-bit queue count. No memory access; it runs only for
 * admitted triangles. It still adds about 35 instructions per proceed site
 * and can cost occupancy in some shaders; it is on by default
 * (ASAHI_M5_RQ_WATERTIGHT=0 disables it, see m5_rq_watertight). */
static nir_def *
m5_boundary_duplicate(nir_builder *b, nir_def *bary, nir_def *distance,
                      nir_def *is_aabb, nir_deref_instr *cursor)
{
   nir_def *u = nir_channel(b, bary, 0), *v = nir_channel(b, bary, 1);
   nir_def *boundary = nir_iand(b, nir_inot(b, is_aabb), nir_fle_imm(b,
      nir_fmin(b, nir_fmin(b, u, v), nir_fsub_imm(b, 1.0f, nir_fadd(b, u, v))),
      0x1p-22f));
   nir_def *packed = nir_load_deref(b, cursor);
   nir_def *key = nir_iand_imm(b, distance, ~M5_BOUNDARY_KEY_LOW);
   /* Equal or adjacent keys, |key - last| <= one key step. */
   nir_def *same_point = nir_ule_imm(b, nir_iadd_imm(b, nir_isub(b, key,
      nir_iand_imm(b, packed, ~M5_BOUNDARY_KEY_LOW)),
      M5_BOUNDARY_KEY_LOW + 1), 2 * (M5_BOUNDARY_KEY_LOW + 1));
   nir_store_deref(b, cursor, nir_bcsel(b,
      nir_iand(b, boundary, nir_inot(b, same_point)),
      nir_ior(b, key, nir_iand_imm(b, packed, M5_CURSOR_INDEX)), packed), 1);
   return nir_iand(b, boundary, same_point);
}

static void
m5_candidate_next(nir_builder *b, nir_deref_instr *active,
                  nir_deref_instr *saved_id, nir_deref_instr *cursor,
                  nir_deref_instr *result[5], nir_deref_instr *minimum,
                  nir_deref_instr *flags, nir_deref_instr *candidate[5],
                  nir_deref_instr *initial_completion, nir_deref_instr *batch_completed,
                  bool queue_first,
                  nir_variable *more, nir_deref_instr *processed, nir_deref_instr *replay,
                  nir_deref_instr *history, nir_deref_instr *prune_t)
{
   nir_def *id = nir_load_deref(b, saved_id);
   nir_def *lane = nir_iand_imm(b, id, 31);
   nir_def *driver = nir_iadd(b, nir_load_preamble(b, 1, 32, .base = 56),
      nir_imul_imm(b, nir_iand_imm(b, id, ~31u), m5_candidate_driver_stride()));
   nir_def *status_address = nir_iadd_imm(b,
      nir_iadd(b, driver, nir_ishl_imm(b, lane, 3)), 256);

   nir_variable *status_value = initial_completion
      ? nir_local_variable_create(b->impl, glsl_uint16_t_type(), "rq_status")
      : NULL;
   nir_push_loop(b);
   nir_def *status = NULL;
   if (!queue_first) {
      if (batch_completed)
         nir_push_if(b, nir_inot(b, nir_load_deref(b, batch_completed)));
      if (initial_completion) {
         /* The native first read is indexed263/selector3/marker1/wait128.
          * BASE7/selector4/marker0 is witnessed only after continue_ray.
          * Keep the marked read's value live in the actual query decision. */
         nir_push_if(b, nir_load_deref(b, initial_completion));
         nir_store_var(b, status_value,
            nir_load_scs_register_agx(b, 1, 16, driver, lane,
               .base = 263, .format = PIPE_FORMAT_R8_UINT,
               .component = 3, .flags = 9), 1);
         nir_store_deref(b, initial_completion, nir_imm_false(b), 1);
         nir_push_else(b, NULL);
         nir_store_var(b, status_value,
            nir_load_scs_base_agx(b, 1, 16, status_address,
               .base = 7, .format = PIPE_FORMAT_R8_UINT,
               .component = 4, .flags = 8), 1);
         nir_pop_if(b, NULL);
         status = nir_load_var(b, status_value);
      } else {
         status = nir_load_scs_base_agx(b, 1, 16, status_address,
            .base = 7, .format = PIPE_FORMAT_R8_UINT, .component = 4, .flags = 8);
      }
      /* flags8 emits the captured traversal completion wait. Committed fields
       * are not readable until it completes, even when the candidate count is
       * zero and no application candidate will be produced. */
      if (initial_completion)
         m5_import_native_opaque(b, id, driver, lane, result, minimum, flags, status);
      if (batch_completed) {
         /* Gen2 next consumes a completed queue before reading completion
          * again. Import its immutable opaque result once, before exposing
          * this batch's first application candidate. No traversal runs while
          * the singleton query consumes the rest of that completed batch. */
         nir_store_deref(b, batch_completed, nir_imm_true(b), 1);
         nir_pop_if(b, NULL);
      }
   }
   nir_def *index = m5_cursor_index(b, cursor);
   nir_def *control = nir_load_scs_indexed_agx(b, 1, 32, driver, lane,
      .base = 1728, .component = 2);
   nir_def *count = nir_iand_imm(b, nir_ushr_imm(b, control, 21), 15);
   /* Private first-queue decode transport for the pinned standalone
    * compute fixture. No additional completion wait or queue mutation. */
   const char *diag_text = getenv("ASAHI_M5_RQ_ZERO_SPAN_DIAG_ADDRESS");
   uint64_t diag_address = diag_text ? strtoull(diag_text, NULL, 0) : 0;
   if (diag_address && b->shader->info.stage == MESA_SHADER_COMPUTE) {
      nir_push_if(b, nir_ieq_imm(b, index, 0));
      nir_def *invocation = nir_channel(b,
         nir_load_global_invocation_id(b, 32), 0);
      nir_def *output = nir_iadd_imm(b,
         nir_ishl_imm(b, nir_u2u64(b, invocation), 6), diag_address);
      nir_def *ray_tmax = nir_load_scs_uniform_agx(b, 1, 32,
         nir_load_preamble(b, 1, 64, .base = 152),
         nir_ishl_imm(b, id, 6), .base = 24,
         .format = PIPE_FORMAT_R32_UINT, .component = 0);
      nir_store_global(b, nir_vec4(b, control, count,
         nir_u2u32(b, status), index), nir_iadd_imm(b, output, 12),
         .align_mul = 4, .write_mask = 15);
      nir_store_global(b, nir_vec4(b, nir_b2i32(b, nir_load_deref(b, active)),
         ray_tmax, id, driver), nir_iadd_imm(b, output, 28),
         .align_mul = 4, .write_mask = 15);
      nir_pop_if(b, NULL);
   }
   nir_push_if(b, nir_ult(b, index, count));
   /* Gen2 native next restarts on an owner mismatch, then pops the
    * previously processed candidate prefix before exposing a new candidate.
    * Private opt-in diagnostic: qualify the uncommitted replay path first. */
   if (replay) {
      nir_push_if(b, nir_ine_imm(b, nir_load_deref(b, replay), 0));
      nir_def *replay_index = nir_isub(b, nir_load_deref(b, processed), nir_load_deref(b, replay));
      nir_def *bit = nir_ishl(b, nir_imm_int(b, 1), replay_index);
      nir_push_if(b, nir_iand(b, nir_ult_imm(b, replay_index, 32),
         nir_ine_imm(b, nir_iand(b, nir_load_deref(b, history), bit), 0)));
      struct agx_m5_ray_candidate old = agx_nir_m5_load_candidate(b, driver, lane, index);
      nir_store_deref(b, prune_t, old.distance, 1);
      nir_pop_if(b, NULL);
      nir_store_deref(b, replay, nir_iadd_imm(b, nir_load_deref(b, replay), -1), 1);
      m5_cursor_set_index(b, cursor, nir_iadd_imm(b, index, 1));
      nir_push_else(b, NULL);
      nir_store_deref(b, processed, nir_iadd_imm(b, nir_load_deref(b, processed), 1), 1);
   }
   nir_def *packet[5];
   nir_def *ray_flags = nir_load_deref(b, flags);
   nir_def *opaque, *is_aabb;
   m5_candidate_packet(b, driver, id, index, packet, ray_flags, &opaque,
                       &is_aabb);
   m5_cursor_set_index(b, cursor, nir_iadd_imm(b, index, 1));
   nir_def *distance = nir_channel(b, packet[1], 2);
   nir_def *best_t = nir_channel(b, nir_load_deref(b, result[1]), 2);
   /* A procedural candidate is always handed to the application: the
    * shader computes its own intersection distance. */
   nir_def *before_best = nir_fge(b, best_t, distance);
   {
      /* A native batch may already contain another triangle at the exact
       * distance just committed. Gen2 query.next clips queued candidates
       * strictly against its committed distance before exposing them.
       * Preserve the inclusive initial API endpoint when no hit exists;
       * ignored/unconfirmed candidates therefore do not suppress ties. */
      nir_def *committed_kind = nir_ushr_imm(b,
         nir_channel(b, nir_load_deref(b, result[2]), 3), 30);
      before_best = nir_bcsel(b, nir_ieq_imm(b, committed_kind, 0),
         before_best, nir_flt(b, distance, best_t));
   }
   nir_def *in_interval = nir_ior(b, is_aabb, nir_iand(b,
      nir_fge(b, distance, nir_load_deref(b, minimum)), before_best));
   /* RayFlagsCullOpaque/CullNoOpaque, then SpvRayFlagsSkipTrianglesKHR
    * (0x100) and SpvRayFlagsSkipAABBsKHR (0x200) on the candidate kind. */
   nir_def *culled = nir_bcsel(b, opaque,
      nir_test_mask(b, ray_flags, 0x40), nir_test_mask(b, ray_flags, 0x80));
   culled = nir_ior(b, culled, nir_bcsel(b, is_aabb,
      nir_test_mask(b, ray_flags, 0x200), nir_test_mask(b, ray_flags, 0x100)));
   nir_push_if(b, nir_iand(b, in_interval, nir_inot(b, culled)));
   const bool watertight = m5_rq_watertight();
   if (watertight)
      nir_push_if(b, nir_inot(b, m5_boundary_duplicate(b,
         nir_channels(b, packet[3], 6), distance, is_aabb, cursor)));
   /* Opaque triangles commit without application involvement; an opaque
    * AABB is still a candidate (rayQueryGetIntersectionCandidateAABBOpaqueEXT). */
   nir_push_if(b, nir_iand(b, opaque, nir_inot(b, is_aabb)));
   for (unsigned i = 0; i < 5; ++i)
      nir_store_deref(b, result[i], packet[i], 15);
   m5_replay_commit(b, processed, history, prune_t, distance);
   nir_push_if(b, nir_test_mask(b, ray_flags, 4));
   nir_store_deref(b, active, nir_imm_false(b), 1);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   for (unsigned i = 0; i < 5; ++i)
      nir_store_deref(b, candidate[i], packet[i], 15);
   nir_store_var(b, more, nir_imm_true(b), 1);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   if (watertight)
      nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   if (replay)
      nir_pop_if(b, NULL);
   /* This lowering runs after continue lowering. Keep queue exhaustion in
    * the else branch so both paths reach the natural loop backedge; emitting
    * nir_jump_continue here would reach a backend that only accepts breaks. */
   nir_push_else(b, NULL);

   if (queue_first || batch_completed)
      /* Queue exhaustion needs the returned status, but a latched completed
       * batch needs no new completion dependency. Continuation invalidates
       * that latch and retains its real completion wait on the next loop. */
      status = nir_load_scs_base_agx(b, 1, 16, status_address,
         .base = 7, .format = PIPE_FORMAT_R8_UINT, .component = 4);
   /* Native Gen2 state1 survives consumption of a nonempty completed queue,
    * including the one-AABB control0x200008/status0 batch. Re-testing only the
    * busy byte and cleared low control bit would lose later intersections. */
   nir_def *resume = agx_nir_m5_should_resume_batch(b, status, control, count);
   nir_push_if(b, nir_inot(b, resume));
   nir_store_deref(b, active, nir_imm_false(b), 1);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);

   m5_candidate_request_batch(b, id, driver, lane, status_address, control,
      prune_t ? nir_load_deref(b, prune_t) : nir_channel(b, nir_load_deref(b, result[1]), 2));
   if (batch_completed)
      nir_store_deref(b, batch_completed, nir_imm_false(b), 1);
   if (queue_first) {
      /* Continuation starts a new asynchronous batch. Keep this completion
       * wait even if its returned byte is unread; compiler DCE must retain
       * the scs_rt_completion hardware side effect and real destination. */
      nir_load_scs_base_agx(b, 1, 16, status_address,
         .base = 7, .format = PIPE_FORMAT_R8_UINT,
         .component = 4, .flags = 8);
   }
   m5_cursor_set_index(b, cursor, nir_imm_int(b, 0));
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
}

bool
agx_nir_m5_split_ray_query_arrays(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   bool progress = false;

   nir_foreach_function_temp_variable_safe(var, impl) {
      if (!var->data.ray_query || !glsl_type_is_array(var->type))
         continue;
      const struct glsl_type *elem = glsl_get_array_element(var->type);
      const unsigned len = glsl_get_length(var->type);
      if (glsl_type_is_array(elem) || len == 0)
         continue;

      /* A dynamically selected element of a small array is resolved into
       * control flow before splitting, so each element becomes a separate
       * query with its own copy of the operation (the qualified form for
       * arrays of up to the native slot count). Larger arrays keep their
       * array form: agx_nir_lower_m5_ray_query gives every element its own
       * state and selects the native slot per use. */
      nir_builder indirect = nir_builder_create(impl);
      if (len <= AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES)
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(op->intrinsic))
               continue;
            nir_deref_instr *d = nir_src_as_deref(op->src[0]);
            if (d->deref_type != nir_deref_type_array ||
                nir_src_is_const(d->arr.index))
               continue;
            nir_deref_instr *parent = nir_deref_instr_parent(d);
            if (parent->deref_type != nir_deref_type_var || parent->var != var)
               continue;
            indirect.cursor = nir_before_instr(instr);
            nir_def **values = nir_intrinsic_infos[op->intrinsic].has_dest ? calloc(len, sizeof(*values)) : NULL;
            for (unsigned i = 0; i < len; ++i) {
               nir_push_if(&indirect, nir_ieq_imm(&indirect, d->arr.index.ssa, i));
               nir_deref_instr *selected = nir_build_deref_array_imm(&indirect, parent, i);
               nir_intrinsic_instr *copy = nir_instr_as_intrinsic(nir_instr_clone(shader, instr));
               copy->src[0] = nir_src_for_ssa(&selected->def);
               nir_builder_instr_insert(&indirect, &copy->instr);
               if (values)
                  values[i] = &copy->def;
               nir_push_else(&indirect, NULL);
            }
            nir_def *value = values ? nir_undef(&indirect, op->def.num_components, op->def.bit_size) : NULL;
            for (unsigned i = len; i-- > 0;) {
               nir_pop_if(&indirect, NULL);
               if (values)
                  value = nir_if_phi(&indirect, values[i], value);
            }
            if (value)
               nir_def_rewrite_uses(&op->def, value);
            nir_instr_remove(instr);
            free(values);
            progress = true;
         }
      }
      nir_remove_dead_derefs_impl(impl);
      /* Every use of the variable must be a constant, in-range element. */
      bool constant_only = true;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *d = nir_instr_as_deref(instr);
            if (d->deref_type != nir_deref_type_var || d->var != var)
               continue;
            nir_foreach_use(src, &d->def) {
               nir_instr *use = nir_src_use_instr(src);
               nir_deref_instr *a = use->type == nir_instr_type_deref
                  ? nir_instr_as_deref(use) : NULL;
               if (!a || a->deref_type != nir_deref_type_array ||
                   !nir_src_is_const(a->arr.index) ||
                   nir_src_as_uint(a->arr.index) >= len)
                  constant_only = false;
            }
         }
      }
      if (!constant_only)
         continue;

      nir_variable **elems = calloc(len, sizeof(*elems));
      for (unsigned i = 0; i < len; ++i) {
         elems[i] = nir_local_variable_create(impl, elem, var->name);
         elems[i]->data.ray_query = true;
      }

      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *a = nir_instr_as_deref(instr);
            if (a->deref_type != nir_deref_type_array)
               continue;
            nir_deref_instr *parent = nir_deref_instr_parent(a);
            if (!parent || parent->deref_type != nir_deref_type_var ||
                parent->var != var)
               continue;
            b.cursor = nir_before_instr(&a->instr);
            nir_deref_instr *split = nir_build_deref_var(
               &b, elems[nir_src_as_uint(a->arr.index)]);
            nir_def_rewrite_uses(&a->def, &split->def);
            nir_instr_remove(&a->instr);
         }
      }
      free(elems);
      progress = true;
   }

   if (progress) {
      nir_remove_dead_derefs_impl(impl);
      nir_remove_dead_variables(shader, nir_var_function_temp, NULL);
      nir_progress(true, impl, nir_metadata_control_flow);
   } else {
      nir_no_progress(impl);
   }
   return progress;
}

/* GLSL passes rayQueryEXT to functions as inout, so after inlining the callee
 * works on a temporary query that is assigned from the caller's query on entry
 * (and copied back on return). Ray queries are opaque objects: such a
 * temporary is an alias of the caller's query, not a second live query.
 * Rewrite every use of a whole-variable copy to the source variable and drop
 * the copies, so a wrapper function does not look like two queries.
 */
static bool
m5_rq_alias_copies(nir_function_impl *impl)
{
   bool progress = false;

   for (bool again = true; again;) {
      again = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
            if (store->intrinsic != nir_intrinsic_store_deref)
               continue;

            nir_deref_instr *dst = nir_src_as_deref(store->src[0]);
            nir_intrinsic_instr *load = nir_src_as_intrinsic(store->src[1]);
            if (dst->deref_type != nir_deref_type_var ||
                !dst->var->data.ray_query || !load ||
                load->intrinsic != nir_intrinsic_load_deref)
               continue;

            nir_deref_instr *src = nir_src_as_deref(load->src[0]);
            if (src->deref_type != nir_deref_type_var ||
                !src->var->data.ray_query)
               continue;

            /* Self copy, e.g. a copy-back after aliasing. */
            if (src->var == dst->var) {
               nir_instr_remove(&store->instr);
               progress = again = true;
               continue;
            }

            /* Only alias a temporary whose sole definition is this copy. */
            unsigned stores = 0;
            nir_foreach_block(b2, impl) {
               nir_foreach_instr(i2, b2) {
                  if (i2->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *s2 = nir_instr_as_intrinsic(i2);
                  if (s2->intrinsic != nir_intrinsic_store_deref &&
                      s2->intrinsic != nir_intrinsic_copy_deref)
                     continue;
                  nir_deref_instr *d2 = nir_src_as_deref(s2->src[0]);
                  if (nir_deref_instr_get_variable(d2) == dst->var)
                     stores++;
               }
            }
            if (stores != 1 || dst->var->data.mode != nir_var_function_temp)
               continue;

            nir_variable *alias = dst->var, *target = src->var;
            nir_foreach_block(b2, impl) {
               nir_foreach_instr_safe(i2, b2) {
                  if (i2->type != nir_instr_type_deref)
                     continue;
                  nir_deref_instr *d = nir_instr_as_deref(i2);
                  if (d->deref_type != nir_deref_type_var || d->var != alias)
                     continue;
                  d->var = target;
                  d->modes = target->data.mode;
               }
            }
            nir_instr_remove(&store->instr);
            progress = again = true;
         }
      }
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);
   return progress;
}

/* A dynamically selected element's state is staged through per-operation
 * scalar shadows: each field is copied in once before the operation and the
 * fields the operation writes are copied back once after it, so the element
 * selection costs one select tree per field rather than one per access. */
struct m5_rq_shadows {
   nir_function_impl *impl;
   nir_deref_instr *query;
   unsigned count;
   struct {
      nir_variable *field;
      nir_variable *shadow;
   } entries[48];
};

static nir_deref_instr *
m5_rq_op_field(nir_builder *b, struct m5_rq_shadows *shadows,
               nir_variable *field)
{
   nir_deref_instr *query = shadows->query;
   nir_deref_instr *elem = m5_rq_field_deref(b, query, field);
   if (query->deref_type != nir_deref_type_array ||
       nir_src_is_const(query->arr.index))
      return elem;
   assert(shadows->count < ARRAY_SIZE(shadows->entries));
   nir_variable *shadow =
      nir_local_variable_create(shadows->impl, elem->type, field->name);
   nir_deref_instr *staged = nir_build_deref_var(b, shadow);
   nir_copy_deref(b, staged, elem);
   shadows->entries[shadows->count].field = field;
   shadows->entries[shadows->count].shadow = shadow;
   shadows->count++;
   return staged;
}

/* Whether a lowered operation may write a state field. */
static bool
m5_rq_op_writes(nir_intrinsic_op op, const struct m5_rq_state *state,
                nir_variable *field)
{
   bool result = false, candidate = false;
   for (unsigned i = 0; i < 5; ++i) {
      result |= field == state->result[i];
      candidate |= field == state->candidate[i];
   }
   switch (op) {
   case nir_intrinsic_rq_initialize:
      return true;
   case nir_intrinsic_rq_proceed:
      return field == state->active || field == state->started ||
             field == state->owner || field == state->resident_id ||
             field == state->cursor || result || candidate ||
             field == state->batch_completed ||
             field == state->initial_completion ||
             field == state->processed || field == state->replay ||
             field == state->commit_history || field == state->prune_t ||
             field == state->capture_event;
   case nir_intrinsic_rq_terminate:
      return field == state->active;
   case nir_intrinsic_rq_confirm_intersection:
   case nir_intrinsic_rq_generate_intersection:
      return field == state->active || result ||
             field == state->commit_history || field == state->prune_t;
   default:
      return false;
   }
}

static void
m5_rq_op_writeback(nir_builder *b, struct m5_rq_shadows *shadows,
                   nir_intrinsic_op op, const struct m5_rq_state *state)
{
   for (unsigned i = 0; i < shadows->count; ++i) {
      if (!m5_rq_op_writes(op, state, shadows->entries[i].field))
         continue;
      nir_copy_deref(b,
         m5_rq_field_deref(b, shadows->query, shadows->entries[i].field),
         nir_build_deref_var(b, shadows->entries[i].shadow));
   }
}

/* Per-query state: one value per logical query, so an array query gets an
 * array of the field type. The state of a dynamically selected element is
 * resolved to selects over the element states after lowering
 * (nir_lower_indirect_derefs_to_if_else_trees over shader temporaries), so
 * query state never reaches thread-private memory. */
static nir_variable *
m5_rq_var(nir_shader *shader, nir_function_impl *impl,
          const struct m5_rq_state *state, const struct glsl_type *type,
          const char *name)
{
   const struct glsl_type *field = m5_rq_field_type(state->query->type, type);
   if (state->dynamic)
      return nir_variable_create(shader, nir_var_shader_temp, field, name);
   return nir_local_variable_create(impl, field, name);
}

bool
agx_nir_lower_m5_ray_query(nir_shader *shader,
                           const struct agx_m5_ray_query_layout *layout)
{
   const bool kernel = layout && layout->compute_kernel &&
      (shader->info.stage == MESA_SHADER_TESS_CTRL ||
       shader->info.stage == MESA_SHADER_TESS_EVAL ||
       shader->info.stage == MESA_SHADER_GEOMETRY ||
       shader->info.stage == MESA_SHADER_VERTEX ||
       shader->info.stage == MESA_SHADER_FRAGMENT);
   if (!layout || (!kernel && shader->info.stage != MESA_SHADER_COMPUTE) ||
       (!kernel && shader->info.workgroup_size_variable) ||
       layout->payload_uniform > 508 ||
       (layout->payload_uniform & 1) || layout->bvh_uniform > 508 ||
       (layout->bvh_uniform & 1) || layout->status_uniform > 510 ||
       (layout->status_uniform & 1) || layout->context_uniform > 508 ||
       (layout->context_uniform & 1) || layout->as_table_uniform > 508 ||
       (layout->as_table_uniform & 1) ||
       (!!layout->context_uniform != !!layout->as_table_uniform))
      return false;

   if (debug_get_bool_option("ASAHI_M5_RQ_DIAG", false))
      nir_print_shader(shader, stderr);
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (m5_rq_alias_copies(impl))
      nir_remove_dead_variables(shader, nir_var_function_temp, NULL);
   struct util_dynarray queries = UTIL_DYNARRAY_INIT;
   struct util_dynarray ops = UTIL_DYNARRAY_INIT;
   /* Logical queries: scalar query variables plus every element of a query
    * array. Native slots are assigned per logical query by m5_rq_slot(). */
   unsigned logical_total = 0;
   nir_foreach_function(f, shader) {
      if (!f->impl)
         continue;
      nir_foreach_block(block, f->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(intr->intrinsic))
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (f->impl != impl || !deref)
               do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
            /* A query is a scalar variable or one element of a
             * one-dimensional query array. */
            nir_deref_instr *root = deref;
            if (root->deref_type == nir_deref_type_array)
               root = nir_deref_instr_parent(root);
            if (root->deref_type != nir_deref_type_var ||
                !root->var->data.ray_query ||
                (glsl_type_is_array(root->var->type) &&
                 (glsl_type_is_array(glsl_get_array_element(root->var->type)) ||
                  glsl_get_length(root->var->type) == 0)) ||
                (deref != root && !glsl_type_is_array(root->var->type)))
               do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
            struct m5_rq_state *owner_state = NULL;
            util_dynarray_foreach(&queries, struct m5_rq_state, state) {
               if (state->query == root->var)
                  owner_state = state;
            }
            if (!owner_state) {
               struct m5_rq_state state = {
                  .query = root->var,
                  .first_logical = logical_total,
                  .logical_count = glsl_type_is_array(root->var->type)
                     ? glsl_get_length(root->var->type) : 1,
               };
               logical_total += state.logical_count;
               util_dynarray_append(&queries, state);
               owner_state = util_dynarray_top_ptr(&queries, struct m5_rq_state);
            }
            if (deref != root) {
               if (!nir_src_is_const(deref->arr.index))
                  owner_state->dynamic = true;
               else if (nir_src_as_uint(deref->arr.index) >=
                        owner_state->logical_count)
                  do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
            }
            switch (intr->intrinsic) {
            case nir_intrinsic_rq_initialize:
               /* Candidate policy is evaluated per invocation, so valid
                * dynamically selected Vulkan flags need no specialization.
                * Retain the narrower normal-path contract until qualified. */
               if ((!m5_candidate_api() && !nir_src_is_const(intr->src[2])) ||
                   (nir_src_is_const(intr->src[2]) &&
                    ((nir_src_as_uint(intr->src[2]) &
                      ~(m5_candidate_api() ? 0x3ffu : 0x3du)) ||
                     ((nir_src_as_uint(intr->src[2]) & 0x30u) == 0x30u) ||
                     ((nir_src_as_uint(intr->src[2]) & 3u) == 3u))) ||
                   intr->src[1].ssa->bit_size != 64)
                  do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
               break;
            case nir_intrinsic_rq_load:
               if (nir_def_is_unused(&intr->def))
                  break;
               if (!m5_rq_supported_load(intr) ||
                   (!layout->instance_traversal &&
                    (nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_instance_id ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_instance_custom_index ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_front_face ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_instance_sbt_index ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_object_ray_origin ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_object_ray_direction ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_object_to_world ||
                     nir_intrinsic_ray_query_value(intr) == nir_ray_query_value_intersection_world_to_object)))
                  do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
               break;
            case nir_intrinsic_rq_proceed:
            case nir_intrinsic_rq_terminate:
               break;
            case nir_intrinsic_rq_confirm_intersection:
            case nir_intrinsic_rq_generate_intersection:
               if (!m5_candidate_api())
                  do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
               break;
            default:
               do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
            }
            util_dynarray_append(&ops, intr);
         }
      }
   }
   if (!util_dynarray_num_elements(&queries, struct m5_rq_state))
      do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
   if (m5_candidate_api() && !layout->instance_traversal)
      do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
   /* The ordinary path keeps one native slot per query variable. */
   if (!m5_candidate_api() &&
       util_dynarray_num_elements(&queries, struct m5_rq_state) >
          AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES)
      do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);
   if (m5_candidate_api() &&
       util_dynarray_num_elements(&queries, struct m5_rq_state) > 1 &&
       debug_get_bool_option("ASAHI_M5_RQ_INLINE_PACKET", false))
      do { if (debug_get_bool_option("ASAHI_M5_RQ_REFUSAL_DIAG", false)) fprintf(stderr, "M5-RQ refusal line%d queries=%zu\n", __LINE__, util_dynarray_num_elements(&queries, struct m5_rq_state)); goto unsupported; } while (0);

   const unsigned query_count =
      util_dynarray_num_elements(&queries, struct m5_rq_state);
   /* Remove native-only state when every initializer of this query has a
    * constant policy that requires candidate traversal. Unknown or mixed
    * policies retain the native path; multiple owners remain ineligible. */
   if (m5_native_opaque() && shader->info.stage == MESA_SHADER_COMPUTE &&
       query_count == 1 && logical_total == 1 &&
       shader->info.ray_queries == 1) {
      util_dynarray_foreach(&queries, struct m5_rq_state, state) {
         util_dynarray_foreach(&ops, nir_intrinsic_instr *, op) {
            nir_intrinsic_instr *intr = *op;
            if (intr->intrinsic != nir_intrinsic_rq_initialize ||
                nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0])) != state->query)
               continue;
            state->native_opaque_possible |= !nir_src_is_const(intr->src[2]) ||
               !(nir_src_as_uint(intr->src[2]) & (2 | 4 | 0x40 | 0x100));
         }
      }
   }
   if (debug_get_bool_option("ASAHI_M5_RQ_CODEGEN_DIAG", false))
      fprintf(stderr, "M5-RQ codegen queries=%u logical=%u replay=%u\n",
              query_count, logical_total, logical_total > 1);

   /* The logical query whose traversal is resident in the shared native
    * payload. Logical queries can share a native slot, so residency is
    * tracked here as well as through the published owner token. */
   nir_variable *resident_logical = NULL;
   if (m5_candidate_api() && logical_total > 1) {
      resident_logical = nir_local_variable_create(impl, glsl_uint_type(),
                                                   "rq_resident_logical");
      nir_builder init = nir_builder_at(nir_before_impl(impl));
      nir_store_var(&init, resident_logical, nir_imm_int(&init, ~0u), 1);
   }
   bool any_dynamic = false;
   util_dynarray_foreach(&queries, struct m5_rq_state, state) {
      any_dynamic |= state->dynamic;
      state->as = m5_rq_var(shader, impl, state, glsl_uint64_t_type(), "rq_as");
      state->origin = m5_rq_var(shader, impl, state,
         glsl_vector_type(GLSL_TYPE_FLOAT, 3), "rq_origin");
      state->direction = m5_rq_var(shader, impl, state,
         glsl_vector_type(GLSL_TYPE_FLOAT, 3), "rq_direction");
      state->tmin = m5_rq_var(shader, impl, state, glsl_float_type(), "rq_tmin");
      state->tmax = m5_rq_var(shader, impl, state, glsl_float_type(), "rq_tmax");
      state->mask = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_mask");
      state->flags = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_flags");
      state->active = m5_rq_var(shader, impl, state, glsl_bool_type(), "rq_active");
      for (unsigned i = 0; i < 5; ++i)
         state->result[i] = m5_rq_var(shader, impl, state, glsl_uvec4_type(),
                                      "rq_packet");
      if (m5_candidate_api()) {
         state->started = m5_rq_var(shader, impl, state, glsl_bool_type(), "rq_started");
         state->owner = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_owner");
         state->resident_id = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_resident");
         state->cursor = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_cursor");
         /* Native next consumes same-owner completed queue entries before
          * reading completion. Activation and continuation finish their batch
          * before queue access; only queue exhaustion reads its status. Multiple owners and the
          * unqualified native opaque import keep the ordinary path. */
         if (logical_total == 1 && !state->native_opaque_possible &&
             debug_get_bool_option("ASAHI_M5_RQ_QUEUE_FIRST_PROBE", false))
            state->queue_first =
               !getenv("ASAHI_M5_RQ_ZERO_SPAN_DIAG_ADDRESS");
         /* Each singleton compute query owns its completed candidate batch.
          * Owner switching, graphics layouts and the raw-status recorder keep
          * their existing completion/import path. */
         if (logical_total == 1 && shader->info.stage == MESA_SHADER_COMPUTE &&
             !state->queue_first && !getenv("ASAHI_M5_RQ_ZERO_SPAN_DIAG_ADDRESS"))
            state->batch_completed = m5_rq_var(shader, impl, state,
               glsl_bool_type(), "rq_batch_completed");
         if (state->native_opaque_possible)
            state->initial_completion = m5_rq_var(shader, impl, state,
               glsl_bool_type(), "rq_initial_completion");
         /* A single logical query cannot lose its owner to another query.
          * Reinitialization clears traversal state, so there is no candidate
          * prefix to restore. Keep replay state only for multiple owners. */
         if (logical_total > 1) {
            state->processed = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_processed");
            state->replay = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_replay");
            state->commit_history = m5_rq_var(shader, impl, state, glsl_uint_type(), "rq_commit_history");
            state->prune_t = m5_rq_var(shader, impl, state, glsl_float_type(), "rq_prune_t");
         }
         if (m5_model_capture_address())
            state->capture_event = m5_rq_var(shader, impl, state,
               glsl_uint_type(), "rq_capture_event");
         for (unsigned i = 0; i < 5; ++i)
            state->candidate[i] = m5_rq_var(shader, impl, state,
               glsl_uvec4_type(), "rq_candidate");
      }
   }

   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&ops, nir_intrinsic_instr *, op) {
      nir_intrinsic_instr *intr = *op;
      if (intr->intrinsic == nir_intrinsic_rq_load &&
          nir_def_is_unused(&intr->def)) {
         nir_instr_remove(&intr->instr);
         continue;
      }
      b.cursor = nir_before_instr(&intr->instr);
      nir_deref_instr *query = nir_src_as_deref(intr->src[0]);
      struct m5_rq_shadows shadows = {.impl = impl, .query = query};
      struct m5_rq_state *state = NULL;
      util_dynarray_foreach(&queries, struct m5_rq_state, candidate) {
         if (candidate->query == nir_deref_instr_get_variable(query)) {
            state = candidate;
            break;
         }
      }
      assert(state);
      nir_deref_instr *as = m5_rq_op_field(&b, &shadows, state->as);
      nir_deref_instr *origin = m5_rq_op_field(&b, &shadows, state->origin);
      nir_deref_instr *direction = m5_rq_op_field(&b, &shadows, state->direction);
      nir_deref_instr *tmin = m5_rq_op_field(&b, &shadows, state->tmin);
      nir_deref_instr *tmax = m5_rq_op_field(&b, &shadows, state->tmax);
      nir_deref_instr *mask = m5_rq_op_field(&b, &shadows, state->mask);
      nir_deref_instr *active = m5_rq_op_field(&b, &shadows, state->active);
      nir_deref_instr *result[5];
      for (unsigned i = 0; i < 5; ++i)
         result[i] = m5_rq_op_field(&b, &shadows, state->result[i]);
      nir_deref_instr *started = NULL, *resident_id = NULL, *cursor = NULL, *owner = NULL;
      nir_deref_instr *batch_completed = state->batch_completed
         ? m5_rq_op_field(&b, &shadows, state->batch_completed) : NULL;
      nir_deref_instr *candidate[5] = {0};
      nir_deref_instr *initial_completion = state->initial_completion
         ? m5_rq_op_field(&b, &shadows, state->initial_completion) : NULL;
      nir_deref_instr *capture_event = state->capture_event
         ? m5_rq_op_field(&b, &shadows, state->capture_event) : NULL;
      if (m5_candidate_api()) {
         started = m5_rq_op_field(&b, &shadows, state->started);
         owner = m5_rq_op_field(&b, &shadows, state->owner);
         resident_id = m5_rq_op_field(&b, &shadows, state->resident_id);
         cursor = m5_rq_op_field(&b, &shadows, state->cursor);
         for (unsigned i = 0; i < 5; ++i)
            candidate[i] = m5_rq_op_field(&b, &shadows, state->candidate[i]);
      }
      nir_deref_instr *processed = state->processed ? m5_rq_op_field(&b, &shadows, state->processed) : NULL;
      nir_deref_instr *replay = state->replay ? m5_rq_op_field(&b, &shadows, state->replay) : NULL;
      nir_deref_instr *history = state->commit_history ? m5_rq_op_field(&b, &shadows, state->commit_history) : NULL;
      nir_deref_instr *prune_t = state->prune_t ? m5_rq_op_field(&b, &shadows, state->prune_t) : NULL;
      nir_deref_instr *flags = m5_rq_op_field(&b, &shadows, state->flags);
      nir_def *value = NULL;
      switch (intr->intrinsic) {
      case nir_intrinsic_rq_initialize:
         if (m5_candidate_api()) {
            nir_store_deref(&b, started, nir_imm_false(&b), 1);
            if (batch_completed)
               nir_store_deref(&b, batch_completed, nir_imm_false(&b), 1);
            if (capture_event)
               nir_store_deref(&b, capture_event, nir_imm_int(&b, 0), 1);
            /* Cursor index 0; with m5_rq_watertight also a NaN boundary
             * distance key, which matches no candidate distance. */
            nir_store_deref(&b, cursor,
               nir_imm_int(&b, m5_rq_watertight() ? 0x7ffffff0 : 0), 1);
            if (processed) {
               nir_store_deref(&b, processed, nir_imm_int(&b, 0), 1);
               nir_store_deref(&b, replay, nir_imm_int(&b, 0), 1);
               nir_store_deref(&b, history, nir_imm_int(&b, 0), 1);
               nir_store_deref(&b, prune_t, intr->src[7].ssa, 1);
            }
            for (unsigned i = 0; i < 5; ++i)
               nir_store_deref(&b, candidate[i], nir_imm_ivec4(&b, 0, 0, 0, 0), 15);
         }
         nir_store_deref(&b, flags, intr->src[2].ssa, 1);
         nir_store_deref(&b, as, intr->src[1].ssa, 1);
         nir_store_deref(&b, origin, intr->src[4].ssa, 7);
         nir_store_deref(&b, direction, intr->src[6].ssa, 7);
         nir_store_deref(&b, tmin, intr->src[5].ssa, 1);
         nir_store_deref(&b, tmax, intr->src[7].ssa, 1);
         /* Only the low eight cull-mask bits take part in traversal. */
         nir_store_deref(&b, mask, nir_iand_imm(&b, intr->src[3].ssa, 0xff), 1);
         nir_def *has_as = nir_ine_imm(&b, intr->src[1].ssa, 0);
         if (m5_candidate_api()) {
            /* SkipTriangles (0x100) / SkipAABBs (0x200) are applied per
             * candidate kind in m5_candidate_next; the candidate profile
             * forces non-opaque traversal (control 0x402), so every triangle
             * reaches that filter. Metal's assume_geometry_type compiles to
             * byte-identical code for every mask (witness-geom), so no packet
             * level geometry filter is known. */
         } else {
            /* The opaque profile commits triangles in hardware and carries
             * triangle leaves only: SkipTriangles makes it empty. */
            has_as = nir_iand(&b, has_as,
               nir_inot(&b, nir_test_mask(&b, intr->src[2].ssa, 0x100)));
         }
         nir_store_deref(&b, active, has_as, 1);
         if (layout->context_uniform) {
            /* The driver-owned empty header clears readiness; ordinary
             * native envelopes publish 1 at AS byte0xa0. Do this before any
             * query entry, packet publication, or intersector operation. */
            nir_push_if(&b, has_as);
            nir_def *ready = nir_load_global(&b, 1, 32,
               nir_iadd_imm(&b, intr->src[1].ssa, 0xa0), .align_mul = 4);
            nir_store_deref(&b, active, nir_ine_imm(&b, ready, 0), 1);
            nir_pop_if(&b, NULL);
         }
         for (unsigned i = 0; i < 5; ++i)
            nir_store_deref(&b, result[i], nir_imm_ivec4(&b, 0, 0, 0, 0), 15);
         /* Committed distance is initialized to tmax before traversal. */
         nir_store_deref(&b, result[1],
                       nir_vec4(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                intr->src[7].ssa, nir_imm_int(&b, 0)),
                       15);
         break;
      case nir_intrinsic_rq_proceed: {
         nir_variable *more = NULL;
         if (m5_candidate_api()) {
            more = nir_local_variable_create(impl, glsl_bool_type(), "rq_more");
            nir_store_var(&b, more, nir_imm_false(&b), 1);
         }
         const bool fresh_query = m5_candidate_mode() &&
            !debug_get_bool_option("ASAHI_M5_RQ_INLINE_PACKET", false);
         nir_push_if(&b, nir_load_deref(&b, active));
         if (m5_candidate_api()) {
            nir_push_if(&b, nir_inot(&b, nir_load_deref(&b, started)));
            nir_store_deref(&b, owner,
               fresh_query ? m5_rq_enter(&b, state, query)
                           : nir_imm_int(&b, 0), 1);
            nir_store_deref(&b, resident_id, nir_load_scs_thread_index_agx(&b), 1);
            nir_pop_if(&b, NULL);
            m5_model_capture(&b, state, capture_event, owner, resident_id, cursor, active,
                             result[1], 1);
            nir_def *activate = nir_inot(&b, nir_load_deref(&b, started));
            nir_def *logical = NULL;
            if (logical_total > 1) {
               unsigned constant_logical;
               logical = m5_rq_logical(&b, state, query, &constant_logical);
               if (!logical)
                  logical = nir_imm_int(&b, constant_logical);
               /* Another logical query took the payload since this one
                * last traversed, possibly through the same native slot. */
               activate = nir_ior(&b, activate,
                  nir_ine(&b, nir_load_var(&b, resident_logical), logical));
               nir_def *saved_id = nir_load_deref(&b, resident_id);
               nir_def *driver = nir_iadd(&b,
                  nir_load_preamble(&b, 1, 32, .base = 56),
                  nir_imul_imm(&b, nir_iand_imm(&b, saved_id, ~31u), m5_candidate_driver_stride()));
               nir_def *control = nir_load_scs_indexed_agx(&b, 1, 32, driver,
                  nir_iand_imm(&b, saved_id, 31), .base = 1728, .component = 2);
               nir_def *resident_owner = nir_iand_imm(&b,
                  nir_ushr_imm(&b, control, 1), 0xfffff);
               activate = nir_ior(&b, activate, nir_ine(&b, resident_owner,
                  nir_iand_imm(&b, nir_load_deref(&b, owner), 0xfffff)));
            }
            /* Native next reactivates a query through intersect_ray when the
             * shared payload belongs to a different query. Its token selects
             * the saved traversal checkpoint; each logical query keeps its
             * committed/candidate data in separate SSA variables. The single
             * query path needs no ownership load or extra activation branch. */
            nir_push_if(&b, activate);
            if (resident_logical)
               nir_store_var(&b, resident_logical, logical, 1);
            if (replay) {
               nir_store_deref(&b, replay, nir_load_deref(&b, processed), 1);
               nir_store_deref(&b, prune_t, nir_load_deref(&b, tmax), 1);
            }
         }
         nir_def *address = nir_load_deref(&b, as);
         nir_def *selector = nir_load_preamble(&b, 1, 64, .base = layout->bvh_uniform);
         if (layout->context_uniform) {
            nir_def *context = nir_load_preamble(&b, 1, 64,
                                                  .base = layout->context_uniform);
            for (unsigned i = 0; i < (layout->instance_traversal ? 8 : 6); ++i) {
               nir_def *words = nir_load_global(&b, 4, 32,
                  nir_iadd_imm(&b, context, 16 * i), .align_mul = 16);
               nir_store_scs_agx(&b, words, .base = 0x6000 + 16 * i);
            }
         }
         if (layout->instance_traversal) {
            selector = nir_u2u64(&b, nir_load_global(&b, 1, 32,
               nir_iadd_imm(&b, address, 0xb0), .align_mul = 4));
         } else if (layout->as_table_uniform) {
            nir_def *entry = nir_iadd_imm(&b, nir_load_scs_thread_index_agx(&b), 1);
            nir_def *table = nir_load_preamble(&b, 1, 64,
                                                .base = layout->as_table_uniform);
            nir_store_global(&b, address, nir_iadd(&b, table,
               nir_u2u64(&b, nir_ishl_imm(&b, entry, 4))),
               .align_mul = 16, .write_mask = 1);
            nir_barrier(&b, .execution_scope = SCOPE_INVOCATION,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global);
            selector = nir_u2u64(&b, entry);
         }
         /* Entry produces a real GPR value. Tracking it through SSA/RA also
          * makes any SR59 value live across entry safe from an implicit r0
          * clobber; no architectural change to SR59 is assumed. */
         nir_def *query_owner = m5_candidate_api()
            ? nir_load_deref(&b, owner)
            : fresh_query ? nir_enter_query_agx(&b, .base = 4)
                          : nir_imm_int(&b, 0);
         nir_def *lo = nir_load_global(
            &b, 3, 32, nir_iadd_imm(&b, address, 0x50), .align_mul = 16);
         nir_def *hi = nir_load_global(
            &b, 3, 32, nir_iadd_imm(&b, address, 0x5c), .align_mul = 4);
         nir_def *base =
            nir_load_preamble(&b, 1, 64, .base = layout->payload_uniform);
         nir_def *id = m5_candidate_api() ? nir_load_deref(&b, resident_id)
                                           : nir_load_scs_thread_index_agx(&b);
         nir_def *index = nir_ishl_imm(&b, id, 6);
         nir_def *ray_flags = nir_load_deref(&b, flags);
         nir_def *opaque = nir_test_mask(&b, ray_flags, 1);
         nir_def *first_hit = nir_test_mask(&b, ray_flags, 4);
         nir_def *query_control = nir_bcsel(&b, opaque,
            nir_imm_int(&b, 0x122), nir_imm_int(&b, 0x120));
         if (fresh_query)
            base = nir_load_preamble(&b, 1, 64, .base = 152);
         nir_def *native_opaque_eligible = state->native_opaque_possible
            ? m5_native_opaque_eligible(&b, ray_flags) : nir_imm_false(&b);
         m5_init_triangle_packet(
            &b, base, index, nir_load_deref(&b, origin),
            nir_load_deref(&b, direction), nir_load_deref(&b, tmin),
            prune_t ? nir_load_deref(&b, tmax) : m5_candidate_api() ? nir_channel(&b, nir_load_deref(&b, result[1]), 2)
                               : nir_load_deref(&b, tmax),
            lo, hi, nir_load_deref(&b, mask),
            layout->instance_traversal ? nir_imm_int(&b, 0) : query_control,
            opaque, m5_candidate_api() ? nir_imm_false(&b) : first_hit,
            nir_ishl_imm(&b, nir_iand_imm(&b, ray_flags, 0x30), 4),
            native_opaque_eligible);
         nir_def *instance_type = NULL, *instance_bary = NULL, *instance_metadata = NULL, *instance_face = NULL;
         nir_def *candidate_distance = NULL;
         if (layout->instance_traversal)
            m5_trace_instance(&b, selector, id, opaque, query_owner,
                               nir_load_deref(&b, tmin), native_opaque_eligible, &instance_type, &instance_bary, &instance_metadata, &instance_face,
                               &candidate_distance);
         else {
            agx_nir_m5_trace_opaque(
               &b, nir_imm_int64(&b, 0), selector, base, index,
               nir_load_preamble(&b, 1, 32, .base = layout->status_uniform),
               nir_vec3(&b, nir_imm_int(&b, layout->launch_header),
                        nir_imm_int(&b, 0), id),
               NULL, NULL);
         }
         if (m5_candidate_api()) {
            if (initial_completion)
               nir_store_deref(&b, initial_completion,
                  m5_defer_completion(&b) ? nir_imm_true(&b) : native_opaque_eligible, 1);
            nir_store_deref(&b, resident_id, id, 1);
            nir_store_deref(&b, started, nir_imm_true(&b), 1);
            m5_cursor_set_index(&b, cursor, nir_imm_int(&b, 0));
            nir_pop_if(&b, NULL);
            m5_candidate_next(&b, active, resident_id, cursor, result,
                              tmin, flags, candidate, initial_completion, batch_completed, state->queue_first, more, processed, replay, history, prune_t);
            m5_model_capture(&b, state, capture_event, owner, resident_id, cursor, active,
                             result[1], 2);
            nir_pop_if(&b, NULL);
            value = nir_load_var(&b, more);
            break;
         }
         for (unsigned i = 0; i < 4; ++i)
            nir_store_deref(&b, result[i],
                          nir_load_scs_uniform_agx(
                             &b, 4, 32, base, index, .base = 16 * i,
                             .format = PIPE_FORMAT_R32_UINT, .component = 0),
                          15);
         if (layout->instance_traversal) {
            if (candidate_distance)
               nir_store_deref(&b, result[1], nir_vector_insert_imm(&b,
                  nir_load_deref(&b, result[1]), candidate_distance, 2), 15);
            nir_store_deref(&b, result[4], instance_metadata, 15);
            nir_store_deref(&b, result[2], nir_vec4(&b,
               instance_face, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
               instance_type), 15);
            nir_store_deref(&b, result[3], nir_vec4(&b, nir_imm_int(&b, 0),
               nir_channel(&b, instance_bary, 0),
               nir_channel(&b, instance_bary, 1), nir_imm_int(&b, 0)), 15);
         }
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         nir_pop_if(&b, NULL);
         /* Opaque traversal consumes the whole query, with no candidate for
          * application code to confirm. Subsequent proceed calls are inert. */
         value = nir_imm_false(&b);
         break;
      }
      case nir_intrinsic_rq_terminate:
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         break;
      case nir_intrinsic_rq_confirm_intersection:
         m5_replay_commit(&b, processed, history, prune_t,
            nir_channel(&b, nir_load_deref(&b, candidate[1]), 2));
         for (unsigned i = 0; i < 5; ++i)
            nir_store_deref(&b, result[i], nir_load_deref(&b, candidate[i]), 15);
         nir_push_if(&b, nir_test_mask(&b, nir_load_deref(&b, flags), 4));
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         nir_pop_if(&b, NULL);
         break;
      case nir_intrinsic_rq_generate_intersection: {
         /* Commit the candidate AABB at the application's distance when it
          * lies inside [tmin, committed t]; the candidate packet already
          * carries the generated (bounding box) kind in its type bits. The
          * committed distance is published to the native packet before the
          * next continuation by m5_candidate_next. */
         nir_def *t = intr->src[1].ssa;
         nir_def *best = nir_channel(&b, nir_load_deref(&b, result[1]), 2);
         nir_push_if(&b, nir_iand(&b, nir_fge(&b, best, t),
                                  nir_fge(&b, t, nir_load_deref(&b, tmin))));
         /* Gen2 commit stores the application's distance at112, but saves
          * the original candidate distance at104. Prefix replay publishes
          * the latter at116, preserving the native candidate order while
          * the API's generated distance remains private. */
         m5_replay_commit(&b, processed, history, prune_t,
            nir_channel(&b, nir_load_deref(&b, candidate[1]), 2));
         for (unsigned i = 0; i < 5; ++i)
            nir_store_deref(&b, result[i], nir_load_deref(&b, candidate[i]), 15);
         nir_store_deref(&b, result[1], nir_vector_insert_imm(&b,
            nir_load_deref(&b, result[1]), t, 2), 15);
         nir_push_if(&b, nir_test_mask(&b, nir_load_deref(&b, flags), 4));
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         nir_pop_if(&b, NULL);
         nir_pop_if(&b, NULL);
         break;
      }
      case nir_intrinsic_rq_load:
         if (!nir_intrinsic_committed(intr) && m5_candidate_api())
            for (unsigned i = 0; i < 5; ++i)
               result[i] = candidate[i];
         switch (nir_intrinsic_ray_query_value(intr)) {
         case nir_ray_query_value_flags:
            value = nir_load_deref(&b, flags);

            break;
         case nir_ray_query_value_tmin:
            value = nir_load_deref(&b, tmin);
            break;
         case nir_ray_query_value_world_ray_origin:
            value = nir_load_deref(&b, origin);
            break;
         case nir_ray_query_value_world_ray_direction:
            value = nir_load_deref(&b, direction);
            break;
         case nir_ray_query_value_intersection_type:
            value = nir_ushr_imm(
               &b, nir_channel(&b, nir_load_deref(&b, result[2]), 3), 30);
            if (!nir_intrinsic_committed(intr))
               value = nir_iadd_imm(&b, value, -1);
            break;
         case nir_ray_query_value_intersection_t:
            value = nir_channel(&b, nir_load_deref(&b, result[1]), 2);
            break;
         case nir_ray_query_value_intersection_primitive_index:
            value = nir_iand_imm(
               &b, nir_channel(&b, nir_load_deref(&b, result[2]), 3), 0x3fffffff);
            break;
         case nir_ray_query_value_intersection_geometry_index:
            /* Native geometry-id witness masks the second hit word read
             * together with primitive/type at driver+256. */
            value = nir_iand_imm(&b,
               nir_channel(&b, nir_load_deref(&b, result[2]), 0), 0xffffff);
            break;
         case nir_ray_query_value_intersection_front_face:
            value = nir_ilt_imm(&b, nir_channel(&b, nir_load_deref(&b, result[2]), 0), 0);
            break;
         case nir_ray_query_value_intersection_candidate_aabb_opaque:
            value = nir_ine_imm(&b, nir_channel(&b, nir_load_deref(&b, result[0]), 0), 0);
            break;
         case nir_ray_query_value_intersection_instance_sbt_index:
         case nir_ray_query_value_intersection_object_ray_direction:
         case nir_ray_query_value_intersection_object_ray_origin:
         case nir_ray_query_value_intersection_object_to_world:
         case nir_ray_query_value_intersection_world_to_object: {
            nir_def *node = nir_pack_64_2x32(&b,
               nir_channels(&b, nir_load_deref(&b, result[4]), 3));
            const nir_ray_query_value field = nir_intrinsic_ray_query_value(intr);
            if (field == nir_ray_query_value_intersection_instance_sbt_index ||
                field == nir_ray_query_value_intersection_object_to_world) {
               nir_def *relative = nir_load_global(&b, 1, 64,
                  nir_iadd_imm(&b, node, 104), .align_mul = 8);
               nir_def *transform = nir_iadd(&b, node, relative);
               if (field == nir_ray_query_value_intersection_instance_sbt_index) {
                  value = nir_iand_imm(&b, nir_load_global(&b, 1, 32,
                     nir_iadd_imm(&b, transform, 48), .align_mul = 4), 0xffffff);
               } else {
                  value = nir_load_global(&b, 3, 32,
                     nir_iadd_imm(&b, transform, 12 * nir_intrinsic_column(intr)),
                     .align_mul = 4);
               }
            } else if (field == nir_ray_query_value_intersection_world_to_object) {
               value = nir_load_global(&b, 3, 32,
                  nir_iadd_imm(&b, node, 16 + 12 * nir_intrinsic_column(intr)),
                  .align_mul = 4);
            } else {
               /* Native node stores four packed vec3 columns of world-to-
                * object at16. Translation applies to origins only. */
               const bool is_origin =
                  field == nir_ray_query_value_intersection_object_ray_origin;
               nir_def *ray = nir_load_deref(&b, is_origin ? origin : direction);
               nir_def *column[4];
               for (unsigned c = 0; c < 4; ++c)
                  column[c] = nir_load_global(&b, 3, 32,
                     nir_iadd_imm(&b, node, 16 + 12 * c), .align_mul = 4);
               nir_def *component[3];
               for (unsigned r = 0; r < 3; ++r) {
                  component[r] = is_origin ? nir_channel(&b, column[3], r)
                                           : nir_imm_float(&b, 0);
                  for (unsigned c = 0; c < 3; ++c)
                     component[r] = nir_ffma(&b, nir_channel(&b, ray, c),
                        nir_channel(&b, column[c], r), component[r]);
               }
               value = nir_vec3(&b, component[0], component[1], component[2]);
            }
            break;
         }
         case nir_ray_query_value_intersection_instance_id:
            value = nir_channel(&b, nir_load_deref(&b, result[4]), 2);
            break;
         case nir_ray_query_value_intersection_instance_custom_index:
            value = nir_channel(&b, nir_load_deref(&b, result[4]), 3);
            break;
         case nir_ray_query_value_intersection_barycentrics:
            value = nir_channels(&b, nir_load_deref(&b, result[3]), 6);
            break;
         default:
            UNREACHABLE("preflighted M5 query load");
         }
         break;
      default:
         UNREACHABLE("preflighted M5 query operation");
      }
      m5_rq_op_writeback(&b, &shadows, intr->intrinsic, state);
      if (value)
         nir_def_rewrite_uses(&intr->def, value);
      nir_instr_remove(&intr->instr);
   }
   util_dynarray_fini(&ops);
   util_dynarray_foreach(&queries, struct m5_rq_state, state)
      state->query->data.ray_query = false;
   util_dynarray_fini(&queries);
   shader->info.ray_queries = 0;
   nir_progress(true, impl, nir_metadata_none);
   if (any_dynamic) {
      NIR_PASS(_, shader, nir_lower_var_copies);
      NIR_PASS(_, shader, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_shader_temp, ~0u);
   }
   NIR_PASS(_, shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, shader, nir_opt_dce);
   NIR_PASS(_, shader, nir_remove_dead_variables,
            nir_var_shader_temp | nir_var_function_temp, NULL);
   return true;
unsupported:
   util_dynarray_fini(&ops);
   util_dynarray_fini(&queries);
   return false;
}

/* Native query operations only touch the invocation's SCS stack and the
 * per-query state they own; they have no observer other than other query
 * operations. */
static bool
m5_rq_native_query_op(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_enter_query_agx:
   case nir_intrinsic_intersect_ray_agx:
   case nir_intrinsic_continue_ray_agx:
   case nir_intrinsic_load_scs_agx:
   case nir_intrinsic_store_scs_agx:
   case nir_intrinsic_load_scs_indexed_agx:
   case nir_intrinsic_store_scs_indexed_agx:
   case nir_intrinsic_load_scs_uniform_agx:
   case nir_intrinsic_store_scs_uniform_agx:
   case nir_intrinsic_load_scs_base_agx:
   case nir_intrinsic_store_scs_base_agx:
   case nir_intrinsic_load_scs_register_agx:
   case nir_intrinsic_load_scs_thread_index_agx:
      return true;
   default:
      return false;
   }
}

struct m5_rq_taint {
   struct set *loops;
   bool progress;
   bool external;
};

static bool
m5_rq_src_clean(nir_src *src, void *data)
{
   return !nir_def_instr(src->ssa)->pass_flags;
}

static void
m5_rq_taint_instr(nir_instr *instr, struct m5_rq_taint *t)
{
   if (!instr->pass_flags) {
      instr->pass_flags = 1;
      t->progress = true;
   }
}

/* Mark everything data- or control-dependent on a native query operation.
 * region is true inside control flow selected by such a value. Sets *jump
 * when a break/continue of the innermost enclosing loop is so selected. */
static void
m5_rq_taint_list(struct exec_list *list, bool region, bool *jump,
                 struct m5_rq_taint *t)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (region || m5_rq_native_query_op(instr) ||
                !nir_foreach_src(instr, m5_rq_src_clean, NULL))
               m5_rq_taint_instr(instr, t);
            if (instr->pass_flags && instr->type == nir_instr_type_jump) {
               nir_jump_type type = nir_instr_as_jump(instr)->type;
               if (type == nir_jump_break || type == nir_jump_continue)
                  *jump = true;
               else
                  t->external = true;
            }
         }
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         const bool selected = region ||
            nir_def_instr(nif->condition.ssa)->pass_flags;
         m5_rq_taint_list(&nif->then_list, selected, jump, t);
         m5_rq_taint_list(&nif->else_list, selected, jump, t);
         if (selected) {
            nir_block *after = nir_cf_node_as_block(nir_cf_node_next(node));
            nir_foreach_phi(phi, after)
               m5_rq_taint_instr(&phi->instr, t);
         }
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         bool own_jump = false;
         const bool selected = region || _mesa_set_search(t->loops, loop);
         m5_rq_taint_list(&loop->body, selected, &own_jump, t);
         if (nir_loop_has_continue_construct(loop))
            m5_rq_taint_list(&loop->continue_list, selected, &own_jump, t);
         if (!selected && own_jump) {
            /* The trip count depends on the query: every value produced in
             * the loop does too. */
            _mesa_set_add(t->loops, loop);
            t->progress = true;
         }
         /* So does which break left it: the phis after the loop merge one
          * value per break, and those values may be defined before it. */
         if (_mesa_set_search(t->loops, loop)) {
            nir_block *after = nir_cf_node_as_block(nir_cf_node_next(node));
            nir_foreach_phi(phi, after)
               m5_rq_taint_instr(&phi->instr, t);
         }
         break;
      }
      default:
         t->external = true;
         break;
      }
   }
}

static void
m5_rq_collect_ifs(struct exec_list *list, struct set *ifs)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nir_def_instr(nif->condition.ssa)->pass_flags)
            _mesa_set_add(ifs, nif);
         m5_rq_collect_ifs(&nif->then_list, ifs);
         m5_rq_collect_ifs(&nif->else_list, ifs);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         m5_rq_collect_ifs(&loop->body, ifs);
         if (nir_loop_has_continue_construct(loop))
            m5_rq_collect_ifs(&loop->continue_list, ifs);
      }
   }
}

/* Remove the outermost selected control flow. Selected instructions outside
 * it were removed already; the selected phis after it are gone too. */
static void
m5_rq_remove_selected_cf(struct exec_list *list, struct set *ifs,
                         struct set *loops)
{
   foreach_list_typed_safe(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (_mesa_set_search(ifs, nif)) {
            nir_cf_node_remove(node);
         } else {
            m5_rq_remove_selected_cf(&nif->then_list, ifs, loops);
            m5_rq_remove_selected_cf(&nif->else_list, ifs, loops);
         }
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (_mesa_set_search(loops, loop)) {
            nir_cf_node_remove(node);
         } else {
            m5_rq_remove_selected_cf(&loop->body, ifs, loops);
            if (nir_loop_has_continue_construct(loop))
               m5_rq_remove_selected_cf(&loop->continue_list, ifs, loops);
         }
      }
   }
}

bool
agx_nir_m5_remove_dead_ray_queries(nir_shader *shader)
{
   bool any = false;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            any |= m5_rq_native_query_op(instr);
      }
   }
   if (!any || !nir_shader_get_entrypoint(shader))
      return false;
   /* Calls would hide both query operations and side effects. */
   unsigned impls = 0;
   nir_foreach_function_impl(impl, shader)
      impls++;
   if (impls != 1)
      return false;

   /* The lowering's merges often carry one value from every path; fold them
    * first so they are not mistaken for query-selected values. */
   bool progress = false;
   NIR_PASS(progress, shader, nir_opt_remove_phis);
   NIR_PASS(progress, shader, nir_opt_copy_prop);
   NIR_PASS(progress, shader, nir_opt_dce);

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_shader_clear_pass_flags(shader);
   struct m5_rq_taint t = {
      .loops = _mesa_pointer_set_create(NULL),
   };
   do {
      bool jump = false;
      t.progress = false;
      m5_rq_taint_list(&impl->body, false, &jump, &t);
      /* A selected jump outside every loop would end the invocation. */
      t.external |= jump;
   } while (t.progress && !t.external);

   /* Every surviving side effect must be independent of the query. Only
    * native query operations may be selected; reorderable loads are pure. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (!instr->pass_flags || m5_rq_native_query_op(instr))
            continue;
         if (instr->type == nir_instr_type_call ||
             (instr->type == nir_instr_type_intrinsic &&
              !(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic]
                   .flags & NIR_INTRINSIC_CAN_ELIMINATE))) {
            if (!t.external &&
                debug_get_bool_option("ASAHI_M5_RQ_DEAD_DIAG", false)) {
               fprintf(stderr, "M5-RQ live query consumer: ");
               nir_print_instr(instr, stderr);
               fprintf(stderr, "\n");
               if (debug_get_bool_option("ASAHI_M5_RQ_DEAD_DIAG_SHADER", false))
                  nir_print_shader(shader, stderr);
            }
            t.external = true;
         }
      }
   }
   if (t.external) {
      _mesa_set_destroy(t.loops, NULL);
      nir_shader_clear_pass_flags(shader);
      return progress;
   }

   /* Selected values are only used by selected instructions. Detach them,
    * then remove selected instructions outside selected control flow (and
    * the phis that merge it), and finally that control flow itself. */
   struct set *ifs = _mesa_pointer_set_create(NULL);
   m5_rq_collect_ifs(&impl->body, ifs);
   struct util_dynarray defs = UTIL_DYNARRAY_INIT;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_def *def = instr->pass_flags ? nir_instr_def(instr) : NULL;
         if (def && !nir_def_is_unused(def))
            util_dynarray_append(&defs, def);
      }
   }
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   util_dynarray_foreach(&defs, nir_def *, def) {
      /* New instructions are zero-allocated: pass_flags stays clear. */
      nir_def_rewrite_uses(*def, nir_undef(&b, (*def)->num_components,
                                           (*def)->bit_size));
   }
   util_dynarray_fini(&defs);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->pass_flags && instr->type != nir_instr_type_jump)
            nir_instr_remove(instr);
      }
   }
   m5_rq_remove_selected_cf(&impl->body, ifs, t.loops);
   _mesa_set_destroy(ifs, NULL);
   _mesa_set_destroy(t.loops, NULL);
   nir_shader_clear_pass_flags(shader);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}
