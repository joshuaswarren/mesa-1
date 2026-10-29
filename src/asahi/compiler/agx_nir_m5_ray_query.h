/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler/nir/nir.h"
struct nir_builder;

/* Candidate layout witnessed in the G17G triangle/instancing query program.
 * This is the candidate-query driver profile, not the ordinary intersector's
 * 72-byte profile. The caller owns the profile, allocation and live queue.
 * Merely reading these records does not advance or commit traversal. */
struct agx_m5_ray_candidate {
   nir_def *compressed_address;
   nir_def *flags;
   nir_def *barycentrics;
   nir_def *distance;
};

/* Native triangle import boundary: initial tmax is inclusive; committed ties
 * keep the existing hit. Used by lowering and exact constant-fold regressions. */
nir_def *agx_nir_m5_should_import_triangle(struct nir_builder *b,
                                          nir_def *native_type, nir_def *prior_type,
                                          nir_def *distance, nir_def *best_t,
                                          nir_def *minimum);

nir_def *agx_nir_m5_candidate_count(struct nir_builder *b, nir_def *driver,
                                   nir_def *lane);
struct agx_m5_ray_candidate
agx_nir_m5_load_candidate(struct nir_builder *b, nir_def *driver,
                          nir_def *lane, nir_def *index);

/* Execute an already initialized M5 opaque-triangle query. The caller owns
 * the SCS packet and pointer table, BVH state, and resident intersection
 * programs. These addresses are SCS ABI values, not device virtual addresses.
 *
 * This is a compiler building block, not a public ray-query lowering pass.
 * Procedural/non-opaque candidates require their own accept/ignore protocol.
 * launch_words explicitly supplies staging halves 0/2/4; the caller must
 * derive these from its native query layout. They are not inferred from GPRs.
 * Optional counters are for validation and disappear when not requested.
 */
void agx_nir_m5_trace_opaque(struct nir_builder *b, nir_def *request,
                             nir_def *bvh_state, nir_def *payload_base,
                             nir_def *payload_index, nir_def *status_base,
                             nir_def *launch_words, nir_def **initial_pending,
                             nir_def **continuations);

/* Initialize the captured 64-byte, untransformed triangle-query packet.
 * query_flags is the caller's native configuration word (the retained opaque
 * reference uses 0x122). Address/continuation tables are initialized separately.
 * Inputs must be finite; negative t_min and invalid direction handling belong
 * to the caller. Bounds describe the acceleration structure's root AABB.
 */
void agx_nir_m5_init_triangle_packet(struct nir_builder *b,
                                     nir_def *payload_base,
                                     nir_def *payload_index, nir_def *origin,
                                     nir_def *direction, nir_def *t_min,
                                     nir_def *t_max, nir_def *bounds_min,
                                     nir_def *bounds_max, nir_def *mask,
                                     nir_def *query_flags, nir_def *opaque,
                                     nir_def *first_hit, nir_def *triangle_cull);
/* Explicit per-invocation launch ABI for the M5 native query pipeline.
 * These are half-word uniform indices, not GPU addresses. The caller must bind
 * the same native AS to bvh_uniform and the rq_initialize AS operand, provide
 * the resident IFT, and allocate the per-invocation SCS packet/status region.
 * This interface does not advertise a Vulkan feature or create that state.
 */
struct agx_m5_ray_query_layout {
   unsigned payload_uniform;
   unsigned bvh_uniform;
   unsigned status_uniform;
   uint32_t launch_header;
   /* Optional per-dispatch native context and AS pointer table. Zero keeps
    * the caller-initialized ABI used by standalone compiler probes. */
   unsigned context_uniform;
   unsigned as_table_uniform;
   bool instance_traversal;
   bool compute_kernel;
};

/* Lower standard NIR rq_* operations for opaque triangle queries and arrays.
 * Each query retains its own parameters and committed packet; interleaved
 * queries may reuse the physical traversal packet after a proceed completes.
 * Unsupported operations/flags are rejected before changing the shader.
 * Candidate enumeration, instances, and other query layouts still
 * need separate lowering; callers must not silently substitute this pass.
 */
/* Split ray-query arrays that are only accessed with constant indices into
 * separate ray-query variables, so each element can be range-merged and
 * mapped to its own native query context. */
/* Queue-exhaustion predicate shared with the production lowering and host
 * regression: caller has already consumed this completed batch. */
nir_def *agx_nir_m5_should_resume_batch(struct nir_builder *b, nir_def *status,
                                         nir_def *control, nir_def *count);

bool agx_nir_m5_split_ray_query_arrays(nir_shader *shader);

bool agx_nir_lower_m5_ray_query(nir_shader *shader,
                                const struct agx_m5_ray_query_layout *layout);

/* Remove an already lowered native query whose results cannot reach any
 * remaining memory, output or vertex-emission side effect. Geometry-shader
 * lowering clones the query into kernels from which it strips the API
 * stores; the query there is dead work. Native query operations are removed
 * only when every one of them is provably dead; otherwise only the
 * value-preserving phi/copy cleanup the analysis needs is applied. Returns
 * progress. */
bool agx_nir_m5_remove_dead_ray_queries(nir_shader *shader);
