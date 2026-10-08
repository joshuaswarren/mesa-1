/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Recorded-dependency tracking for Honeykrisp control streams.
 *
 * A command buffer is a list of control streams, each submitted as one
 * render (VDM) or compute (CDM) command. The kernel orders commands through
 * two per-command fields, vdm_barrier and cdm_barrier, each naming a prefix
 * of the respective subqueue that must complete before the command runs
 * (include/drm-uapi/asahi_drm.h, drm_asahi_cmd_header). The value NONE means
 * the command waits on nothing on that subqueue, not even earlier submits.
 *
 * Vulkan orders work through explicit dependencies: pipeline barriers,
 * events, subpass dependencies (which the common render-pass code lowers to
 * pipeline barriers) and semaphores. Every one of them is prefix-shaped: the
 * first synchronization scope of vkCmdPipelineBarrier2 is "all commands
 * earlier in submission order" restricted to the source stages, and the
 * second is "all commands later in submission order" restricted to the
 * destination stages (Vulkan 1.3, 7.6 "Pipeline Barriers"). The kernel's
 * prefix encoding therefore loses nothing against the barrier model on one
 * queue; what it cannot express is stage granularity (a barrier from vertex
 * work into fragment work still waits on and gates whole commands), and it
 * cannot distinguish two commands in the same prefix. Both would need UAPI
 * extensions.
 *
 * This module records, per stream, the latest earlier stream on each
 * subqueue that a recorded dependency orders it after ("cut"), plus which
 * kinds of pre-command-buffer work it must wait on ("prior"). Kinds
 * approximate the Vulkan stage classes at the granularity the hardware has:
 * graphics, compute and transfer. Streams the driver adds on its own
 * (queries, timestamps, events, heap management) carry the implicit orderings
 * Vulkan requires of them; a stream a meta operation records inherits the
 * application dependency it stands in for because it is created under the
 * same recorded state.
 *
 * At submission the cuts become exact prefix indices, cross-command-buffer
 * dependencies are resolved against the other command buffers in the same
 * submission. With EXACT_PRIOR_BARRIERS, dependencies reaching before this
 * submission name exact queue-local hardware ordinals; destination requirements
 * persist across submissions. Older kernels retain the "all earlier submits"
 * representation and existing per-submit requirement lifetime.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;
struct hk_queue;

/* Which class of Vulkan pipeline stages a stream's work belongs to. Streams
 * the driver adds for its own bookkeeping are HK_SYNC_ALL: every source scope
 * covers them and they wait as their own semantics require.
 */
enum hk_sync_kind {
   HK_SYNC_GFX = 1 << 0,
   HK_SYNC_COMPUTE = 1 << 1,
   HK_SYNC_XFER = 1 << 2,
   HK_SYNC_ALL = HK_SYNC_GFX | HK_SYNC_COMPUTE | HK_SYNC_XFER,

   /* Destination scopes only, never a stream's kind: graphics work in the
    * fragment and attachment stages alone. A render's fragment half waits on
    * requirements of both graphics classes, its vertex (tiling) half only on
    * HK_SYNC_GFX ones. Every other graphics destination, and every stream
    * kind, counts as HK_SYNC_GFX. */
   HK_SYNC_GFX_FRAG = 1 << 3,
};

#define HK_SYNC_NUM_KINDS 4

/* Subqueue index: 0 render (VDM), 1 compute (CDM). */
enum hk_sync_sq {
   HK_SYNC_VDM = 0,
   HK_SYNC_CDM = 1,
};

#define HK_SYNC_NUM_SQ 2

/* A requirement on one subqueue: the latest stream recorded in this command
 * buffer that must complete first (NULL: none in this command buffer), and
 * the kinds of streams recorded before this command buffer that must
 * complete first.
 */
struct hk_sync_dep {
   struct hk_cs *cut;
   uint8_t prior;
};

/*
 * Represents a firmware timestamp request.  Handle is a kernel timestamp object
 * handle, not a GEM handle.
 *
 * The kernel/firmware uses the handle/offset_B to write. We use the address to
 * read the results back. We could deduplicate this, but this is convenient.
 */
struct agx_timestamp_req {
   uint64_t addr;
   uint32_t handle;
   uint32_t offset_B;
};

/*
 * Tile hazards. A render pass instance's load, store and resolve operations
 * only affect pixels inside its render area (Vulkan 1.4, "Render Pass Load
 * Operations" / "Render Pass Store Operations"), so two instances that write
 * pixel-disjoint parts of one attachment need no application barrier. The
 * hardware loads and stores whole tiles, however: a tile the render area only
 * partly covers is reloaded from memory and stored back whole. Two instances
 * that share such a tile race when they run concurrently -- the later one can
 * reload the tile before the earlier one stored it and then write the stale
 * pixels back -- and with per-command barriers nothing else orders them.
 *
 * Sparse multiview masks have the same hazard in the layer dimension: BG/EOT
 * reloads and stores every layer below the highest enabled view, preserving
 * inactive layers. A full X/Y area does not make those writes independent.
 * Record every attachment footprint, including complete writes that a later
 * pass may preserve. Order overlapping footprints when either pass preserves
 * pixels or layers. Complete independent writes retain no extra dependency.
 * Footprints use 32x32-pixel cells, the largest tile (agx_select_tile_size); smaller tiles
 * nest inside them, so the test is conservative for every layout.
 *
 * The rule applies within a command buffer at recording, across the command
 * buffers of one submission and against this queue's earlier, possibly still
 * running submissions at submit time. Other queues and other processes are
 * not covered (see the g1fix notes).
 */
#define HK_SYNC_TILE_CELL_LOG2 5
#define HK_SYNC_MAX_PASS_FOOTPRINTS 20
#define HK_SYNC_RECENT_FOOTPRINTS 256

struct hk_sync_footprint {
   /* The hardware preserves pixels or inactive layers outside the API writes. */
   bool preserve;
   /* Identity of the written image; compared, never dereferenced. */
   const void *image;
   uint32_t level;
   /* Layers [layer0, layer1). */
   uint32_t layer0, layer1;
   /* Inclusive 32x32 cell bounds. */
   int32_t cx0, cx1, cy0, cy1;
};

/* A pass footprint and its first and last attachment-producing streams. */
struct hk_sync_pass_record {
   struct hk_sync_footprint fp;
   struct hk_cs *first, *last;
};

/* A footprint of an earlier submission on this queue and the queue timeline
 * point that submission signals. */
struct hk_sync_recent {
   struct hk_sync_footprint fp;
   uint64_t point;
   enum hk_sync_sq sq;
};

/* Per-stream state, embedded in hk_cs. */
struct hk_sync_stream {
   /* Recording order. Equal to list order for every pair the tracking
    * compares: a pre-graphics stream takes the number reserved just below
    * its render so it sorts before it, where the list places it. */
   uint32_t seq;

   /* Set at submission: 1-based position among the commands of this
    * subqueue in the ioctl being built. */
   uint32_t index;

   /* Resolved barrier fields for the ioctl being built. */
   uint16_t barrier[HK_SYNC_NUM_SQ];

   /* Exact historical source prefixes, resolved afresh at each submission. */
   uint64_t prior[HK_SYNC_NUM_SQ];

   uint8_t kind;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];

   /* For a render: what its vertex (tiling) half needs, a subset of dep
    * without the fragment-only requirements (HK_SYNC_GFX_FRAG) and tile
    * hazards. Resolved like dep into barrier_ta/prior_ta; ta_split when
    * they differ from barrier/prior. */
   struct hk_sync_dep dep_ta[HK_SYNC_NUM_SQ];
   uint16_t barrier_ta[HK_SYNC_NUM_SQ];
   uint64_t prior_ta[HK_SYNC_NUM_SQ];
   bool ta_split;
};

/* Per-command-buffer state. */
struct hk_sync_state {
   uint32_t next_seq;

   /* Latest stream of each kind on each subqueue, and the latest stream of
    * any kind. */
   struct hk_cs *last[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   struct hk_cs *last_any[HK_SYNC_NUM_SQ];

   /* What the next stream of each kind must wait on. Monotonic: a barrier's
    * second scope covers every later command. This is also the command
    * buffer's tail requirement, applied to later command buffers of the same
    * submission at queue submit. */
   struct hk_sync_dep req[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];

   /* Dependencies added to every stream created while set. Used for the
    * driver's own passes that continue a render (resolves). */
   struct hk_sync_dep internal[HK_SYNC_NUM_SQ];
   bool internal_active;

   /* Render pass instance bookkeeping. */
   bool in_pass;
   bool resume_pending;
   bool heap_pass;
   struct hk_cs *pass_vdm;

   /* The open general compute stream must be re-validated against the
    * requirements before it receives more work. */
   bool cs_check_pending;

   /* A meta helper selected an exact stream; do not replace it. */
   bool pin_cs;

   /* Kind for streams created now. 0 infers from the recording context. */
   uint8_t kind_override;

   /* Depth of transfer entry points on the stack. */
   uint16_t xfer_depth;

   /* A top-of-pipe timestamp waiting for the next stream to carry it as its
    * start timestamp. */
   struct agx_timestamp_req pending_start_ts;

   /* Tile hazards: attachment footprints of the open pass instance and
    * the latest earlier render stream it shares a tile with. The command
    * buffer keeps the recorded instances (hk_cmd_buffer::sync_pass_records). */
   struct hk_sync_footprint pass_fp[HK_SYNC_MAX_PASS_FOOTPRINTS];
   uint8_t pass_fp_count;
   struct hk_cs *pass_tile_wait[HK_SYNC_NUM_SQ];
   /* Attachment producers, including a compute replacement of a clear. */
   struct hk_cs *pass_tile_first, *pass_tile_last;
};

/* Per-queue state: kinds of streams ever submitted on each subqueue. A
 * dependency on earlier submits of a kind that never ran is no dependency. */
struct hk_sync_queue {
   uint8_t ever[HK_SYNC_NUM_SQ];
   bool exact_prior;
   uint64_t submitted[HK_SYNC_NUM_SQ];
   uint64_t last_absolute[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint64_t tail[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];

   /* Tile hazards across submissions: a ring of the footprints recently
    * submitted on this queue. An entry is live until the queue timeline
    * reaches its point. `evicted` is the latest point of an entry the ring
    * dropped, which keeps the check conservative while that point is live. */
   struct hk_sync_recent recent[HK_SYNC_RECENT_FOOTPRINTS];
   uint32_t recent_next, recent_count;
   uint64_t recent_max, evicted;
};

void hk_sync_reset(struct hk_cmd_buffer *cmd);

/* Stage masks to stream kinds, for a first synchronization scope. */
uint8_t hk_sync_kinds_src(VkPipelineStageFlags2 stages);

/* Union of the scopes of every barrier in a dependency. */
void hk_sync_dependency_kinds(const VkDependencyInfo *info, uint8_t *src,
                              uint8_t *dst);

/* Record an execution dependency: all earlier streams of the source kinds
 * complete before any later stream of the destination kinds starts. */
void hk_sync_barrier(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                     uint8_t dst_kinds);

/* The requirement that "all earlier streams of these kinds complete", for
 * the stream self when it is one the requirement will be applied to (it is
 * then every stream before self), or for a stream still to be created. */
void hk_sync_src_dep(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                     struct hk_cs *self, struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* Every later stream of the given kinds waits on cs. */
void hk_sync_after_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                          uint8_t dst_kinds);

/* Hooks from the control-stream machinery. */
void hk_sync_stream_created(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_cs **slot, bool compute);
void hk_sync_check_cs(struct hk_cmd_buffer *cmd);

/* Stage class of work entering the open general compute stream. Keep this
 * equivalent to stream_kind() for &cmd->current_cs.cs. */
#define hk_sync_general_cs_kind(cmd)                                           \
   ((cmd)->sync.kind_override ? (cmd)->sync.kind_override                       \
    : (cmd)->sync.xfer_depth  ? HK_SYNC_XFER                                    \
                              : HK_SYNC_COMPUTE)
void hk_sync_forget_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_merge(struct hk_cmd_buffer *cmd, struct hk_cs *into,
                   struct hk_cs *from);

/* Obtain the general compute stream, ended and recreated if it is not
 * positioned after dep. The returned stream carries dep. */
struct hk_cs *hk_sync_get_cs_after(struct hk_cmd_buffer *cmd,
                                   const struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* End the general compute stream if it carries an end timestamp, so a query
 * command that follows cannot run before the stamp is taken. */
void hk_sync_end_cs_if_timestamped(struct hk_cmd_buffer *cmd);

/* Timestamps. */
void hk_sync_write_timestamp(struct hk_cmd_buffer *cmd,
                             VkPipelineStageFlags2 stage,
                             struct agx_timestamp_req req);
void hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd);

/* Render pass instance boundaries. */
void hk_sync_begin_rendering(struct hk_cmd_buffer *cmd, bool resuming);
void hk_sync_first_render_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_end_rendering(struct hk_cmd_buffer *cmd);
void hk_sync_heap_used(struct hk_cmd_buffer *cmd);

/* Tile hazards: a written attachment of the pass instance being begun, then
 * the end of the list, before the instance creates its first stream. */
void hk_sync_pass_footprint(struct hk_cmd_buffer *cmd, const void *image,
                            uint32_t level, uint32_t layer0, uint32_t layer1,
                            uint32_t level_width, uint32_t level_height,
                            VkRect2D area, bool preserve_layers);
void hk_sync_pass_footprints_done(struct hk_cmd_buffer *cmd);
void hk_sync_replace_render(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                            struct hk_cs *into);
/* Whether a submission records tile-hazard footprints, and whether the
 * queue may still be running one, so the caller fetches the completed
 * timeline point only when it can matter. */
bool hk_sync_submit_needs_completed(const struct hk_queue *queue,
                                    struct hk_cmd_buffer *const *cmdbufs,
                                    unsigned count);

/* The driver's own resolve pass after a render: waits on the render. */
void hk_sync_resolve_begin(struct hk_cmd_buffer *cmd);
void hk_sync_resolve_end(struct hk_cmd_buffer *cmd);

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class. Defined in hk_cmd_buffer.h. */

/* Submission: resolve every stream's barrier fields for one ioctl payload.
 * cmdbufs are the command buffers in submission order; first_cdm_is_flush
 * says the payload opens with a driver-added compute command that every
 * command must run after; chain_first says the commands must transitively
 * follow the first hardware command (the kernel attaches the ioctl's input
 * syncs only to it).
 */
bool hk_sync_resolve_submit(struct hk_queue *queue,
                            struct hk_cmd_buffer *const *cmdbufs,
                            unsigned count, bool first_cdm_is_flush,
                            bool chain_first, uint64_t completed);
