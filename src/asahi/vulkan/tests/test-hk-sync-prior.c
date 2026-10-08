/* SPDX-License-Identifier: MIT
 * Calls real hk_sync and production split-rebase helper; no GPU or DRM calls.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "hk_cmd_buffer.h"
#include "hk_drm_prior_core.h"
#include "hk_queue.h"
_Static_assert(sizeof(struct drm_asahi_cmd_render) == 256, "legacy render");
_Static_assert(sizeof(struct drm_asahi_cmd_compute) == 64,
               "legacy Mesa compute");
_Static_assert(sizeof(struct drm_asahi_cmd_render_with_prior) == 272,
               "render wire");
_Static_assert(offsetof(struct drm_asahi_cmd_render_with_prior, prior_vdm) ==
                  256,
               "render prior vdm");
_Static_assert(offsetof(struct drm_asahi_cmd_render_with_prior, prior_cdm) ==
                  264,
               "render prior cdm");
_Static_assert(sizeof(struct drm_asahi_cmd_compute_with_prior) == 88,
               "compute wire");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, scs_layout) ==
                  64,
               "kernel scs_layout gap");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, prior_vdm) ==
                  72,
               "compute prior vdm");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, prior_cdm) ==
                  80,
               "compute prior cdm");
static unsigned tests, failures;
static void *test_context;

/* Fixtures need only the recorded synchronization fields, not a Vulkan device.
 * Keep all fixture allocations under one owner, including the overflow case. */
static void *
test_alloc(size_t size)
{
   void *ptr = rzalloc_size(test_context, size);
   if (!ptr) {
      fputs("Failed to allocate synchronization test fixture\n", stderr);
      exit(EXIT_FAILURE);
   }
   return ptr;
}
#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)
static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = test_alloc(sizeof(*q));
   q->sync.exact_prior = exact;
   return q;
}
static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = test_alloc(sizeof(*c));
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   return c;
}
static struct hk_cs *
add(struct hk_cmd_buffer *c, bool compute)
{
   struct hk_cs *s = test_alloc(sizeof(*s));
   s->type = compute ? HK_CS_CDM : HK_CS_VDM;
   list_addtail(&s->node, &c->control_streams);
   hk_sync_stream_created(c, s, &s, compute);
   return s;
}
static bool
submit(struct hk_queue *q, struct hk_cmd_buffer *c)
{
   struct hk_cmd_buffer *cs[] = {c};
   return hk_sync_resolve_submit(q, cs, 1, false, false, 0);
}
/* The fixture has no open compute stream. A request to close one is a bug. */
void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

/* Record a real full-frame pass with a sparse-mask preservation footprint.
 * Layer ranges describe BG/EOT's physical access, not just enabled views. */
static struct hk_cs *
render_pass(struct hk_cmd_buffer *cmd, const void *image, uint32_t layers,
            bool preserve_layers)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, 0, layers, 32, 32,
                         (VkRect2D){{0, 0}, {32, 32}}, preserve_layers);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   cs->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, cs, &cmd->current_cs.gfx, false);
   list_addtail(&cs->node, &cmd->control_streams);
   hk_sync_end_rendering(cmd);
   return cs;
}

static void
test_multiview_layer_preservation(void)
{
   const void *image = (void *)0x1234;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *first = render_pass(cmd, image, 1, false);
   struct hk_cs *sparse = render_pass(cmd, image, 2, true);
   T("sparse layer store waits on full XY layer0 clear",
     sparse->sync.dep[HK_SYNC_VDM].cut == first);
   struct hk_queue *queue = new_queue(true);
   T("multiview same command buffer submit", submit(queue, cmd));
   T("layer preservation resolves to exact earlier render",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index);

   cmd = new_cmd();
   first = render_pass(cmd, image, 2, false);
   struct hk_cs *whole = render_pass(cmd, image, 2, false);
   T("complete layer writes do not add a tile dependency",
     !whole->sync.dep[HK_SYNC_VDM].cut);
   struct hk_cs *different = render_pass(cmd, (void *)0x5678, 2, true);
   T("different attachment does not add a layer dependency",
     !different->sync.dep[HK_SYNC_VDM].cut);

   queue = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   first = render_pass(a, image, 1, false);
   sparse = render_pass(b, image, 2, true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("multiview across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("across command buffers keeps layer store ordered",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index);

   queue = new_queue(true);
   a = new_cmd(); b = new_cmd();
   render_pass(a, image, 1, false);
   sparse = render_pass(b, image, 2, true);
   queue->drm.timeline_value = 1;
   T("multiview first queue submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("multiview later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0));
   T("across queue submissions waits on prior layer store",
     sparse->sync.barrier[HK_SYNC_VDM] == 0);
}

/* The empty-render optimizer executes the API clear on the compute engine. */
static struct hk_cs *
compute_clear_pass(struct hk_cmd_buffer *cmd, const void *image,
                   uint32_t base_layer, bool end)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, base_layer, base_layer + 1, 32, 32,
                         (VkRect2D){{0, 0}, {32, 32}}, false);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *render = test_alloc(sizeof(*render));
   render->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, render, &cmd->current_cs.gfx, false);
   list_addtail(&render->node, &cmd->control_streams);
   struct hk_cs *clear = test_alloc(sizeof(*clear));
   clear->type = HK_CS_CDM;
   clear->stats.cmds = 1;
   cmd->sync.kind_override = HK_SYNC_GFX;
   hk_sync_stream_created(cmd, clear, &cmd->current_cs.cs, true);
   list_addtail(&clear->node, &cmd->control_streams);
   cmd->sync.kind_override = 0;
   hk_sync_replace_render(cmd, render, clear);
   hk_sync_forget_stream(cmd, render);
   list_del(&render->node);
   if (end)
      hk_sync_end_rendering(cmd);
   return clear;
}

static void
test_compute_clear_layer_preservation(void)
{
   const void *image = (void *)0x1234;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *clear = compute_clear_pass(cmd, image, 0, true);
   struct hk_cs *sparse = render_pass(cmd, image, 2, true);
   T("sparse layer store waits on compute replacement clear",
     sparse->sync.dep[HK_SYNC_CDM].cut == clear);
   T("compute replacement does not fabricate render dependency",
     sparse->sync.dep[HK_SYNC_VDM].cut == NULL);
   struct hk_queue *queue = new_queue(true);
   T("compute clear same command buffer submit", submit(queue, cmd));
   T("compute clear resolves to correct engine prefix",
     sparse->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   queue = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   clear = compute_clear_pass(a, image, 0, true);
   sparse = render_pass(b, image, 2, true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("compute clear across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("across command buffers names compute not render engine",
     sparse->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   queue = new_queue(true);
   a = new_cmd(); b = new_cmd();
   compute_clear_pass(a, image, 0, true);
   sparse = render_pass(b, image, 2, true);
   queue->drm.timeline_value = 1;
   T("compute clear first queue submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("compute clear later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0));
   T("prior compute clear hazard names only compute history",
     sparse->sync.barrier[HK_SYNC_CDM] == 0 &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
   T("completed prior clear submission", hk_sync_resolve_submit(
      queue, single, 1, false, false, 1));
   T("completed prior clear has no remaining hazard",
     sparse->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);

   /* Separate API layers are written on separate engines; preserving both
    * needs both producers, not just whichever has the larger sequence. */
   cmd = new_cmd();
   struct hk_cs *graphics = render_pass(cmd, image, 1, false);
   clear = compute_clear_pass(cmd, image, 1, true);
   sparse = render_pass(cmd, image, 3, true);
   T("sparse third layer waits on both physical producers",
     sparse->sync.dep[HK_SYNC_VDM].cut == graphics &&
     sparse->sync.dep[HK_SYNC_CDM].cut == clear);

   cmd = new_cmd();
   clear = compute_clear_pass(cmd, image, 0, false);
   struct hk_cs *continued = test_alloc(sizeof(*continued));
   continued->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, continued, &cmd->current_cs.gfx, false);
   list_addtail(&continued->node, &cmd->control_streams);
   T("render continuation waits on replaced clear",
     continued->sync.dep[HK_SYNC_CDM].cut == clear);
   hk_sync_end_rendering(cmd);
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   if (!test_context)
      return EXIT_FAILURE;
   test_multiview_layer_preservation();
   test_compute_clear_layer_preservation();
   /* Each matching destination retains the exact old source prefix, including
    * reuse of a recording and independent physical destinations on the queue. */
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *d = add(b, false);
   T("first submit", submit(q, a));
   T("later graphics submit", submit(q, b));
   T("exact old compute prefix",
     d->sync.prior[1] == 1 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *i = add(c, true);
   T("unrelated compute submit", submit(q, c));
   T("unrelated compute no graphics dependency",
     i->sync.prior[0] == 0 && i->sync.barrier[0] == DRM_ASAHI_BARRIER_NONE);
   T("repeat recorded graphics", submit(q, b));
   T("repeat still names original1 not newer2",
     d->sync.prior[1] == 1 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   c = new_cmd();
   add(c, true);
   hk_sync_barrier(c, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("new tail submit", submit(q, c));
   T("repeat after new barrier", submit(q, b));
   T("new tail names compute3", d->sync.prior[1] == 3);
   struct hk_queue *isolated = new_queue(true);
   T("other queue same CB", submit(isolated, b));
   T("queue isolation no inherited dependency",
     d->sync.prior[1] == 0 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   T("original queue same CB again", submit(q, b));
   T("no baked queue-specific prior in recording", d->sync.prior[1] == 3);
   q = new_queue(true);
   a = new_cmd();
   b = new_cmd();
   c = new_cmd();
   add(a, false);
   hk_sync_barrier(b, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   d = add(c, true);
   T("barrier-only source", submit(q, a));
   T("barrier-only submit", submit(q, b));
   T("barrier-only destination", submit(q, c));
   T("barrier-only prior render1",
     d->sync.prior[0] == 1 && d->sync.barrier[0] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   a = new_cmd();
   b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   d = add(b, false);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("same-submit", hk_sync_resolve_submit(q, pair, 2, false, false, 0));
   T("same-submit keeps relative1",
     d->sync.barrier[1] == 1 && d->sync.prior[1] == 0);
   q = new_queue(true);
   a = new_cmd();
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("empty source barrier", submit(q, a));
   b = new_cmd();
   d = add(b, false);
   T("empty source destination", submit(q, b));
   T("empty source no future dependency",
     d->sync.prior[1] == 0 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   T("synthetic SSTT only", hk_sync_resolve_submit(q, NULL, 0, true, false, 0));
   T("SSTT increments actual CDM ordinal",
     q->sync.submitted[1] == 1 && q->sync.last_absolute[1][1] == 1);
   a = new_cmd();
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("tail after SSTT", submit(q, a));
   b = new_cmd();
   d = add(b, false);
   T("destination after SSTT tail", submit(q, b));
   T("SSTT historical dependency1", d->sync.prior[1] == 1);
   q = new_queue(false);
   a = new_cmd();
   b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   d = add(b, false);
   T("legacy source", submit(q, a));
   T("legacy destination", submit(q, b));
   T("legacy unchanged no new history",
     d->sync.prior[1] == 0 && q->sync.submitted[1] == 0 &&
        d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   q->sync.submitted[1] = UINT64_MAX;
   a = new_cmd();
   d = add(a, true);
   d->sync.index = 73;
   T("ordinal overflow rejected", !submit(q, a));
   T("overflow no counter/stream mutation",
     q->sync.submitted[1] == UINT64_MAX && d->sync.index == 73);
   struct drm_asahi_cmd_render_with_prior r = {0};
   struct drm_asahi_cmd_header h = {.cmd_type = DRM_ASAHI_CMD_RENDER,
                                    .size = sizeof(r),
                                    .vdm_barrier = 2,
                                    .cdm_barrier = 5};
   uint64_t base[] = {10, 20};
   T("split rebase", hk_drm_rebase_prior(&h, &r, 3, 4, base, true));
   T("prior slice gets exact global12",
     h.vdm_barrier == DRM_ASAHI_BARRIER_NONE && r.prior_vdm == 12);
   T("same slice keeps relative1", h.cdm_barrier == 1 && r.prior_cdm == 0);
   h.vdm_barrier = 0;
   h.cdm_barrier = DRM_ASAHI_BARRIER_NONE;
   T("deliberate zero rebase", hk_drm_rebase_prior(&h, &r, 7, 8, base, true));
   T("deliberate zero preserved", h.vdm_barrier == 0 && r.prior_vdm == 12);
   struct drm_asahi_cmd_compute_with_prior co = {0};
   h = (struct drm_asahi_cmd_header){.cmd_type = DRM_ASAHI_CMD_COMPUTE,
                                     .size = sizeof(co),
                                     .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
                                     .cdm_barrier = 1};
   T("compute split layout", hk_drm_rebase_prior(&h, &co, 0, 1, base, true));
   T("compute prior21 scs remains zero",
     co.prior_cdm == 21 && co.scs_layout == 0 &&
        h.cdm_barrier == DRM_ASAHI_BARRIER_NONE);
   h.cdm_barrier = 1;
   base[1] = UINT64_MAX;
   struct drm_asahi_cmd_header before = h;
   T("split overflow rejected",
     !hk_drm_rebase_prior(&h, &co, 0, 1, base, true));
   T("split overflow no header mutation", !memcmp(&h, &before, sizeof(h)));
   h.size = 64;
   h.cdm_barrier = 1;
   base[1] = 20;
   T("legacy split", hk_drm_rebase_prior(&h, &co, 0, 1, base, false));
   T("legacy split old zero", h.cdm_barrier == 0);

   /* Sparse maintenance contributes exactly one hardware ordinal before app
    * streams. */
   q = new_queue(true);
   a = new_cmd();
   d = add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cmd_buffer *single[] = {a};
   T("SSTT plus application",
     hk_sync_resolve_submit(q, single, 1, true, false, 0));
   T("SSTT plus app counters", q->sync.submitted[1] == 2 &&
                                  d->sync.index == 2 &&
                                  d->sync.barrier[1] == 1);
   b = new_cmd();
   d = add(b, false);
   T("SSTT app later destination", submit(q, b));
   T("SSTT app exact prefix2", d->sync.prior[1] == 2);
   /* A later barrier-only buffer with both source kinds keeps distinct engine
    * ordinals. */
   q = new_queue(true);
   a = new_cmd();
   add(a, true);
   add(a, false);
   T("both engine sources", submit(q, a));
   b = new_cmd();
   hk_sync_barrier(b, HK_SYNC_ALL, HK_SYNC_ALL);
   T("all-kind standalone barrier", submit(q, b));
   c = new_cmd();
   d = add(c, true);
   T("all-kind destination", submit(q, c));
   T("VDM CDM order",
     d->sync.prior[HK_SYNC_VDM] == 1 && d->sync.prior[HK_SYNC_CDM] == 1);
   /* Real resolver preflight rejects relative representation overflow before
    * mutation. */
   q = new_queue(true);
   a = new_cmd();
   for (unsigned n = 0; n < 65535; n++)
      add(a, true);
   d = list_first_entry(&a->control_streams, struct hk_cs, node);
   d->sync.index = 91;
   T("relative overflow rejected", !submit(q, a));
   T("relative overflow untouched",
     q->sync.submitted[1] == 0 && d->sync.index == 91);
   /* Legacy packet prefixes retain all original bytes, extensions are omitted.
    */
   struct drm_asahi_cmd_render lr;
   memset(&lr, 0xa5, sizeof(lr));
   struct drm_asahi_cmd_render_with_prior er = {.legacy = lr,
                                                .prior_vdm = 17,
                                                .prior_cdm = 29};
   T("legacy render bytes", memcmp(&lr, &er, sizeof(lr)) == 0);
   struct drm_asahi_cmd_compute lc;
   memset(&lc, 0x5a, sizeof(lc));
   struct drm_asahi_cmd_compute_with_prior ec = {.legacy = lc,
                                                 .prior_vdm = 17,
                                                 .prior_cdm = 29};
   T("legacy compute bytes", memcmp(&lc, &ec, sizeof(lc)) == 0);
   printf("RESULT checks=%u failures=%u\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
