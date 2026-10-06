/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_pass_profile.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "agx_bo.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "drm-uapi/asahi_drm.h"
#include "util/hash_table.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "util/format/u_format.h"
#include "vk_format.h"

DEBUG_GET_ONCE_OPTION(hk_pass_profile, "HK_PASS_PROFILE", NULL)

#define PROF_SLOTS     16384
#define PROF_WINDOW_NS 2000000000ull

/* Firmware timestamps of one command, in GPU ticks: the start of the
 * command (render vertex start, compute start) and its end (render fragment
 * end, compute end) -- the same pair Honeykrisp uses for timestamp queries. */
struct prof_slot {
   uint64_t start, end;
};

struct prof_record {
   const char *key;
   uint64_t submit_ns;
   uint32_t cmds;
   bool pending;
};

struct prof_entry {
   uint64_t count, cmds, span_ns, marginal_ns;
};

/* A completed command in the current window. */
struct prof_done {
   const char *key;
   uint64_t start, end;
   uint32_t cmds;
};

struct hk_pass_profile {
   simple_mtx_t lock;
   FILE *out;
   struct agx_bo *bo;
   uint32_t handle;
   struct prof_slot *slots;
   struct prof_record records[PROF_SLOTS];
   uint32_t head, tail;
   void *mem;
   struct hash_table *keys;    /* interned key strings */
   struct hash_table *entries; /* key -> prof_entry, current window */
   struct util_dynarray done;
   uint64_t window_start_ns;
   uint64_t dropped, overflowed;
};

void
hk_pass_profile_init(struct hk_device *dev)
{
   const char *path = debug_get_option_hk_pass_profile();
   if (!path)
      return;

   struct hk_pass_profile *p = calloc(1, sizeof(*p));
   if (!p)
      return;

   char name[512];
   snprintf(name, sizeof(name), "%s.%ld", path, (long)getpid());
   p->out = fopen(name, "a");
   p->bo = agx_bo_create(&dev->dev, PROF_SLOTS * sizeof(struct prof_slot), 0,
                         AGX_BO_WRITEBACK | AGX_BO_SHARED, "Pass profile");
   if (!p->out || !p->bo || agx_bind_timestamps(&dev->dev, p->bo, &p->handle)) {
      mesa_loge("HK_PASS_PROFILE: cannot set up %s", name);
      if (p->out)
         fclose(p->out);
      if (p->bo)
         agx_bo_unreference(&dev->dev, p->bo);
      free(p);
      return;
   }

   p->slots = agx_bo_map(p->bo);
   memset(p->slots, 0, PROF_SLOTS * sizeof(struct prof_slot));
   simple_mtx_init(&p->lock, mtx_plain);
   p->mem = ralloc_context(NULL);
   p->keys = _mesa_string_hash_table_create(p->mem);
   p->entries = _mesa_pointer_hash_table_create(p->mem);
   util_dynarray_init(&p->done, p->mem);
   p->window_start_ns = os_time_get_nano();
   fprintf(p->out, "# HK_PASS_PROFILE pid=%ld window_ms=%llu\n", (long)getpid(),
           (unsigned long long)(PROF_WINDOW_NS / 1000000));
   fflush(p->out);
   dev->pass_profile = p;
}

void
hk_pass_profile_fini(struct hk_device *dev)
{
   struct hk_pass_profile *p = dev->pass_profile;
   if (!p)
      return;

   fclose(p->out);
   agx_bo_unreference(&dev->dev, p->bo);
   simple_mtx_destroy(&p->lock);
   ralloc_free(p->mem);
   free(p);
   dev->pass_profile = NULL;
}

static char
load_char(VkAttachmentLoadOp op, bool incomplete)
{
   if (op == VK_ATTACHMENT_LOAD_OP_LOAD || incomplete)
      return 'L';
   if (op == VK_ATTACHMENT_LOAD_OP_CLEAR)
      return 'C';
   if (op == VK_ATTACHMENT_LOAD_OP_NONE)
      return 'N';
   return '-';
}

static char
store_char(VkAttachmentStoreOp op)
{
   if (op == VK_ATTACHMENT_STORE_OP_STORE)
      return 'S';
   if (op == VK_ATTACHMENT_STORE_OP_NONE)
      return 'N';
   return '-';
}

static bool
view_compressed(const struct hk_image_view *iview)
{
   if (!iview)
      return false;
   const struct hk_image *image =
      container_of(iview->vk.image, struct hk_image, vk);
   return image->planes[iview->planes[0].image_plane].layout.compressed;
}

static const char *
short_format(VkFormat format)
{
   return util_format_short_name(vk_format_to_pipe_format(format));
}

void
hk_pass_profile_describe(struct hk_rendering_state *render,
                         const VkRenderingInfo *info, bool incomplete)
{
   char *d = render->cr.prof_desc;
   const size_t size = sizeof(render->cr.prof_desc);
   int n = snprintf(d, size, "%ux%u%s%s", render->cr.width, render->cr.height,
                    render->tilebuffer.nr_samples > 1 ? " msaa" : "",
                    incomplete ? " inc" : "");

   for (unsigned i = 0; i < info->colorAttachmentCount && n < (int)size; ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      if (!att->imageView)
         continue;
      n += snprintf(d + n, size - n, " c%u=%s:%c%c%s%s", i,
                    short_format(render->color_att[i].vk_format),
                    load_char(att->loadOp, incomplete),
                    store_char(att->storeOp),
                    view_compressed(render->color_att[i].iview) ? "z" : "",
                    att->resolveMode ? "R" : "");
   }

   const VkRenderingAttachmentInfo *z = info->pDepthAttachment;
   const VkRenderingAttachmentInfo *s = info->pStencilAttachment;
   if (z && z->imageView && n < (int)size) {
      n += snprintf(d + n, size - n, " z=%s:%c%c%s",
                    short_format(render->depth_att.vk_format),
                    load_char(z->loadOp, incomplete), store_char(z->storeOp),
                    view_compressed(render->depth_att.iview) ? "z" : "");
   }
   if (s && s->imageView && n < (int)size) {
      snprintf(d + n, size - n, " s=:%c%c", load_char(s->loadOp, incomplete),
               store_char(s->storeOp));
   }
}

static const char *
intern(struct hk_pass_profile *p, const char *key)
{
   struct hash_entry *e = _mesa_hash_table_search(p->keys, key);
   if (e)
      return e->key;
   char *copy = ralloc_strdup(p->mem, key);
   _mesa_hash_table_insert(p->keys, copy, copy);
   return copy;
}

static int
alloc_slot(struct hk_pass_profile *p, const char *key, uint32_t cmds)
{
   if (p->head - p->tail >= PROF_SLOTS) {
      p->overflowed++;
      return -1;
   }
   const unsigned i = p->head++ % PROF_SLOTS;
   memset(&p->slots[i], 0, sizeof(p->slots[i]));
   p->records[i] = (struct prof_record){
      .key = intern(p, key),
      .submit_ns = os_time_get_nano(),
      .cmds = cmds,
      .pending = true,
   };
   return i;
}

void
hk_pass_profile_render(struct hk_device *dev, const struct hk_cs *cs,
                       struct drm_asahi_cmd_render *c)
{
   struct hk_pass_profile *p = dev->pass_profile;
   if (!p || c->ts_vtx.start.handle || c->ts_vtx.end.handle ||
       c->ts_frag.start.handle || c->ts_frag.end.handle)
      return;

   struct agx_zls_control_packed zls_packed = cs->cr.zls_control;
   agx_unpack(NULL, &zls_packed, ZLS_CONTROL, zls);

   /* A stream that runs the partial background program reloads everything
    * an earlier stream of the same pass stored; one that runs the partial
    * end-of-tile program stores everything for a later stream. */
   const bool first = cs->cr.bg.main.usc != cs->cr.bg.partial.usc;
   const bool last = cs->cr.eot.main.usc != cs->cr.eot.partial.usc;
   const char *pos = first ? (last ? "only" : "first") : (last ? "last" : "mid");

   char key[256];
   snprintf(key, sizeof(key), "VDM %s | %s | zls=%s%s%s%s%s%s",
            cs->cr.prof_desc[0] ? cs->cr.prof_desc : "(meta)", pos,
            zls.z_load ? "zL" : "", zls.z_store ? "zS" : "",
            zls.s_load ? "sL" : "", zls.s_store ? "sS" : "",
            zls.z_load_compress || zls.z_store_compress ? "+zc" : "",
            cs->cr.depth.buffer || cs->cr.stencil.buffer ? "" : "(none)");

   simple_mtx_lock(&p->lock);
   int i = alloc_slot(p, key, cs->stats.cmds);
   simple_mtx_unlock(&p->lock);
   if (i < 0)
      return;

   const uint32_t base = i * sizeof(struct prof_slot);
   c->ts_vtx.start = (struct drm_asahi_timestamp){p->handle, base};
   c->ts_frag.end = (struct drm_asahi_timestamp){p->handle, base + 8};
}

void
hk_pass_profile_note_dispatch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              int code)
{
   if (!cs)
      return;

   /* Inside a meta operation, the function that began it; outside, the
    * dispatches of draws and of the application itself. */
   const char *site = cmd->prof_site ? cmd->prof_site : "cmd";
   struct hk_prof_origin *o = cs->prof_origin;
   const unsigned n = ARRAY_SIZE(cs->prof_origin);
   unsigned i;
   for (i = 0; i < n && o[i].count; ++i) {
      if (o[i].site == site && o[i].code == code)
         break;
   }
   if (i == n)
      return;
   if (!o[i].count)
      o[i] = (struct hk_prof_origin){.site = site, .code = code};
   if (o[i].count < UINT16_MAX)
      o[i].count++;

   /* Keep the most frequent first. */
   while (i > 0 && o[i].count > o[i - 1].count) {
      struct hk_prof_origin t = o[i - 1];
      o[i - 1] = o[i];
      o[i] = t;
      --i;
   }
}

void
hk_pass_profile_compute(struct hk_device *dev, const struct hk_cs *cs,
                        struct drm_asahi_cmd_compute *c)
{
   struct hk_pass_profile *p = dev->pass_profile;
   if (!p || c->ts.start.handle || c->ts.end.handle)
      return;

   /* origins: site/pc<libagx program> or site/<shader stage>, most frequent
    * first. Program numbers index enum libagx_program (libagx_shaders.h). */
   char key[256];
   const unsigned n = cs->stats.cmds;
   int len = snprintf(key, sizeof(key), "CDM dispatches=%s flushes=%s |",
                      n <= 1 ? "<=1" : n <= 4 ? "2-4" : n <= 16 ? "5-16" : ">16",
                      cs->stats.flushes ? "y" : "n");
   for (unsigned i = 0; i < ARRAY_SIZE(cs->prof_origin) &&
                        cs->prof_origin[i].count && len < (int)sizeof(key); ++i) {
      const struct hk_prof_origin *o = &cs->prof_origin[i];
      if (o->code >= HK_PROF_STAGE)
         len += snprintf(key + len, sizeof(key) - len, " %s/%s", o->site,
                         _mesa_shader_stage_to_abbrev(o->code - HK_PROF_STAGE));
      else if (o->code >= 0)
         len += snprintf(key + len, sizeof(key) - len, " %s/pc%d", o->site,
                         o->code);
      else
         len += snprintf(key + len, sizeof(key) - len, " %s/?", o->site);
   }

   simple_mtx_lock(&p->lock);
   int i = alloc_slot(p, key, n);
   simple_mtx_unlock(&p->lock);
   if (i < 0)
      return;

   const uint32_t base = i * sizeof(struct prof_slot);
   c->ts.start = (struct drm_asahi_timestamp){p->handle, base};
   c->ts.end = (struct drm_asahi_timestamp){p->handle, base + 8};
}

static int
cmp_done_end(const void *a, const void *b)
{
   const struct prof_done *x = a, *y = b;
   return x->end < y->end ? -1 : x->end > y->end;
}

struct sort_item {
   const char *key;
   struct prof_entry *e;
};

static int
cmp_item(const void *a, const void *b)
{
   const struct sort_item *x = a, *y = b;
   return x->e->marginal_ns < y->e->marginal_ns   ? 1
          : x->e->marginal_ns > y->e->marginal_ns ? -1
                                                  : 0;
}

/* Commands overlap (the vertex work of one render runs under the fragment
 * work of the previous one), so summing their spans overcounts. In end
 * order, each command is charged only the part of its span after
 * everything that ended before it: the marginal busy time it adds. The
 * marginal times sum to the GPU busy time of the window. */
static void
flush_window(struct hk_device *dev, struct hk_pass_profile *p, uint64_t now)
{
   const double window_s = (now - p->window_start_ns) / 1e9;
   unsigned nd = util_dynarray_num_elements(&p->done, struct prof_done);
   struct prof_done *d = p->done.data;
   if (nd)
      qsort(d, nd, sizeof(*d), cmp_done_end);

   uint64_t frontier = 0, busy = 0;
   for (unsigned i = 0; i < nd; ++i) {
      struct hash_entry *he = _mesa_hash_table_search(p->entries, d[i].key);
      struct prof_entry *e;
      if (he) {
         e = he->data;
      } else {
         e = calloc(1, sizeof(*e));
         _mesa_hash_table_insert(p->entries, d[i].key, e);
      }
      const uint64_t from = MAX2(d[i].start, frontier);
      const uint64_t marginal =
         agx_gpu_timestamp_to_ns(&dev->dev, d[i].end > from ? d[i].end - from : 0);
      e->count++;
      e->cmds += d[i].cmds;
      e->span_ns += agx_gpu_timestamp_to_ns(&dev->dev, d[i].end - d[i].start);
      e->marginal_ns += marginal;
      busy += marginal;
      frontier = MAX2(frontier, d[i].end);
   }

   unsigned n = _mesa_hash_table_num_entries(p->entries);
   struct sort_item *items = calloc(n ? n : 1, sizeof(*items));
   unsigned k = 0;
   hash_table_foreach(p->entries, he)
      items[k++] = (struct sort_item){he->key, he->data};
   qsort(items, k, sizeof(*items), cmp_item);

   fprintf(p->out,
           "== window t=%.3f s=%.3f cmds=%u gpu_busy_ms=%.1f "
           "dropped=%" PRIu64 " overflowed=%" PRIu64 "\n",
           p->window_start_ns / 1e9, window_s, nd, busy / 1e6, p->dropped,
           p->overflowed);
   for (unsigned i = 0; i < k; ++i) {
      const struct prof_entry *e = items[i].e;
      fprintf(p->out,
              "%9.2f ms busy %9.2f ms span %6" PRIu64 " x %7" PRIu64
              " draws  %s\n",
              e->marginal_ns / 1e6, e->span_ns / 1e6, e->count, e->cmds,
              items[i].key);
   }
   fflush(p->out);
   free(items);

   hash_table_foreach(p->entries, he)
      free(he->data);
   _mesa_hash_table_clear(p->entries, NULL);
   util_dynarray_clear(&p->done);
   p->window_start_ns = now;
}

void
hk_pass_profile_harvest(struct hk_device *dev)
{
   struct hk_pass_profile *p = dev->pass_profile;
   if (!p)
      return;

   const uint64_t now = os_time_get_nano();
   simple_mtx_lock(&p->lock);

   /* Commands may complete out of order across queues: take every completed
    * record, then advance the tail past the completed prefix. A record still
    * pending after five seconds is dropped. */
   for (uint32_t s = p->tail; s != p->head; ++s) {
      const unsigned i = s % PROF_SLOTS;
      struct prof_record *r = &p->records[i];
      if (!r->pending)
         continue;
      volatile struct prof_slot *t = &p->slots[i];
      const uint64_t end = t->end, start = t->start;
      if (!end || !start) {
         if (now - r->submit_ns > 5000000000ull) {
            r->pending = false;
            p->dropped++;
         }
         continue;
      }
      r->pending = false;
      if (end < start) {
         p->dropped++;
         continue;
      }
      struct prof_done done = {r->key, start, end, r->cmds};
      util_dynarray_append(&p->done, done);
   }

   while (p->tail != p->head && !p->records[p->tail % PROF_SLOTS].pending)
      p->tail++;

   if (now - p->window_start_ns >= PROF_WINDOW_NS)
      flush_window(dev, p, now);

   simple_mtx_unlock(&p->lock);
}
