/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/u_queue.h"

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"

/* v56f/v57d: environment knobs read on command-recording paths are read once
 * per call site. getenv() is a linear scan of the environment block and
 * showed up on Minecraft's render thread (getenv 0.93%, strncmp 0.50%). */
#define HK_ENV_BOOL_ONCE(name, def)                                        \
   ({                                                                      \
      static int hk_env_once_ = -1;                                        \
      if (unlikely(hk_env_once_ < 0))                                      \
         hk_env_once_ = debug_get_bool_option(name, def);                  \
      (bool)hk_env_once_;                                                  \
   })
#define HK_ENV_NUM_ONCE(name, def)                                         \
   ({                                                                      \
      static int64_t hk_env_num_once_;                                     \
      static int hk_env_num_init_;                                         \
      if (unlikely(!p_atomic_read(&hk_env_num_init_))) {                   \
         hk_env_num_once_ = debug_get_num_option(name, def);               \
         p_atomic_set(&hk_env_num_init_, 1);                               \
      }                                                                    \
      hk_env_num_once_;                                                    \
   })
#define HK_ENV_PRESENT_ONCE(name)                                          \
   ({                                                                      \
      static int hk_env_present_ = -1;                                     \
      if (unlikely(hk_env_present_ < 0))                                   \
         hk_env_present_ = getenv(name) != NULL;                           \
      (bool)hk_env_present_;                                               \
   })
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_native_sampler.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_acceleration_structure.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct hk_shader_binary_cache_entry;
struct vk_pipeline_cache;

#define HK_SHADER_BINARY_CACHE_MAX 16

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;

   /* A compilation input that changes the emitted binary but is not part of
    * the caller's key bytes, so it has to be hashed alongside them: `stop`
    * decides whether the part ends the shader or falls through to a footer
    * the fast linker appends.
    *
    * internal_key_hash() hashes sizeof(struct hk_internal_key) + key_size
    * bytes, which is wider than builder + key_size + the key data once a
    * member sits between key_size and key[]. Every producer of this struct
    * therefore zeroes the whole allocation, not just the header, so that no
    * padding byte reaches the hash uninitialized.
    */
   bool stop;

   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   struct agxdecode_ctx *decode_ctx;

   struct hk_descriptor_table occlusion_queries;
   struct hk_sampler_heap samplers;

   /* Native sampled-image indices are stable for an image-view lifetime.
    * G17P shaders dereference this table on the GPU, including descriptors
    * updated after command recording. Other generations keep inline state.
    */
   struct hk_descriptor_table sampled_images;

   struct vk_meta_device meta;

   /* VK_KHR_acceleration_structure.  radix_sort is created on first use
    * behind radix_sort_lock, because building it compiles a dozen compute
    * pipelines that most devices never need. */
   struct vk_acceleration_structure_build_args accel_struct_args;
   struct radix_sort_vk *radix_sort;
   bool radix_sort_failed;
   simple_mtx_t radix_sort_lock;
   struct agx_bg_eot_cache bg_eot;
   /* frg-codegen v4 BG/EOT device cache (hk_cmd_draw.c). */
   struct hk_g17p_bg_eot_dev_cache *g17p_bg_eot_dev_cache;

   /* VK_KHR_ray_tracing_pipeline shader group handles.  A handle is one
    * device-unique index, so a handle captured from a pipeline library keeps
    * naming the same group in every pipeline that imports the library.  It
    * starts at one: zero is the value an unwritten shader binding table
    * record has, and no group may collide with it. */
   uint32_t rt_group_handle_alloc;

   /* Device-lifetime templates for precompiled G17P meta compute programs.
    * The libagx programs and their executable BOs have the same lifetime. */
   struct {
      void *templates[LIBAGX_NUM_PROGRAMS];
   } g17p_compute_program_cache;

   /* v48: kernel-maintained copy of this VM's sticky status
    * (DRM_ASAHI_FEATURE_VM_STATUS_MIRROR). NULL: query per check instead. */
   struct {
      struct agx_bo *bo;
      int32_t *word;
   } vm_status_mirror;

   /* Immutable diagnostic selection. Never rescan the process environment
    * from render-pass, draw, dispatch, or per-program publication paths. */
   struct {
      bool trace;
      bool aggregate;
      bool fault_trace;
      bool prefault_stack;
      bool prefault_driver;
      bool prefault_main;
      bool prefault_ioto;
      bool prefault_streams;
   } g17p_shader_perf;

   struct {
      bool launch_trace;
   } g17p_diagnostics;

   struct {
      /* Immutable for the device's lifetime: the native texel-fetch sampler
       * table. Its address is published to the GPU as a sampler table base by
       * every G17P background reload program, so nothing that any subsystem
       * writes at draw time may share this buffer object. */
      struct agx_bo *bo;

      /* G17P texture and sampler heaps are allocated at full capacity during
       * device creation, so their VAs never move.  Keep the pair in immutable
       * device-owned storage: uploading it per draw makes its address part of
       * every indexed-texture Ioto key and defeats program reuse. */
      struct agx_bo *indexed_heap_pointers_bo;
      uint64_t indexed_heap_pointers;

      /* The geometry/tessellation heap descriptor. hk_init_heap() writes it
       * from the CPU on the first tessellated or geometry-shaded draw, and the
       * libagx bump allocator then atomically updates poly_heap::bottom from
       * the GPU on every such draw. Kept in its own buffer object so that
       * mutation cannot reach the sampler table above. */
      struct agx_bo *heap_bo;
      uint64_t heap;
   } rodata;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* A small device-owned cache of authenticated immutable shader payloads.
    * Every API create still gets a distinct vk_shader wrapper and distinct
    * mutable link caches. */
   struct {
      simple_mtx_t lock;
      struct hk_shader_binary_cache_entry
         *entries[HK_SHADER_BINARY_CACHE_MAX];
   } shader_binary_cache;

   /* v56e HK_G17P_LAZY_GENERIC=2: background compiles of generic variants. */
   struct {
      simple_mtx_t lock;
      bool initialized;
      struct util_queue queue;
   } g17p_lazy;

   /* Device-wide compact VDM backing. Command buffers from different pools
    * suballocate immutable spans from the same finite G17P aperture. */
   struct {
      simple_mtx_t lock;
      struct list_head slabs;
   } vdm;

   /* Native G17P indirect compute grid helper.  A compact indirect helper
    * record carries no dimensions and no resource table, so the only way a
    * runtime grid can reach the main record is for this program to publish
    * the six-u32 geometry block.  Compiled once, from NIR, through the same
    * path as any other Honeykrisp compute shader, so it uses the driver's own
    * proven root/uniform ABI rather than a replayed one.  NULL on every other
    * chip, and NULL on G17P if the backend refused it -- in which case
    * indirect dispatch stays refused instead of launching a program that does
    * not exist. */
   struct hk_api_shader *g17p_grid_helper;

   /* The same contract for an indirect-local grid, which an earlier kernel
    * already wrote as a launch geometry: this helper copies it. */
   struct hk_api_shader *g17p_geometry_copy_helper;

   /* The same contract again, for an indirect-global grid whose arguments
    * live in a sparse residency buffer: identical except that it consults the
    * buffer's residency bitmap and publishes a zero grid for arguments in a
    * non-resident page.  Kept as its own program so the ordinary indirect
    * dispatch path records exactly the code it recorded before. */
   struct hk_api_shader *g17p_grid_helper_sparse;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

   struct {
      struct u_rwlock lock;
      struct util_dynarray list;
      struct util_dynarray counts;
   } external_bos;
};

void hk_g17p_compute_program_cache_finish(struct hk_device *dev);
bool hk_g17p_compute_program_cache_prewarm(struct hk_device *dev,
                                           enum libagx_program program,
                                           size_t data_size);

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_NOBATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

/* Force the geometry/tessellation heap to exist now rather than inside the
 * first draw that needs it.  Idempotent and safe to call from any thread. */
void hk_device_ensure_geometry_heap(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             const struct hk_native_sampler_descriptor *desc,
                             struct hk_rc_sampler **out);

void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}

/* frg-codegen v4: device-lifetime cache of single-attachment G17P BG/EOT
 * launch programs (hk_cmd_draw.c). */
void hk_g17p_bg_eot_dev_cache_init(struct hk_device *dev);
void hk_g17p_bg_eot_dev_cache_finish(struct hk_device *dev);
