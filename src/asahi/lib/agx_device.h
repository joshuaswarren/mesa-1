/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#endif
#include "drm-uapi/asahi_drm.h"
#include "util/ralloc.h"
#include "util/cnd_monotonic.h"
#include "util/u_thread.h"
#include "util/simple_mtx.h"
#include "util/sparse_array.h"
#include "util/timespec.h"
#include "util/u_printf.h"
#include "util/vma.h"

/* Internal-only submit marker consumed by the D3DKMT winsys.  It is not a
 * drm_asahi UAPI flag: native frontends set it only after serialising a
 * complete per-pipe GPU write set into SET_*_ATTACHMENTS records. */
#define AGX_SUBMIT_WRITE_SET_COMPLETE (1u << 31)
#include "agx_bo.h"
#include "agx_chip.h"
#include "agx_pack.h"
#include "agx_submit_virt.h"
#include "decode.h"
#include "layout.h"
#include "libagx_dgc.h"

/*
 * honeykrisp-h1-execution.md SS2.1 row D2: vdrm.h and asahi_proto.h are not
 * self-contained without libdrm (asahi_proto.h embeds struct vdrm_shmem,
 * pulled from vdrm.h's own libdrm-using guts) and neither is needed by this
 * header itself -- only by pointer (struct vdrm_device *dev->vdrm,
 * struct asahi_ccmd_submit_res *agx_submit_virt::extres below). On Windows
 * (no HAVE_LIBDRM) forward-declare instead; the real includes move into the
 * virtio-only TUs that need the full definitions (agx_device_virtio.c,
 * agx_device.c under HAVE_LIBDRM).
 */
#if defined(HAVE_LIBDRM) && !defined(AGX_NO_VIRTIO)
#include "vdrm.h"
#include "asahi_proto.h"
#else
struct vdrm_device;
struct asahi_ccmd_submit_res;
#endif

enum agx_dbg {
   AGX_DBG_TRACE = BITFIELD_BIT(0),
   AGX_DBG_BODUMP = BITFIELD_BIT(1),
   AGX_DBG_NO16 = BITFIELD_BIT(2),
   AGX_DBG_DIRTY = BITFIELD_BIT(3),
   AGX_DBG_PRECOMPILE = BITFIELD_BIT(4),
   AGX_DBG_PERF = BITFIELD_BIT(5),
   AGX_DBG_NOCOMPRESS = BITFIELD_BIT(6),
   AGX_DBG_NOCLUSTER = BITFIELD_BIT(7),
   AGX_DBG_SYNC = BITFIELD_BIT(8),
   AGX_DBG_STATS = BITFIELD_BIT(9),
   AGX_DBG_RESOURCE = BITFIELD_BIT(10),
   AGX_DBG_BATCH = BITFIELD_BIT(11),
   AGX_DBG_NOWC = BITFIELD_BIT(12),
   AGX_DBG_SYNCTVB = BITFIELD_BIT(13),
   AGX_DBG_SMALLTILE = BITFIELD_BIT(14),
   AGX_DBG_NOMSAA = BITFIELD_BIT(15),
   AGX_DBG_NOSHADOW = BITFIELD_BIT(16),
   AGX_DBG_BODUMPVERBOSE = BITFIELD_BIT(17),
   AGX_DBG_SCRATCH = BITFIELD_BIT(18),
   AGX_DBG_NOSOFT = BITFIELD_BIT(19),
   AGX_DBG_FEEDBACK = BITFIELD_BIT(20),
   AGX_DBG_1QUEUE = BITFIELD_BIT(21),
   AGX_DBG_NOMERGE = BITFIELD_BIT(22),
   /* Track BOs without serializing submissions, then decode only a failed
    * submit while its command stream and referenced allocations are live. */
   AGX_DBG_FAULTTRACE = BITFIELD_BIT(23),
   /* Decode every command stream at submit time without waiting for the GPU. */
   AGX_DBG_ASYNCTRACE = BITFIELD_BIT(24),
};

/* How many power-of-two levels in the BO cache do we want? 2^14 minimum chosen
 * as it is the page size that all allocations are rounded to
 */
#define MIN_BO_CACHE_BUCKET (14) /* 2^14 = 16KB */
#define MAX_BO_CACHE_BUCKET (22) /* 2^22 = 4MB */

/* Fencepost problem, hence the off-by-one */
#define NR_BO_CACHE_FLAG_BUCKETS 16
#define NR_BO_CACHE_BUCKETS \
   ((MAX_BO_CACHE_BUCKET - MIN_BO_CACHE_BUCKET + 1) * NR_BO_CACHE_FLAG_BUCKETS)

/* Forward decl only, do not pull in all of NIR */
struct nir_shader;

#define BARRIER_RENDER  (1 << DRM_ASAHI_SUBQUEUE_RENDER)
#define BARRIER_COMPUTE (1 << DRM_ASAHI_SUBQUEUE_COMPUTE)

struct agx_submit_virt {
   uint32_t extres_count;
   struct asahi_ccmd_submit_res *extres;
   uint32_t ring_idx;
   uint32_t flags;
   uint32_t read_range_count;
   const struct agx_submit_read_range *read_ranges;
   uint32_t fence_dependency_count;
   const struct agx_submit_fence_dependency *fence_dependencies;
};

typedef struct {
   struct agx_bo *(*bo_alloc)(struct agx_device *dev, size_t size, size_t align,
                              enum agx_bo_flags flags);
   int (*bo_bind)(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
                  uint32_t count);
   /*
    * Optional WDDM queue interlock for tiled-resource page-table updates.
    * The update waits until `syncobj` reaches `wait_value`, then the paging
    * context publishes `wait_value + 1` to that same monitored fence.  This
    * is the D3DKMTUpdateGpuVirtualAddress ordering model; DRM backends leave
    * the hook NULL and retain their synchronous VM_BIND path.
    */
   int (*bo_bind_timeline)(struct agx_device *dev, uint32_t queue_id,
                           uint32_t syncobj, uint64_t wait_value,
                           struct drm_asahi_gem_bind_op *ops,
                           uint32_t count);
   void (*bo_mmap)(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
                   size_t length);
   ssize_t (*get_params)(struct agx_device *dev, void *buf, size_t size);
   int (*submit)(struct agx_device *dev, struct drm_asahi_submit *submit,
                 struct agx_submit_virt *virt);
   int (*bo_bind_object)(struct agx_device *dev,
                         struct drm_asahi_gem_bind_object *bind);
   int (*bo_unbind_object)(struct agx_device *dev, uint32_t object_handle);

   /*
    * honeykrisp-h1-execution.md SS1 "widen the table by three ops": lets
    * agx_device.c's remaining out-of-ops-table drm/OS call sites
    * (asahi_simple_ioctl's four callers -- VM_CREATE, QUEUE_CREATE,
    * QUEUE_DESTROY, GET_TIME -- plus agx_bo_free()'s munmap and GEM_CLOSE
    * tail) route through the ops table like every other backend operation,
    * so a non-libdrm backend (e.g. the Windows winsys, agx_device_win.c)
    * never needs its own copy of agx_bo_free()/agx_create_command_queue()/
    * agx_destroy_command_queue()/agx_get_gpu_timestamp() -- only these three
    * new leaf ops.
    *
    * simple_ioctl: passthrough for the four asahi_simple_ioctl callers;
    * `cmd` is always one of the DRM_IOCTL_ASAHI_* pinned literals. The Linux
    * body (asahi_simple_ioctl) already dispatches on dev->is_virtio itself,
    * so the SAME function is installed in both agx_device_drm_ops and
    * agx_virtio_device_ops.
    *
    * bo_munmap: unmap bo->_map (bo->size bytes). Called from agx_bo_free()
    * and before moving a cached CPU mapping to an application-selected
    * address for VK_EXT_map_memory_placed.
    *
    * bo_free: release the OS/kernel object referenced by `handle` (GEM_CLOSE
    * on Linux). Deliberately takes the raw handle, not `struct agx_bo *`,
    * unlike bo_munmap/bo_mmap: agx_bo_free() must memset() the `bo`
    * bookkeeping struct to all-zero *before* issuing the kernel-visible free
    * (see the comment at its call site -- avoids a lookup-table race with a
    * concurrently reused handle number), so by the time this op runs
    * `bo->handle` has already been zeroed and cannot be read back out of
    * `bo`.
    */
   int (*simple_ioctl)(struct agx_device *dev, unsigned cmd, void *req);
   void (*bo_munmap)(struct agx_device *dev, struct agx_bo *bo);
   int (*bo_sync_for_device)(struct agx_device *dev, struct agx_bo *bo,
                             size_t offset, size_t size);
   void (*bo_free)(struct agx_device *dev, uint32_t handle);

   /*
    * Optional process-VA hooks.  Linux/virtio leave these NULL and retain the
    * local util_vma_heap implementation byte-for-byte.  WDDM installs them
    * because every D3DKMT device opened by one process shares a GPU-VA root,
    * including devices opened from different UMD DLLs.
    */
   bool (*va_init)(struct agx_device *dev, uint64_t main_base,
                   uint64_t main_size, uint64_t usc_base, uint64_t usc_size);
   uint64_t (*va_alloc)(struct agx_device *dev, uint64_t size_B,
                        uint64_t align_B, enum agx_va_flags flags,
                        uint64_t fixed_va);
   void (*va_free)(struct agx_device *dev, uint64_t addr, uint64_t size_B,
                   enum agx_va_flags flags);

   /*
    * Optional fixed-ABI BO hook.  AGX2 shaders bake the zero, scratch and
    * printf addresses into code.  G17P instead carries printf through runtime
    * root state, leaving only zero, scratch, and its native render context as
    * fixed process mappings.  On WDDM fixed mappings must name one shared
    * backing allocation across every UMD in the process.
    */
   struct agx_bo *(*bo_alloc_fixed)(struct agx_device *dev, size_t size,
                                    enum agx_bo_flags flags, uint64_t fixed_va,
                                    uint32_t bind_flags, const char *label);
} agx_device_ops_t;

int agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
                size_t size_B, uint64_t offset_B, uint32_t flags);

int agx_bind_timestamps(struct agx_device *dev, struct agx_bo *bo,
                        uint32_t *handle);

/*
 * Linux/virtio-shared op bodies backing the three ops added just above.
 * Both agx_device_drm_ops (agx_device.c) and agx_virtio_device_ops
 * (agx_device_virtio.c) install the SAME functions here: generic DRM-core
 * behavior (GEM_CLOSE, plain munmap) and asahi_simple_ioctl's own existing
 * is_virtio dispatch don't differ between the two Linux backends -- exactly
 * why this code was unconditional (not backend-specific) before this ops
 * widening existed. Declared here (like agx_bo_bind/agx_bind_timestamps
 * above) so agx_device_virtio.c can reference the agx_device.c definitions.
 */
int asahi_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req);
void agx_drm_bo_munmap(struct agx_device *dev, struct agx_bo *bo);
int agx_drm_bo_sync_for_device(struct agx_device *dev, struct agx_bo *bo,
                               size_t offset, size_t size);
void agx_drm_bo_close(struct agx_device *dev, uint32_t handle);

struct agx_device {
   /* Optional allocator for linked executables: returns a referenced BO with
    * the allocation at *offset. Without it, each program gets its own BO. */
   struct agx_bo *(*code_alloc)(struct agx_device *dev, size_t size,
                                uint32_t *offset);

   uint32_t debug;

   /* Precompiled libagx binary table */
   const uint32_t **libagx_programs;

   char name[64];
   struct drm_asahi_params_global params;
   bool is_virtio;
   /* Native DRM 0.2 added the four render resource-specifier high dwords.
    * Stock 0.0 kernels do not consume an oversized render packet's tail. */
   bool render_rsrc_spec_hi;
   /* The current Aurora KMD treats attachment records as the authoritative
    * fragmented-allocation writeback set.  Other winsys backends retain the
    * upstream optional-hint semantics. */
   bool write_set_contract;
   agx_device_ops_t ops;

   /* vdrm device */
   struct vdrm_device *vdrm;
   uint32_t next_blob_id;

   /* Device handle */
   int fd;

   /*
    * honeykrisp-h1-execution.md SS1: Windows winsys handle (agx_device_win.c
    * agx_win_open_device()), used as the DeviceIoControl target for every
    * ntagx_ioctl() call. NULL/unused on Linux/virtio, where `fd` above is
    * the live handle instead; `fd` stays -1 on Windows (opaque pointer, not
    * cast to/from `int fd` -- real type is an NT HANDLE, kept void* here so
    * this header stays includable without <windows.h>).
    */
   void *hdev;

   /* VM handle */
   uint32_t vm_id;

   /* Global queue handle */
   uint32_t queue_id;

   /* VMA heaps */
   simple_mtx_t vma_lock;
   uint64_t shader_base;

   /* G17P only: the resident page bound at shader_base, so USC_EXEC_BASE_CP
    * names mapped memory the way it does in every m1n1 witness. */
   struct agx_bo *usc_base_bo;
   struct agx_bo *g17p_render_context_bo;
   struct util_vma_heap main_heap;
   struct util_vma_heap usc_heap;
   struct util_vma_heap replay_heap;
   /* Heap selection must stay stable when replay_heap.free_size reaches 0. */
   bool replay_heap_initialized;
   /* Native command-pool BO aliases with representable 32-bit context offsets.
    * Protected by vma_lock like the other heaps. */
   struct util_vma_heap g17p_vdm_alias_heap;
   uint64_t main_heap_base;
   uint64_t main_heap_size;
   uint64_t guard_size;

   /* To emulate sparse-resident buffers, we map buffers in both the bottom half
    * and top half of the address space. sparse_ro_offset controls the
    * partitioning. This is a power-of-two that &'s zero in bottom (read-write)
    * buffers but non-zero in top (read-only) shadow mappings.
    *
    * In other words, given an address X, we can check if it is in the top half
    * if (X & sparse_ro_offset) != 0.
    *
    * Given a bottom half address X, we can get the top half address
    * equivalently as (X + sparse_ro_offset) or (X | sparse_ro_offset).
    */
   uint64_t sparse_ro_offset;

   struct agx_bo *zero_bo, *scratch_bo;

   struct renderonly *ro;

   /*
    * honeykrisp-h1-execution.md SS2.1 "pthread inventory": simple_mtx_t
    * (already used elsewhere in this struct, e.g. vma_lock/bo_cache.lock
    * above) instead of pthread_mutex_t -- Windows-clean, mechanical rename,
    * no Linux behavior change (simple_mtx_t is a thin pthread wrapper
    * there).
    */
   simple_mtx_t bo_map_lock;
   struct util_sparse_array bo_map;
   /* Exclusive upper bound for every loop over bo_map. */
   uint32_t max_handle;

   /* D3DKMT's conservative CPU-read set is cached by this epoch. It changes
    * only on membership/range events, never on submission, so steady-state
    * queues do not rescan the complete BO map. Other backends leave
    * write_set_contract false and pay no atomic-update cost. */
   uint64_t bo_read_set_generation;

   struct {
      simple_mtx_t lock;

      /* List containing all cached BOs sorted in LRU (Least Recently Used)
       * order so we can quickly evict BOs that are more than 1 second old.
       */
      struct list_head lru;

      /* Power-of-two size classes, subdivided by allocation-flag hash.
       * Exact flags are still checked when fetching; collisions only affect
       * lookup cost. Each bucket is a list of free agx_bo objects.
       */
      struct list_head buckets[NR_BO_CACHE_BUCKETS];

      /* Current size of the BO cache in bytes (sum of sizes of cached BOs) */
      size_t size;

      /* Optional frontend-wide accounting, shared by logical devices. */
      uint64_t *size_counter;

      /* Monotonic CPU-side trim epoch. Cached BOs stamped before the current
       * epoch were not referenced during the latest periodic interval. */
      uint64_t trim_generation;

      /* Number of hits/misses for the BO cache */
      uint64_t hits, misses;

      /* DRM shmem pages remain pinned while a BO is cached. Wake only while
       * the cache has entries, so an idle client can release stale BOs. */
      mtx_t idle_lock;
      struct u_cnd_monotonic idle_cond;
      thrd_t idle_thread;
      bool idle_started, idle_stop, idle_pending;
   } bo_cache;

   struct agxdecode_ctx *agxdecode;

   /* Prepacked USC Sampler word to bind the txf sampler, used for
    * precompiled shaders on both drivers.
    */
   struct agx_usc_sampler_packed txf_sampler;

   /* Simplified device selection */
   enum agx_chip chip;

   struct {
      uint64_t num;
      uint64_t den;
   } user_timestamp_to_ns;

   struct u_printf_ctx printf;
   uint64_t printf_buffer_address;
};

static inline void
agx_bo_read_set_changed(struct agx_device *dev)
{
   if (dev->write_set_contract)
      p_atomic_inc(&dev->bo_read_set_generation);
}

static inline int
agx_bo_sync_for_device(struct agx_bo *bo, size_t offset, size_t size)
{
   assert(bo != NULL && bo->dev != NULL);
   assert(offset <= bo->size && size <= bo->size - offset);
   return bo->dev->ops.bo_sync_for_device(bo->dev, bo, offset, size);
}

/*
 * Determine if an address is in the read-only section. See the documentation
 * for sparse_ro_offset.
 */
static inline bool
agx_addr_is_ro(struct agx_device *dev, uint64_t addr)
{
   return (addr & dev->sparse_ro_offset);
}

/*
 * Convert a read-write address to its read-only shadow address. See the
 * documentation for sparse_ro_offset.
 */
static inline uint64_t
agx_rw_addr_to_ro(struct agx_device *dev, uint64_t addr)
{
   assert(!agx_addr_is_ro(dev, addr));
   return addr + dev->sparse_ro_offset;
}

/*
 * `length` bounds a PLACED mapping.  Zero means the whole BO, which is what
 * every ordinary map wants.
 *
 * A placed map may not be given the BO's size: agx_bo_cache_fetch() hands back
 * BOs up to twice the requested size (agx_bo.c), so a VkDeviceMemory of n bytes
 * can be backed by a BO of 2n, and mapping bo->size with MAP_FIXED at the
 * address the application reserved would land on whatever the application put
 * after its allocation.  VK_EXT_map_memory_placed requires exactly the memory
 * object's range; dEQP-VK.memory.map_placed.* reserves guard pages
 * immediately after it, fills them, and checks the pattern survives.
 */
/*
 * Bytes mapped at `bo->_map`. Backends that map the whole BO may leave
 * `_map_size` zero; everything that tears a mapping down must go through this
 * rather than assuming `bo->size`. See the field comment in agx_bo.h.
 */
static inline size_t
agx_bo_map_size(const struct agx_bo *bo)
{
   return bo->_map_size ?: bo->size;
}

static inline void *
agx_bo_map_placed(struct agx_bo *bo, void *fixed_addr, size_t length)
{
   /* The BO cache deliberately preserves CPU mappings.  A Vulkan allocation
    * may therefore inherit a mapping from an earlier allocation even though
    * the new VkDeviceMemory object has never been mapped.  A placed map must
    * move that stale process mapping to the application's reserved address;
    * returning it would violate VK_EXT_map_memory_placed and used to trip the
    * caller's exact-address assertion.
    *
    * Keep ordinary mapping reuse unchanged.  Only the explicit placed-map
    * contract is allowed to relocate an existing BO mapping. */
   if (fixed_addr && bo->_map && bo->_map != fixed_addr) {
      bo->dev->ops.bo_munmap(bo->dev, bo);
      bo->_map = NULL;
      bo->_map_size = 0;
   }

   if (!bo->_map)
      bo->dev->ops.bo_mmap(bo->dev, bo, fixed_addr, fixed_addr ? length : 0);

   return bo->_map;
}

static inline void *
agx_bo_map(struct agx_bo *bo)
{
   return agx_bo_map_placed(bo, NULL, 0);
}

static inline bool
agx_has_soft_fault(struct agx_device *dev)
{
   return (dev->params.features & DRM_ASAHI_FEATURE_SOFT_FAULTS) &&
          !(dev->debug & AGX_DBG_NOSOFT);
}

/* Base of the G17P USC exec window.  MEASURED RULE: bit 40 of the program VA
 * must be set.  Do not lower this.  See the note in agx_open_device(). */
#define AGX_G17P_USC_WINDOW_BASE UINT64_C(0x10000000000)

/* Metal's HAL200 direct encoder binds fixed offsets below +0x80000.  The
 * kernel-owned PM metrics mapping begins exactly at +0x80000. */
#define AGX_G17P_RENDER_CONTEXT_BASE UINT64_C(0x1000000000)
#define AGX_G17P_RENDER_CONTEXT_SIZE UINT64_C(0x00080000)

static uint32_t
agx_usc_addr(struct agx_device *dev, uint64_t addr)
{
   assert(addr >= dev->shader_base);
   assert((addr - dev->shader_base) <= UINT32_MAX);

   return addr - dev->shader_base;
}

/* The USC exec window the kernel is told about at queue-create time.
 *
 * agx_create_command_queue() passes dev->shader_base as
 * drm_asahi_queue_create.usc_exec_base, and the kernel programs it into
 * USC_EXEC_BASE_{TA,ISP,CP} for every command on that queue.  The uapi
 * contract is a fixed 4 GiB carveout at that base (asahi_drm.h: "we expect
 * userspace to fix a 4GiB VA carveout for USC memory and pass its base address
 * here"), which is exactly the size dev->usc_heap is initialised with.
 *
 * Note the window BASE PAGE is deliberately not required to be mapped:
 * util_vma_heap defaults to alloc_high, so USC allocations descend from
 * shader_base + 4 GiB and the first page of the window is never handed out
 * until the heap is nearly exhausted.  Residency of `shader_base` itself is
 * therefore not a valid health check -- containment of each individual
 * exec pointer is.
 */
#define AGX_USC_WINDOW_SIZE UINT64_C(0x100000000)

static inline bool
agx_va_in_usc_window(const struct agx_device *dev, uint64_t addr)
{
   return addr >= dev->shader_base &&
          (addr - dev->shader_base) < AGX_USC_WINDOW_SIZE;
}

/* On failure the caller retains its fd/platform handle. On success native
 * DRM transfers the fd to common close; Wine retains it in the platform so
 * no Unix descriptor is passed to the Windows CRT. */
bool agx_open_device(void *memctx, struct agx_device *dev);

void agx_close_device(struct agx_device *dev);

static inline struct agx_bo *
agx_lookup_bo(struct agx_device *dev, uint32_t handle)
{
   return util_sparse_array_get(&dev->bo_map, handle);
}

static inline void
agx_include_bo_handle(struct agx_device *dev, uint32_t handle)
{
   /* All backends reserve an all-ones handle as allocation failure. */
   assert(handle != UINT32_MAX);
   const uint32_t exclusive = handle + 1;
   if (exclusive > dev->max_handle)
      dev->max_handle = exclusive;
}

/* Returns UINT32_MAX on allocation failure, preserving the backend errno. */
uint32_t agx_create_command_queue(struct agx_device *dev,
                                  enum drm_asahi_priority priority);
int agx_destroy_command_queue(struct agx_device *dev, uint32_t queue_id);

int agx_import_sync_file(struct agx_device *dev, struct agx_bo *bo, int fd);
int agx_export_sync_file(struct agx_device *dev, struct agx_bo *bo);

void agx_debug_fault(struct agx_device *dev, uint64_t addr);

uint64_t agx_get_gpu_timestamp(struct agx_device *dev);

static inline uint64_t
agx_gpu_timestamp_to_ns(struct agx_device *dev, uint64_t gpu_timestamp)
{
   return (gpu_timestamp * dev->user_timestamp_to_ns.num) /
          dev->user_timestamp_to_ns.den;
}

void agx_get_device_uuid(const struct agx_device *dev, void *uuid);
void agx_get_driver_uuid(void *uuid);
unsigned agx_get_num_cores(const struct agx_device *dev);

struct agx_device_key agx_gather_device_key(struct agx_device *dev);

/* Native HAL200 graphics recording shared by G16G and G17P. */
static inline bool
agx_native_graphics(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G16G || dev->chip == AGX_CHIP_G17P;
}

struct agx_va *agx_va_alloc(struct agx_device *dev, uint64_t size_B,
                            uint64_t align_B, enum agx_va_flags flags,
                            uint64_t fixed_va);
void agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind);

static inline struct drm_asahi_cmd_header
agx_cmd_header(bool compute, uint16_t barrier_vdm, uint16_t barrier_cdm)
{
   return (struct drm_asahi_cmd_header){
      .cmd_type = compute ? DRM_ASAHI_CMD_COMPUTE : DRM_ASAHI_CMD_RENDER,
      .size = compute ? sizeof(struct drm_asahi_cmd_compute)
                      : sizeof(struct drm_asahi_cmd_render),
      .vdm_barrier = barrier_vdm,
      .cdm_barrier = barrier_cdm,
   };
}
