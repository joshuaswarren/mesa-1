/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx: experimental native Direct3D 12 user-mode driver for Apple AGX
 * GPUs, built directly on the Mesa/Asahi substrate (agx_device and friends,
 * the same layer Honeykrisp sits on). There is deliberately no Vulkan and no
 * SPIR-V anywhere in this driver.
 *
 * API ABI: microsoft/DirectX-Headers (MIT). D3D12 behavioral contract:
 * Microsoft public documentation. Hardware truth: src/asahi at the pinned
 * Mesa tag, and nothing else.
 *
 * Campaign front door: ntasp repo, docs/d3d12agx-campaign.md.
 */

#pragma once

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_sampler_reduction_core.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "d12_frontend.h"
#include "d12_agx3_vdm_state_update_core.h"
#include "d12_bundle_state_core.h"
#include "d12_command_validation_core.h"
#include "d12_device_removal_core.h"
#include "d12_native_descriptor_core.h"
#include "d12_queue_timeline_core.h"
#include "d12_sparse_residency_core.h"
#include "d12_render_pass_core.h"
#include "d12_sample_position_core.h"
#include "d12_raytracing_core.h"
#include "d12_viewport_math.h"

#include "agx_bg_eot.h"
#include "agx_compile.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "agx_helpers.h"
#include "agx_uvs.h"
#include "dxil/d12_dxil.h"
#include "layout.h"
#include "poly/nir/poly_nir.h"
#include "poly/tessellator.h"

#undef MESA_LOG_TAG
#define MESA_LOG_TAG "d3d12agx"
#include "util/log.h"
#include "util/macros.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "util/u_atomic.h"
#include "util/u_sync_provider.h"
#include "c11/threads.h"

struct d12_queue;

/* widl/Windows spells aggregate COM returns as an explicit out pointer,
 * while the same DirectX-Headers interfaces return them by value elsewhere. */
#ifdef _WIN32
#define D12_AGGREGATE_RETURN(type) type *
#define D12_AGGREGATE_OUT(type) , type *ret
#define D12_RETURN_AGGREGATE(value)                                          \
   do {                                                                       \
      *ret = (value);                                                         \
      return ret;                                                             \
   } while (0)
#else
#define D12_AGGREGATE_RETURN(type) type
#define D12_AGGREGATE_OUT(type)
#define D12_RETURN_AGGREGATE(value) return (value)
#endif

/*
 * COM plumbing. Every d12 object embeds its COM interface as the FIRST
 * member so the interface pointer and the object pointer are the same
 * address, exactly like vtable-first layouts elsewhere in Mesa.
 */

static inline bool
d12_guid_equal(REFIID a, REFIID b)
{
   return memcmp(a, b, sizeof(GUID)) == 0;
}

static inline bool
d12_handle_is_invalid(HANDLE handle)
{
#ifdef _WIN32
   return !handle || handle == INVALID_HANDLE_VALUE;
#else
   return !handle;
#endif
}

/* Crash-safe vtable defaults (generated). Every slot we do not implement
 * points at a typed stub that reports and fails, so an unimplemented call
 * is a recordable failure instead of a jump to NULL.
 */
#include "d12_stubs.h"

/* Log an unknown riid so we learn what callers actually want. */
void d12_log_unknown_riid(const char *where, REFIID riid);
uint32_t d12_default_sample_positions(unsigned nr_samples);

struct d12_private_data_entry {
   struct d12_private_data_entry *next;
   GUID guid;
   uint8_t *bytes;
   UINT size;
   IUnknown *object;
   bool is_interface;
};

struct d12_destruction_callback {
   PFN_DESTRUCTION_CALLBACK callback;
   void *data;
   UINT id;
};

/* ID3DDestructionNotifier is an embedded tear-off whose IUnknown methods
 * forward to the parent object, preserving COM identity and refcounts. */
struct d12_destruction_notifier {
   ID3DDestructionNotifier iface;
   IUnknown *parent;

   simple_mtx_t lock;
   struct d12_destruction_callback *callbacks;
   uint32_t callback_count;
   uint32_t callback_capacity;
   UINT next_callback_id;
   bool notified;
};

struct d12_private_data_store {
   simple_mtx_t lock;
   struct d12_private_data_entry *entries;
   struct d12_destruction_notifier destruction_notifier;
};

void d12_private_data_store_init(struct d12_private_data_store *store,
                                 IUnknown *parent);
void d12_private_data_store_destroy(struct d12_private_data_store *store);
bool d12_private_data_query_notifier(struct d12_private_data_store *store,
                                     REFIID riid, void **out);
HRESULT d12_private_data_get(struct d12_private_data_store *store,
                             REFGUID guid, UINT *size, void *data);
HRESULT d12_private_data_set(struct d12_private_data_store *store,
                             REFGUID guid, UINT size, const void *data);
HRESULT d12_private_data_set_interface(struct d12_private_data_store *store,
                                       REFGUID guid,
                                       const IUnknown *object);
HRESULT d12_private_data_set_name(struct d12_private_data_store *store,
                                  LPCWSTR name);

#define D12_DEFINE_PRIVATE_DATA_METHODS(prefix, iface_type, from_iface)       \
   static HRESULT STDMETHODCALLTYPE                                           \
   prefix##_GetPrivateData(iface_type *iface, REFGUID guid, UINT *size,       \
                           void *data)                                        \
   {                                                                          \
      return d12_private_data_get(&from_iface(iface)->private_data, guid,     \
                                  size, data);                                \
   }                                                                          \
                                                                              \
   static HRESULT STDMETHODCALLTYPE                                           \
   prefix##_SetPrivateData(iface_type *iface, REFGUID guid, UINT size,        \
                           const void *data)                                  \
   {                                                                          \
      return d12_private_data_set(&from_iface(iface)->private_data, guid,     \
                                  size, data);                                \
   }                                                                          \
                                                                              \
   static HRESULT STDMETHODCALLTYPE                                           \
   prefix##_SetPrivateDataInterface(iface_type *iface, REFGUID guid,          \
                                    const IUnknown *data)                     \
   {                                                                          \
      return d12_private_data_set_interface(                                  \
         &from_iface(iface)->private_data, guid, data);                       \
   }                                                                          \
                                                                              \
   static HRESULT STDMETHODCALLTYPE                                           \
   prefix##_SetName(iface_type *iface, LPCWSTR name)                          \
   {                                                                          \
      return d12_private_data_set_name(&from_iface(iface)->private_data,      \
                                       name);                                 \
   }

/*
 * Backing size for NULL descriptors. 64 KiB is D3D12's maximum constant
 * buffer size, so a null CBV is covered exactly for every offset the API
 * can legally address; a structured SRV/UAV indexed far past the end can
 * still walk off it, which needs per-access bounds checking rather than a
 * bigger buffer.
 */
#define D12_NULL_REGION_SIZE (64 * 1024)

/* d12_list_upload_vb_params returns 0 for "this pipeline fetches no vertex
 * attributes", which is a legitimate result, so failure needs a value that
 * is not a valid GPU address and is not zero.
 */
#define D12_VB_PARAMS_FAILED ((uint64_t)~0ull)

struct d12_device {
   /* One vtable answers every IID from ID3D12Device to ID3D12Device15: the
    * newer interfaces are strict supersets, so the slots are appended to the
    * ID3D12Device2 table (see struct d12_device_vtbl in d12_device.c). */
   ID3D12Device iface; /* must be first */
   /* D3D12Core configuration is a device-owned tear-off too.  It exposes
    * serialization helpers without changing the ID3D12Device2 ABI used by
    * the main interface. */
   ID3D12DeviceConfiguration1 iface_config1;
   uint32_t refcount;
   struct d12_private_data_store private_data;

   /* The real GPU. Owned: closed on final release. */
   struct agx_device dev;

   /* Platform sync objects use one provider surface on DRM and D3DKMT. */
   struct util_sync_provider *sync;

   /* The adapter identity DXGI and D3DKMT use for zero-copy interop. */
   LUID adapter_luid;

   /* Precompiled libagx kernel cache (copy/fill engines etc.), the same
    * mechanism Honeykrisp uses for its internal dispatches.
    */
   struct agx_bg_eot_cache bg_eot;

   /*
    * AGX register spills and NIR scratch are allocated by a hardware helper
    * program. Keep one grow-only pool per hardware stage class, matching
    * Honeykrisp. Command segments retain the exact BO they were recorded
    * against, so growing a pool cannot invalidate an in-flight submission.
    */
   simple_mtx_t scratch_lock;
   struct agx_scratch scratch_vs, scratch_fs, scratch_cs;

   /* CPU descriptor handles are opaque pointers into a heap's slot array.
    * View creation only receives that pointer, so retain a small registry to
    * locate the owning heap and mirror the slot into its GPU-native texture,
    * PBE or sampler table. Descriptor heaps hold a device ref, hence every
    * registry entry is removed before final device destruction. */
   simple_mtx_t descriptor_heap_lock;
   struct d12_descriptor_heap *descriptor_heaps;

   /* Root SRV/CBV setters receive only a raw GPU VA. Retain the live sparse
    * intervals so read-only root descriptors can select the zero-backed
    * mirror used for unmapped tiles. */
   simple_mtx_t sparse_resource_lock;
   struct list_head sparse_resources;

   /* Occlusion-query scratch: cmd_render.isp_oclqry_base must point at
    * valid memory even when no queries run (mirrors hk pointing it at the
    * device query heap).
    */
   struct agx_bo *oclqry_bo;

   /* One AGX_SAMPLER: the sampler state a texel fetch reads as ss0,
    * followed by the tessellation heap's `struct poly_heap` descriptor.
    */
   struct agx_bo *rodata_bo;

   /* THE TESSELLATION HEAP. Unlike geometry, tessellation genuinely needs
    * it: the tessellator allocates domain points from it, and the prefix
    * sum allocates the final index buffer out of it. The whole heap is
    * then bound as the index buffer, with out_draws[2] carrying the
    * element offset -- so it must be one contiguous allocation.
    *
    * `bottom` is reset per command list. hk queues a GPU write instead,
    * because a Vulkan command buffer can be resubmitted; a D3D12 command
    * list must be Reset before re-recording, so the CPU can zero it then.
    */
   struct agx_bo *heap_bo;
   uint64_t heap_desc_va; /* struct poly_heap inside rodata_bo */

   /* Zeroed backing for NULL descriptors, also inside rodata_bo. A null
    * SRV/UAV/CBV must read as zero, not fault; see D12_NULL_REGION_SIZE.
    */
   uint64_t null_va;
   struct poly_heap *heap_desc_cpu;

   D3D_FEATURE_LEVEL feature_level;

   /* Debug: the geometry vertex output buffer, dumped at device
    * destruction when D3D12AGX_DUMP_GS_VB is set. By then every
    * submission has completed.
    */
   void *dbg_gs_vb;
   size_t dbg_gs_vb_size;
   void *dbg_tess_draw;
   void *dbg_tess_counts;
   unsigned dbg_tess_counts_n;
   void *dbg_mesh_grid;
   void *dbg_mesh_disp;
   void *dbg_mesh_verts;
   void *dbg_mesh_idx;
   void *dbg_tcs_buf;
   unsigned dbg_tcs_buf_n;

   /* A one-shot failure transition. D3D12 distinguishes causes:
    * DXGI_ERROR_INVALID_CALL for an application error such as submitting a
    * command list still in the recording state, DXGI_ERROR_DEVICE_REMOVED
    * for a failure below us. The COM frontend reads it through
    * GetDeviceRemovedReason; the WDDM frontend registers a callback that
    * immediately publishes the same transition to the D3D12 core layer.
    */
   struct d12_device_removal_state removal;
};

static inline bool
d12_device_is_removed(const struct d12_device *device)
{
   return device && d12_device_removal_reason(&device->removal) != 0;
}

static inline HRESULT
d12_device_removed_reason(const struct d12_device *device)
{
   const int32_t reason =
      device ? d12_device_removal_reason(&device->removal) : 0;
   return reason ? (HRESULT)reason : S_OK;
}

static inline bool
d12_device_mark_removed(struct d12_device *device, HRESULT reason)
{
   return device && d12_device_removal_report(
                       &device->removal, (int32_t)reason);
}

/* Committed resource: one agx_bo. UMA means every heap type is the same
 * memory; Map is just the BO's CPU mapping. Buffers are flat; textures
 * carry an ail_layout (G3: 2D single-level RGBA8, LINEAR tiling so the CPU
 * can verify pixels directly).
 */
struct d12_resource {
   ID3D12Resource iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_RESOURCE_DESC desc;
   D3D12_HEAP_PROPERTIES heap_props;
   D3D12_HEAP_FLAGS heap_flags;
   struct agx_bo *bo;

   /* A reserved resource owns address space, not backing storage.  Asahi's
    * VM_BIND UAPI maps 16 KiB GEM ranges into this VMA as D3D12 64 KiB tiles
    * are updated. */
   bool reserved;
   struct agx_va *va;
   /* Reserved textures use an AGX hardware sparse table. Reserved buffers
    * use one 32-bit residency word per D3D12 64 KiB tile, consumed by the
    * DXIL status lowering. Both are coherent, queue-ordered GPU metadata. */
   struct agx_bo *sparse_map;
   uint64_t sparse_map_size_B;
   uint64_t sparse_size_B;
   uint32_t sparse_tile_count;
   uint32_t sparse_mip_count;
   uint32_t sparse_subresource_count;
   D3D12_PACKED_MIP_INFO sparse_packed_mips;
   D3D12_TILE_SHAPE sparse_tile_shape;
   D3D12_SUBRESOURCE_TILING *sparse_tilings;
   struct d12_sparse_mapping {
      struct agx_bo *bo;
      uint64_t heap_offset_B;
   } *sparse_mappings;
   simple_mtx_t sparse_lock;
   struct list_head sparse_link;

   /*
    * RESOLVED GPU ADDRESS, the same pattern hk uses for a bound image
    * (hk_image.c: plane->addr = mem->bo->va->addr + offset). A COMMITTED
    * resource owns its whole BO and sits at offset 0; a PLACED resource is
    * a window into a heap's BO. Everything downstream must read this and
    * never bo->va->addr, or a placed resource silently addresses the start
    * of the heap instead of its own storage.
    */
   uint64_t addr;
   uint64_t stencil_addr;   /* separate plane, same rule */
   uint64_t offset_B;       /* into bo, for CPU maps */
   bool placed;             /* does not own bo */
   /* Keep the placement properties by value. The storage BO is retained,
    * but D3D12 explicitly permits the public heap object to be released
    * while placed resources remain alive. A raw heap-object pointer would
    * therefore become dangling. */
   D3D12_HEAP_DESC placed_heap_desc;

   /* Textures only */
   struct ail_layout layout;

   /* Depth/stencil only. AGX stores stencil in a physically separate plane
    * from depth, so a D24S8/D32S8 resource is two allocations behind one
    * ID3D12Resource. `layout` above is the depth plane.
    */
   bool is_depth_stencil;
   struct agx_bo *stencil_bo; /* NULL when the format has no stencil */
   struct ail_layout stencil_layout;
};

/*
 * ID3D12Heap: one agx_bo of the requested size. On UMA every heap type is
 * the same memory, so a heap really is just an allocation that placed
 * resources carve windows out of.
 *
 * Placed resources hold a REFERENCE to it: D3D12 explicitly allows the
 * application to release the heap while placed resources are still alive,
 * and the storage must outlive them.
 */
struct d12_heap {
   ID3D12Heap iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_HEAP_DESC desc;
   struct agx_bo *bo;
};

static inline struct d12_heap *
d12_memheap_from_iface(ID3D12Heap *iface)
{
   return (struct d12_heap *)iface;
}

HRESULT d12_heap_create(struct d12_device *device,
                        const D3D12_HEAP_DESC *desc, REFIID riid,
                        void **out);
HRESULT d12_heap_create_from_bo(struct d12_device *device,
                               const D3D12_HEAP_DESC *desc,
                               struct agx_bo *bo, REFIID riid, void **out);

void d12_device_get_resource_allocation_info_runtime(
   struct d12_device *device, UINT visible_mask,
   const D3D12_RESOURCE_DESC *desc,
   D3D12_RESOURCE_ALLOCATION_INFO *info);

HRESULT d12_resource_create_placed(struct d12_device *device,
                                   struct d12_heap *heap, uint64_t offset_B,
                                   const D3D12_RESOURCE_DESC *desc,
                                   uint32_t row_pitch_B,
                                   uint32_t slice_pitch_B,
                                   D3D12_RESOURCE_STATES initial_state,
                                   const D3D12_CLEAR_VALUE *clear_value,
                                   REFIID riid, void **out);
HRESULT d12_resource_create_reused(struct d12_device *device,
                                   struct d12_resource *source,
                                   uint64_t source_offset_B,
                                   const D3D12_RESOURCE_DESC *desc,
                                   uint32_t row_pitch_B,
                                   uint32_t slice_pitch_B,
                                   D3D12_RESOURCE_STATES initial_state,
                                   const D3D12_CLEAR_VALUE *clear_value,
                                   REFIID riid, void **out);
HRESULT d12_resource_create_reserved(struct d12_device *device,
                                     const D3D12_RESOURCE_DESC *desc,
                                     D3D12_RESOURCE_STATES initial_state,
                                     const D3D12_CLEAR_VALUE *clear_value,
                                     REFIID riid, void **out);
HRESULT d12_resource_validate_reserved_tier(
   const D3D12_RESOURCE_DESC *desc);
bool d12_resource_explicit_row_major_size(
   const D3D12_RESOURCE_DESC *desc, uint32_t row_pitch_B,
   uint32_t slice_pitch_B, uint64_t *size_B);
HRESULT d12_resource_get_allocation_size(
   struct d12_device *device, const D3D12_RESOURCE_DESC *desc,
   uint32_t row_pitch_B,
   uint32_t slice_pitch_B, bool reserved, uint64_t *size_B);
struct d12_sparse_map_update {
   uint64_t read_address;
   uint32_t index_el;
   bool enabled;
};
HRESULT d12_sparse_update_tile_mappings(
   struct d12_resource *res, UINT num_regions,
   const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, struct d12_heap *heap,
   UINT num_ranges, const D3D12_TILE_RANGE_FLAGS *range_flags,
   const UINT *heap_starts, const UINT *range_counts,
   struct drm_asahi_gem_bind_op **out_ops, uint32_t *out_count,
   struct d12_sparse_map_update **out_map_updates,
   uint32_t *out_map_count);
HRESULT d12_sparse_copy_tile_mappings(
   struct d12_resource *dst, const D3D12_TILED_RESOURCE_COORDINATE *dst_start,
   struct d12_resource *src, const D3D12_TILED_RESOURCE_COORDINATE *src_start,
   const D3D12_TILE_REGION_SIZE *size,
   struct drm_asahi_gem_bind_op **out_ops, uint32_t *out_count,
   struct d12_sparse_map_update **out_map_updates,
   uint32_t *out_map_count);
bool d12_sparse_commit_map_updates(
   struct d12_resource *res,
   const struct d12_sparse_map_update *updates, uint32_t update_count);
HRESULT d12_queue_update_tile_mappings(
   struct d12_queue *queue, struct d12_resource *resource, UINT num_regions,
   const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, struct d12_heap *heap,
   UINT num_ranges, const D3D12_TILE_RANGE_FLAGS *range_flags,
   const UINT *heap_starts, const UINT *range_counts,
   D3D12_TILE_MAPPING_FLAGS flags);
HRESULT d12_queue_copy_tile_mappings(
   struct d12_queue *queue, struct d12_resource *destination,
   const D3D12_TILED_RESOURCE_COORDINATE *destination_start,
   struct d12_resource *source,
   const D3D12_TILED_RESOURCE_COORDINATE *source_start,
   const D3D12_TILE_REGION_SIZE *size, D3D12_TILE_MAPPING_FLAGS flags);
HRESULT d12_sparse_collect_tiles(
   const struct d12_resource *res, UINT num_regions,
   const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, uint32_t **out_tiles,
   uint32_t *out_count);
uint64_t d12_sparse_read_address(struct d12_device *device, uint64_t address);
struct d12_resource *d12_sparse_buffer_for_address(
   struct d12_device *device, uint64_t address);
uint64_t d12_sparse_buffer_residency_address(
   const struct d12_resource *res, uint64_t view_address);
struct d12_descriptor;
uint64_t d12_descriptor_sparse_residency_address(
   const struct d12_descriptor *descriptor);
struct d12_sparse_tile_copy_info {
   bool standard;
   uint32_t subresource;
   uint32_t x;
   uint32_t y;
   uint32_t z;
   uint64_t offset_B;
};
bool d12_sparse_get_tile_copy_info(
   const struct d12_resource *res, uint32_t tile,
   struct d12_sparse_tile_copy_info *info);
HRESULT d12_resource_create_shared_handle(
   struct d12_device *device, ID3D12DeviceChild *object,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle);
HRESULT d12_resource_open_shared_handle(struct d12_device *device,
                                        HANDLE handle, REFIID riid,
                                        void **out);
HRESULT d12_heap_open_shared_handle(struct d12_device *device, HANDLE handle,
                                    REFIID riid, void **out);
bool d12_resource_iface_is_ours(ID3D12Resource *iface);
bool d12_heap_iface_is_ours(ID3D12Heap *iface);

/*
 * One descriptor, whatever the heap type. Sized to exactly the advertised
 * handle increment so CPU-handle arithmetic in applications lands on slot
 * boundaries.
 *
 * Shader-visible descriptors carry a precomputed address rather than a
 * resource pointer plus offset: binding a descriptor table copies that
 * address straight into the root table, and nothing on the GPU ever reads
 * a descriptor.
 */
enum d12_descriptor_kind {
   D12_DESC_NONE = 0,
   D12_DESC_RTV,
   D12_DESC_DSV,
   D12_DESC_CBV,
   D12_DESC_SRV,
   D12_DESC_UAV,
   D12_DESC_SAMPLER,
};

enum d12_buffer_view_kind {
   D12_BUFFER_VIEW_NONE = 0,
   D12_BUFFER_VIEW_TYPED,
   D12_BUFFER_VIEW_RAW,
   D12_BUFFER_VIEW_STRUCTURED,
};

struct d12_descriptor {
   union {
      struct {
         /* GPU-read fields first: an SM 6.6 heap handle indexes this struct
          * IN MEMORY, so the shader-visible layout is part of the ABI. The
          * address must sit at offset 0 (resource_address reads it there) and
          * the UAV counter pointer at offset 8 (D12_DESC_COUNTER_OFFSET).
          */
         uint64_t address;         /* resource base + view offset */
         uint64_t counter_address; /* UAV counter, or 0 */

         struct d12_resource *resource; /* not referenced; app keeps alive */
         union {
            struct {
               uint32_t size_B;
               uint32_t stride_B;
            };

            /* Buffer SRV/UAV metadata mirrors the API instead of storing a
             * truncated byte extent.  Shaders reconstruct size in 64 bits. */
            struct {
               uint32_t buffer_count;
               uint32_t buffer_stride_B;
            };

            /* Texture views do not publish buffer metadata. Reuse those
             * two words for the selected layer range and mip while keeping
             * the descriptor's 40-byte shader-visible ABI unchanged. */
            struct {
               uint32_t texture_layers;
               uint32_t texture_level;
            };

            /* SRVs use the first texture metadata word for the API view
             * swizzle. UAVs instead use the layer/mip interpretation above.
             * The two descriptor kinds are disjoint, so this retains the
             * 40-byte shader-visible descriptor ABI. */
            struct {
               uint32_t texture_component_mapping : 13;
               uint32_t texture_first_level : 4;
               uint32_t texture_level_count : 5;
               uint32_t texture_txf_min_level : 5;
               uint32_t texture_min_lod_oob : 1;
               uint32_t texture_srv_reserved : 4;
               float texture_min_lod;
            };
         };
         DXGI_FORMAT format;
         uint8_t kind; /* enum d12_descriptor_kind */
         union {
            uint8_t buffer_kind; /* enum d12_buffer_view_kind */
            uint8_t dsv_flags;   /* D3D12_DSV_FLAGS */
            uint8_t texture_srv_layers;
         };
         uint8_t texture_plane;
         union {
            uint8_t texture_array;
            uint8_t texture_srv_dimension;
            uint8_t texture_uav_dimension;
         };
      };

      /* Sampler heaps are type-disjoint from resource heaps, so their slots
       * use the same advertised 40-byte increment for a packed AGX sampler
       * plus the state that AGX keeps outside that descriptor.  Retaining the
       * exact border colour and LOD bias makes this real descriptor storage,
       * not a CreateSampler no-op that happens to satisfy an object test. */
      struct {
         struct agx_sampler_packed sampler;
         float sampler_border[4];
         float sampler_lod_bias;
         uint32_t sampler_flags;
         uint32_t sampler_reserved;
         uint8_t sampler_kind; /* D12_DESC_SAMPLER, same offset as kind */
         uint8_t sampler_pad;
         /* Logical D3D12 heaps may contain 2,048 samplers, while one AGX
          * command exposes 1,024 native states.  Exact packed-state
          * deduplication writes the native index here; bindless shaders read
          * it from the coherent descriptor heap. */
         uint16_t sampler_native_index;
      };
   };
};

/* sampler_reserved and the GPU side table use the same flag word. */
#define D12_CUSTOM_BORDER_ENABLE    BITFIELD_BIT(0)
#define D12_CUSTOM_BORDER_POINT     BITFIELD_BIT(1)
#define D12_CUSTOM_BORDER_U         BITFIELD_BIT(2)
#define D12_CUSTOM_BORDER_V         BITFIELD_BIT(3)
#define D12_CUSTOM_BORDER_W         BITFIELD_BIT(4)
#define D12_SAMPLER_MIP_POINT       BITFIELD_BIT(5)

/*
 * D3D12 filter reduction modes. The runnable AGX2 packed sampler has no
 * reduction field, so a MINIMUM/MAXIMUM filter is reduced in the shader over
 * the same texel set the standard filter would weight. SamplerGen5 adds a
 * native reduction field for M5's all-linear/anisotropic subset; these flags
 * remain the portable fallback for AGX2 and mixed filters. Only a footprint
 * wider than one texel needs the shader work, so these bits are set only when
 * the reduction is actually observable.
 */
#define D12_SAMPLER_REDUCTION_MIN   BITFIELD_BIT(6)
#define D12_SAMPLER_REDUCTION_MAX   BITFIELD_BIT(7)
#define D12_SAMPLER_REDUCTION_MASK \
   (D12_SAMPLER_REDUCTION_MIN | D12_SAMPLER_REDUCTION_MAX)

struct d12_custom_border_entry {
   uint32_t flags;
   float lod_bias;
   uint32_t color[4];
};
static_assert(sizeof(struct d12_custom_border_entry) == 24,
              "custom-border table ABI");
static_assert(sizeof(struct d12_descriptor) == 40,
              "must match handle increment");
static_assert(offsetof(struct d12_descriptor, address) == 0 &&
                 offsetof(struct d12_descriptor, counter_address) == 8,
              "shader-visible ABI");
static_assert(offsetof(struct d12_descriptor, buffer_count) ==
                    D12_DESC_BUFFER_COUNT_OFFSET &&
                 offsetof(struct d12_descriptor, buffer_stride_B) ==
                    D12_DESC_BUFFER_STRIDE_OFFSET,
              "shader-visible buffer metadata ABI");
static_assert(offsetof(struct d12_descriptor, sampler_kind) ==
                 offsetof(struct d12_descriptor, kind),
              "descriptor kind offset is common to both heap classes");

/* The RTV/DSV path predates the unified descriptor and still speaks this
 * name; they are the same object.
 */
#define d12_rtv d12_descriptor

struct d12_descriptor_heap {
   ID3D12DescriptorHeap iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_DESCRIPTOR_HEAP_DESC desc;
   struct d12_descriptor *slots; /* NumDescriptors entries */
   /* Shader-visible heaps are coherent GPU BOs.  Finite descriptor tables
    * are still snapshotted into the root table, while unbounded ranges and
    * SM 6.6 heap handles dereference this storage directly. */
   struct agx_bo *bo;

   /* AGX does not consume D3D12's 40-byte descriptor directly for images.
    * Shader-visible resource heaps therefore keep parallel same-index
    * hardware TEXTURE/PBE tables. Sampler heaps similarly keep the packed
    * hardware sampler table named by drm_asahi_cmd_*::sampler_heap. */
   struct agx_bo *texture_bo;
   struct agx_bo *pbe_bo;
   uint8_t texture_stride_B;
   uint8_t pbe_stride_B;
   /* One 64-bit sparse residency-map pointer per logical resource
    * descriptor. Zero keeps dense resources on the no-load fast path. */
   struct agx_bo *residency_bo;
   struct agx_bo *sampler_bo;
   uint32_t sampler_unique_count;

   /* Exact WDDM read-set construction must inspect descriptors which can be
    * reached through an unbounded table or SM 6.6 heap handle. A maximum
    * resource heap contains one million slots, so walking every slot on
    * every ExecuteCommandLists call is not viable. Maintain a compact CPU
    * occupancy bitmap as descriptors are written and visit only live slots.
    * descriptor_heap_lock protects this bitmap and the matching slot reads.
    */
   uint64_t *live_resource_words;
   uint32_t live_resource_word_count;

   struct d12_descriptor_heap *registry_next;
};

/* A finite descriptor table is flattened into the command-list root and
 * texture heaps for AGX.  D3D12 nevertheless permits the application to
 * update a shader-visible descriptor heap after Close and before Execute.
 * Retain the source slot and the two flattened destinations so Execute can
 * refresh exactly the cells that remain live.
 */
struct d12_descriptor_fixup {
   const struct d12_descriptor *source;
   uint64_t tables_gpu;
   uint64_t textures_gpu;
   uint64_t pbes_gpu;
   uint32_t slot;
   uint16_t table_count;
   uint8_t texture_shape;
   bool native_only;
};

/* ExecuteIndirect root constants are sourced from the argument buffer at
 * execution time, not when the command list is recorded.  The draw's root
 * table points at command-list upload storage; this fixup refreshes that
 * storage immediately before submission, just like mutable descriptor-table
 * fixups refresh their flattened cells. */
struct d12_indirect_constant_fixup {
   const struct d12_resource *source;
   uint64_t source_offset_B;
   uint64_t destination_gpu;
   uint32_t size_B;
};

/* A CPU-visible indirect count buffer controls which pre-recorded terminal
 * commands are live.  Each terminal argument block is copied into bounded
 * command-list storage at execution; commands beyond the current count receive
 * a zero argument block and are therefore GPU no-ops. */
struct d12_indirect_argument_fixup {
   const struct d12_resource *source;
   const struct d12_resource *count_source;
   uint64_t source_offset_B;
   uint64_t count_offset_B;
   uint64_t destination_gpu;
   uint32_t size_B;
   uint32_t command_index;
};

/*
 * Bundles are recorded before the direct command list state they inherit is
 * known.  Keep draw calls logical until ExecuteBundle so render targets,
 * viewport/scissor and root bindings come from the calling list.  Pipeline
 * state and primitive topology are deliberately snapshotted separately:
 * D3D12 does not inherit either of those into a bundle.
 */
struct d12_bundle_draw {
   uint32_t vertex_count;
   uint32_t instance_count;
   uint32_t first_vertex;
   uint32_t first_instance;
   uint32_t first_index;
   bool indexed;

   struct d12_pipeline_state *pso;
   struct d12_root_signature *graphics_rs;
   enum mesa_prim topology;
   unsigned patch_control_points;
   bool graphics_rs_set;
   bool topology_set;
   struct d12_bundle_dynamic_state dynamic_state;
};

/*
 * A query heap: storage for Count results of one type. The heap is just
 * a buffer; Begin/End/Resolve give it meaning.
 */
struct d12_query_heap {
   ID3D12QueryHeap iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_QUERY_HEAP_DESC desc;
   D3D12_QUERY_HEAP_FLAGS flags;
   unsigned stride;   /* bytes per result */
   size_t start_offset; /* occlusion BeginQuery snapshots, 0 otherwise */
   struct agx_bo *bo;
};

unsigned d12_query_result_size(D3D12_QUERY_HEAP_TYPE type);
HRESULT d12_query_heap_create(struct d12_device *device,
                              const D3D12_QUERY_HEAP_DESC *desc, REFIID riid,
                              void **out);
HRESULT d12_query_heap_create_flags(struct d12_device *device,
                                    const D3D12_QUERY_HEAP_DESC *desc,
                                    D3D12_QUERY_HEAP_FLAGS flags,
                                    REFIID riid, void **out);

/*
 * The parsed root signature.
 *
 * This is the only thing that connects a root PARAMETER index -- what
 * SetGraphicsRootDescriptorTable and friends name -- to the shader
 * REGISTER a shader actually asks for. Without it the driver has to assume
 * they are the same number, which holds only by coincidence.
 */
#define D12_MAX_ROOT_PARAMS 64
/* Bound aggregate parser storage, not the number of ranges in one table.
 * Descriptor-table records cost one root DWORD regardless of their range
 * count, so imposing a small per-table ceiling rejects otherwise legal root
 * signatures.  1024 matches the pointer-free UMD pipeline-cache format and
 * keeps malformed shader containers from requesting unbounded allocations.
 */
#define D12_MAX_ROOT_RANGES 1024

struct d12_root_range {
   D3D12_DESCRIPTOR_RANGE_TYPE type;
   uint32_t num_descriptors;
   uint32_t base_register;
   uint32_t space;
   uint32_t offset_in_descriptors; /* ~0u means "append" */
};

struct d12_root_param {
   D3D12_ROOT_PARAMETER_TYPE type;
   D3D12_SHADER_VISIBILITY visibility;

   /* Root descriptors and root constants */
   uint32_t base_register;
   uint32_t space;
   uint32_t num_constants;

   /* Descriptor tables */
   uint32_t first_range;
   uint32_t num_ranges;
};

struct d12_root_layout {
   struct d12_root_param params[D12_MAX_ROOT_PARAMS];
   uint32_t num_params;
   struct d12_root_range *ranges;
   uint32_t num_ranges;
   D3D12_STATIC_SAMPLER_DESC1 *static_samplers;
   uint32_t num_static_samplers;
   uint32_t flags;
};

static inline const struct d12_root_range *
d12_root_param_ranges(const struct d12_root_layout *layout,
                      const struct d12_root_param *param)
{
   return param->num_ranges ? &layout->ranges[param->first_range] : NULL;
}

static inline unsigned
d12_root_param_range_base(const struct d12_root_layout *layout,
                          unsigned param_index)
{
   unsigned root_range = 0;
   for (unsigned i = 0; i < param_index; i++) {
      root_range += layout->params[i].type ==
                       D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE
                       ? layout->params[i].num_ranges
                       : 1;
   }
   return root_range;
}

static inline unsigned
d12_static_sampler_root_range(const struct d12_root_layout *layout,
                              unsigned sampler_index)
{
   return d12_root_param_range_base(layout, layout->num_params) +
          sampler_index;
}

struct d12_root_signature {
   ID3D12RootSignature iface; /* must be first */
   uint32_t refcount;
   struct d12_device *device; /* holds a device ref */
   struct d12_root_layout layout;
   struct d12_private_data_store private_data;

   /* Packed in declaration order. Shader visibility may make two records
    * with the same register/space legal for disjoint graphics stages, so
    * retain one runtime descriptor per serialized record. */
   struct d12_descriptor *static_samplers;

   /* The slot assignment derived from the layout. Shaders are compiled
    * against it and the command list fills the root table through it.
    */
   struct d12_binding_map map;
};

/* Root signature blobs (d12_root_sig.c) */
HRESULT d12_serialize_root_signature(const D3D12_ROOT_SIGNATURE_DESC *desc,
                                     void **out_blob, size_t *out_size);
HRESULT d12_serialize_versioned_root_signature(
   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc, void **out_blob,
   size_t *out_size);
HRESULT d12_root_signature_deserializer_create(const void *src, size_t size,
                                               REFIID riid, void **out);
HRESULT d12_versioned_root_signature_deserializer_create(
   const void *src, size_t size, REFIID riid, void **out);
HRESULT d12_versioned_root_signature_deserializer_create_from_library(
   const void *src, size_t size, LPCWSTR subobject_name, REFIID riid,
   void **out);
bool d12_root_signature_parse(const void *blob, size_t blob_size,
                              struct d12_root_layout *out);
bool d12_root_signature_has_rts0(const void *blob, size_t blob_size);
void d12_root_layout_cleanup(struct d12_root_layout *layout);
unsigned d12_root_param_class(const struct d12_root_param *rp);
unsigned d12_descriptor_range_class(D3D12_DESCRIPTOR_RANGE_TYPE type);
/* False when the signature declares more distinct bindings than the dense
 * slot table can name; the map is then incomplete and must not be used. */
bool d12_root_layout_build_map(const struct d12_root_layout *layout,
                               struct d12_binding_map *out);
ID3DBlob *d12_blob_create(const void *data, size_t size);

/* One linked graphics stage */
struct d12_gfx_stage {
   /* Frontend scan before texture lowering: only root/buffer resources. */
   bool g17p_root_only;
   bool g17p_writes_memory;
   struct agx_shader_info info;
   struct agx_linked_shader linked;

   /* The compiled part's own BO, and the address of its preamble inside
    * it. agx_fast_link links main + prolog + epilog only: the PREAMBLE is
    * not copied into linked.bo, so info.preamble_offset is meaningless
    * there. hk keeps the same thing as hk_shader::preamble_addr.
    *
    * Zero when the shader has no preamble.
    */
   struct agx_bo *code_bo;
   uint64_t preamble_va;
   struct agx_counts_packed counts;                  /* vertex */
   /* Exact shader-lifetime AGX3 template axes.  Zero is UNSPECIFIED and
    * therefore rejects live AGX3 draw emission while the compiler still
    * exposes only AGX2 Counts/UVS state. */
   struct d12_agx3_vdm_state_update_program_info agx3_vdm_state_update;
   struct agx_fragment_shader_word_0_packed fs_counts; /* fragment */
   struct agx_fragment_face_2_packed frag_face;        /* fragment */
};

/* One face's stencil test, translated to hardware enums at PSO creation so
 * the draw path pushes packed state rather than re-translating per draw.
 */
struct d12_stencil_face {
   enum agx_zs_func compare;
   enum agx_stencil_op pass, depth_fail, stencil_fail;
};

/* The pipeline's depth/stencil state, already in AGX terms. */
struct d12_depth_stencil_state {
   bool depth_test;  /* DepthEnable */
   bool depth_write; /* DepthWriteMask == ALL */
   enum agx_zs_func depth_func;

   bool stencil_test;
   uint8_t front_stencil_read_mask, front_stencil_write_mask;
   uint8_t back_stencil_read_mask, back_stencil_write_mask;
   struct d12_stencil_face front, back;

   DXGI_FORMAT dsv_format; /* DSVFormat from the PSO desc */
};

/* Rasterizer state that reaches the CULL block. Facing was hardcoded until
 * the depth work measured which winding the hardware calls front.
 */
struct d12_raster_state {
   bool cull_front, cull_back;
   bool front_face_ccw;
   bool depth_clip;
   bool smooth_line;
   enum agx_polygon_mode polygon_mode;
   /* D3D12_RASTERIZER_DESC::ForcedSampleCount. 0 means the attachments pick
    * the rasterization sample count; non-zero forces it for target-independent
    * rasterization, which D3D12 only permits with no depth-stencil and
    * single-sample (or absent) render targets. */
   uint32_t forced_sample_count;
   bool dynamic_depth_bias;
   bool dynamic_strip_cut;
   float depth_bias;
   float depth_bias_clamp;
   float slope_scaled_depth_bias;
};

struct d12_pipeline_library_key {
   uint64_t lo, hi;
};

/* A compiled pipeline: AGX machine code produced from the app's DXIL. */
struct d12_pipeline_state {
   ID3D12PipelineState iface; /* must be first */
   uint32_t refcount;
   /* Pipeline libraries retain a hidden ownership reference. It deliberately
    * does not change the public COM count: a stored PSO may reach public
    * refcount zero, retain its private data, and later be returned again from
    * the same library as required by D3D12. */
   uint32_t library_refcount;
   bool device_ref_active;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   /* Driver-private validation blob returned by GetCachedBlob.  The current
    * implementation still recompiles cached PSOs, but validates that the
    * supplied shader and canonical root layout are an exact match. */
   void *cached_blob;
   size_t cached_blob_size;
   bool is_compute;
   struct d12_pipeline_library_key library_key;
   bool library_key_valid;
   /* A null pixel shader still uses an empty hardware fragment program for
    * depth/stencil rasterization, but it has no colour attachments or stores.
    */
   bool has_pixel_shader;
   /* False only for a stream-output PSO whose RasterizedStream is explicitly
    * D3D12_SO_NO_RASTERIZED_STREAM. */
   bool rasterization_enabled;

   /* Shader-declared texture shape for every dense resource slot. This is
    * intentionally PSO state rather than descriptor state: the same D3D12
    * descriptor may legally be read as an array in one register space and
    * as a non-array in another. */
   uint8_t texture_shapes[D12_MAX_RESOURCE_BINDINGS];

   /* Compute */
   struct agx_bo *code_bo;
   uint64_t code_va;
   struct agx_shader_info info;
   /* Native compute code_va is the CODE entry, never the allocation base. */
   bool g17p_compute_container;
   /* Native indexed Texture.Load ABI with texture/sampler pointer prefix. */
   bool g17p_native_txf;
   uint64_t g17p_rodata_va;

   /* VERTEX ATTRIBUTES. Resolved at PSO creation by matching each
    * D3D12_INPUT_ELEMENT_DESC's (SemanticName, SemanticIndex) against the
    * vertex shader's own DXIL input signature, so the index here is the
    * signature element id -- which is what agx_nir_lower_vbo keys on.
    *
    * nr_vb_attribs is 0 for a PSO with no input layout, and that case
    * keeps the old static_vi prolog exactly as it was.
    */
   struct d12_vb_attrib {
      bool live;
      uint8_t slot;              /* D3D12 InputSlot -> which VBV */
      uint32_t offset_B;         /* AlignedByteOffset */
      enum pipe_format format;
      uint32_t interchange_align; /* to turn a byte stride into elements */
   } vb_attribs[D12_MAX_VB_ATTRIBS];
   unsigned nr_vb_attribs;

   /* Graphics */
   struct d12_gfx_stage vs, fs;
   /* Exact layout used while lowering fragment outputs. A lazily opened
    * render pass must publish the same layout to the hardware even when no
    * colour attachment is bound (the shader's outputs are then discarded at
    * EOT, but they still exist in the tilebuffer while rasterizing).
    */
   struct agx_tilebuffer_layout tib;
   /*
    * Independent-blend PSOs whose declared RTVs all have one format use a
    * dense, slot-stable tilebuffer ABI.  UNKNOWN RTV slots do not write, but
    * reserving their slots lets a bound framebuffer keep one render pass
    * while PSOs vary which subset is live.
    */
   bool stable_unused_rt_layout;
   struct agx_unlinked_uvs_layout uvs;
   struct agx_varyings_vs varyings;
   /* Render targets the pixel shader can actually modify after applying the
    * D3D12 write mask. Bound-but-unwritten attachments must not get an EOT
    * store: doing so turns a stale background load into an observable write.
    */
   uint8_t rt_write_mask;
   struct d12_depth_stencil_state ds;
   struct d12_raster_state raster;
   D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut;

   /* GEOMETRY. AGX has no hardware geometry stage: a GS becomes compute
    * programs whose output the rasterization draw consumes, so a
    * geometry pipeline carries three extra programs rather than one.
    *
    *   sw_vs    the app's vertex shader, rewritten by
    *            poly_nir_lower_vs_before_gs to write its outputs to a
    *            buffer instead of the rasterizer. Compute.
    *   gs_main  the geometry shader body. Compute. Often empty --
    *            poly deletes its store_output, so a GS with no side
    *            effects and static topology compiles to a few bytes.
    *   gs_rast  re-runs the GS body once per output vertex and feeds the
    *            rasterizer, so it is the HARDWARE vertex shader and
    *            replaces pso->vs at draw time.
    *
    * The compute programs are plain code BOs like the compute PSO's;
    * gs_rast is fast-linked like any other vertex stage.
    */
   bool has_gs;
   struct poly_gs_info gs_info;

   /*
    * Native AGX flat coefficient state is not usable for a D3D12 link which
    * mixes integer and floating-point flat varyings.  Such a link is routed
    * through Poly, whose pass-through geometry stage copies the provoking
    * vertex value to every raster vertex.  Poly consumes this location mask
    * from its per-draw parameter block.
    */
   uint64_t flat_outputs;

   /* The software VS output mask, captured BEFORE the poly lowering
    * erases it. The geometry programs address the vertex buffer with it.
    */
   uint64_t sw_vs_outputs;

   struct agx_bo *sw_vs_bo, *gs_count_bo, *gs_pre_bo, *gs_main_bo;
   uint64_t sw_vs_va, gs_count_va, gs_pre_va, gs_main_va;
   struct agx_shader_info sw_vs_info, gs_count_info, gs_pre_info,
      gs_main_info;

   /* When a geometry stage (including the driver-created XFB passthrough)
    * follows tessellation, the software vertex stage immediately ahead of
    * that geometry stage is the domain shader, not the application's vertex
    * shader.  Keep the two compute variants separate: sw_vs still feeds the
    * hull shader, while sw_tes consumes the tessellator's indirect draw and
    * feeds geometry/XFB.
    */
   struct agx_bo *sw_tes_bo;
   uint64_t sw_tes_va;
   struct agx_shader_info sw_tes_info;
   uint64_t sw_tes_outputs;

   struct d12_gfx_stage gs_rast;
   struct agx_unlinked_uvs_layout gs_uvs;
   struct agx_varyings_vs gs_varyings;

   /* TESSELLATION. Same shape as geometry: the vertex shader becomes a
    * compute prepass, the hull shader is compute, and the DOMAIN shader
    * becomes the hardware vertex shader -- the tessellator writes an
    * index buffer and the domain shader is invoked per index.
    *
    * The mode/spacing/winding are declared by the SHADERS, not by API
    * state, and either stage may declare them, so they are merged.
    */
   bool has_tess;
   struct agx_bo *tcs_bo;
   uint64_t tcs_va;
   struct agx_shader_info tcs_info;

   struct d12_gfx_stage tes;
   struct agx_unlinked_uvs_layout tes_uvs;
   struct agx_varyings_vs tes_varyings;

   uint32_t tcs_output_patch_size;
   uint64_t tcs_per_vertex_outputs;
   uint32_t tcs_nr_patch_outputs;
   uint32_t tcs_output_stride;

   /* MESH. There is no hk or poly emulation to borrow -- hk implements
    * no mesh shaders at all -- so the whole path is this driver's own.
    * The mesh shader is compiled as compute and writes through
    * D12_UNIFORM_MESH_PARAMS; two internal NIR shaders complete it:
    *
    *   ms_setup  reads the GPU-written {vertices, primitives} counts and
    *             fills the 5-word indirect draw descriptor. The counts
    *             only exist after the mesh shader has run, so this
    *             cannot be done on the CPU.
    *   ms_rast   the hardware vertex shader: reads the mesh shader's
    *             per-vertex output slots and feeds the rasterizer.
    */
   bool has_mesh;
   struct agx_bo *ms_bo;
   uint64_t ms_va;
   struct agx_shader_info ms_info;
   uint32_t ms_workgroup[3];

   struct agx_bo *ms_setup_bo;
   uint64_t ms_setup_va;
   struct agx_shader_info ms_setup_info;

   /* Amplification: runs before the mesh shader, writes the payload and
    * the DispatchMesh grid; as_setup turns that grid into the six-word
    * indirect-local form the mesh dispatch consumes.
    */
   bool has_amp;
   struct agx_bo *as_bo, *as_setup_bo;
   uint64_t as_va, as_setup_va;
   struct agx_shader_info as_info, as_setup_info;
   uint32_t as_workgroup[3];
   uint32_t as_payload_size;

   struct d12_gfx_stage ms_rast;
   struct agx_unlinked_uvs_layout ms_uvs;
   struct agx_varyings_vs ms_varyings;
   uint64_t ms_outputs;

   enum tess_primitive_mode tess_mode;
   enum gl_tess_spacing tess_spacing;
   bool tess_ccw, tess_points;
};

static inline struct d12_pipeline_state *
d12_pipeline_state_from_iface(ID3D12PipelineState *iface)
{
   return (struct d12_pipeline_state *)iface;
}

struct d12_cmd_allocator {
   ID3D12CommandAllocator iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_COMMAND_LIST_TYPE type;

   /* Lists derived from this allocator that are currently recording.
    * Reset must fail while any are.
    */
   unsigned recording_lists;

};

enum d12_list_state {
   D12_LIST_RECORDING,
   D12_LIST_CLOSED,
   D12_LIST_ERROR,
};

/* A command list is a sequence of hardware command segments, submitted as
 * one multi-command drm_asahi_submit in order:
 *
 * - CDM segments: compute control streams (copies etc.), encoded at record
 *   time exactly the way hk encodes internal dispatches (agx_cdm_launch +
 *   agx_cdm_barrier per dispatch, agx_cdm_terminate at segment close).
 * - RENDER segments: one per ClearRenderTargetView for now — a complete
 *   clear-only tiler pass (background program AGX_BG_CLEAR, end-of-tile
 *   AGX_EOT_STORE, empty VDM stream, PROCESS_EMPTY_TILES).
 *
 * Arg blobs and descriptors bump down from the top of the stream BO; USC
 * words live in a separate AGX_BO_LOW_VA BO for agx_usc_addr.
 */

enum d12_segment_type {
   D12_SEG_CDM,
   D12_SEG_RENDER,
};

enum d12_write_pipe {
   D12_WRITE_VERTEX,
   D12_WRITE_FRAGMENT,
   D12_WRITE_COMPUTE,
   D12_WRITE_PIPE_COUNT,
};

enum d12_write_ref_kind {
   /* A fixed GPU range known while recording.  A zero size means the queue
    * must resolve the address to its complete owning BO before submission
    * (root UAV descriptors carry an address but no API-visible extent). */
   D12_WRITE_REF_RANGE,
   /* A mutable shader-visible descriptor.  D3D12 permits descriptor updates
    * after Close and before ExecuteCommandLists, so it is evaluated only
    * after the normal descriptor refresh at submission time. */
   D12_WRITE_REF_DESCRIPTOR,
   /* An unbounded range or SM 6.6 directly-indexed resource heap. */
   D12_WRITE_REF_HEAP,
   /* A reserved UAV resolves at Execute time to the backing allocations of
    * its then-current tile mapping. */
   D12_WRITE_REF_RESOURCE,
};

struct d12_write_ref {
   enum d12_write_ref_kind kind;
   union {
      struct {
         uint64_t address;
         uint64_t size;
      } range;
      const struct d12_descriptor *descriptor;
      struct d12_resource *resource;
      struct {
         const struct d12_descriptor_heap *heap;
         uint32_t first;
         /* Descriptors covered from `first`, or 0 for "to the end of the
          * heap".  See struct d12_read_ref. */
         uint32_t count;
      } heap;
   };
};

struct d12_write_tracker {
   struct util_dynarray refs; /* struct d12_write_ref */
   bool initialized;
   /* Set when a legal command can write memory whose owning allocation is
    * not derivable on the host.  A Windows submit must never publish the
    * complete-write marker while this is true. */
   bool incomplete;
};

enum d12_read_ref_kind {
   /* Immutable BOs are retained by the command list until Execute captures
    * them into the queue operation. */
   D12_READ_REF_BO,
   /* Root descriptors and other address-only commands are resolved to their
    * owning BO at ExecuteCommandLists. A zero length means the whole BO. */
   D12_READ_REF_RANGE,
   /* Mutable shader-visible descriptors are evaluated after the normal
    * pre-submit descriptor refresh. */
   D12_READ_REF_DESCRIPTOR,
   /* Unbounded/directly-indexed heaps require both their native backing BOs
    * and every live resource descriptor from first onward. */
   D12_READ_REF_HEAP,
   /* A reserved resource resolves at Execute time to its sparse page table
    * and the then-current backing allocations. */
   D12_READ_REF_RESOURCE,
};

struct d12_read_ref {
   enum d12_read_ref_kind kind;
   union {
      struct agx_bo *bo;
      struct {
         uint64_t address;
         uint64_t size;
      } range;
      const struct d12_descriptor *descriptor;
      struct d12_resource *resource;
      struct {
         const struct d12_descriptor_heap *heap;
         uint32_t first;
         /* Descriptors covered from `first`, or 0 for "to the end of the
          * heap".  A root range the signature sized explicitly ends there;
          * declaring the whole tail of a million-slot heap would put every
          * later populated descriptor's allocation into the submission's
          * read set. */
         uint32_t count;
         bool resources;
      } heap;
   };
};

struct d12_read_tracker {
   struct util_dynarray refs; /* struct d12_read_ref */
   bool initialized;
   /* Any legal command whose referenced allocation cannot be derived on the
    * host forces the conservative winsys scan for the entire submission. */
   bool incomplete;
};

struct d12_segment {
   /* Revalidate the admission used when recording each native texture path. */
   bool g17p_txf_metadata_only;
   bool g17p_txf_physical;
   /* Base addresses of the render target SLICES this pass writes. Array and
    * 3D RTVs may select a bounded layer range; layers are contiguous at
    * layer_stride_B, so the view carries an offset the resource does not.
    * Using rt->bo->va->addr here would write every such view to layer 0.
    */
   uint64_t rt_address[AGX_MAX_RENDER_TARGETS];

   enum d12_segment_type type;

   /* GPU write ownership for the three firmware pipe classes.  Trackers are
    * dynamic because a command list may contain 2,048 segments; embedding
    * sixteen attachment records per pipe would add several MiB to every
    * command-list object even when no UAV is used. */
   struct d12_write_tracker writes[D12_WRITE_PIPE_COUNT];

   /* Control-stream extent within stream_bo (start/end offsets). CDM: the
    * dispatch stream. RENDER: the (empty) VDM stream.
    */
   size_t stream_start;
   size_t stream_end;
   uint64_t g17p_vdm_control_base;

   /* CDM */
   uint32_t dispatches;

   /* Bindless AGX samplers are addressed from one heap named by the DRM
    * command, not by a USC sampler-state record. Capture the heap selected
    * while this segment was recorded so a later SetDescriptorHeaps cannot
    * retarget commands already in its control stream. */
   uint64_t sampler_heap;
   uint32_t sampler_count;

   /* Exact helper-data BOs required by this hardware command. These are
    * referenced by the segment until the command allocator guarantees that
    * a submitted recording is idle. */
   struct agx_bo *scratch_cs_bo;
   struct agx_bo *scratch_vs_bo;
   struct agx_bo *scratch_fs_bo;
   bool scratch_cs_preamble;
   bool scratch_vs_preamble;
   bool scratch_fs_preamble;

   /* RENDER */
   uint16_t width_px, height_px;
   uint16_t layers;
   struct agx_tilebuffer_layout tib;
   /* RenderContext+0x8410 supplies this independently of shader program
    * state.  It remains invalid until an exact producer is available. */
   bool agx3_active_context_selector_valid;
   uint32_t agx3_raw_active_context_selector;
   struct d12_resource *rt[AGX_MAX_RENDER_TARGETS];
   enum pipe_format rt_view_format[AGX_MAX_RENDER_TARGETS];
   uint8_t rt_level[AGX_MAX_RENDER_TARGETS];
   uint8_t rt_count;
   uint8_t rt_write_mask;
   bool merge_unused_rts;
   uint32_t bg_usc, eot_usc; /* agx_usc_addr()-encoded */
   uint32_t partial_bg_usc, partial_eot_usc;
   uint64_t g17p_bg_resources, g17p_eot_resources;
   uint64_t g17p_partial_bg_resources, g17p_partial_eot_resources;
   struct agx_counts_packed bg_counts, eot_counts;
   struct agx_counts_packed partial_bg_counts, partial_eot_counts;
   bool has_draws;
   bool process_empty_tiles;

   /* Depth/stencil attachment. Unlike colour, depth never passes through a
    * background/end-of-tile USC program: the fixed-function ZLS unit loads
    * and stores it, driven entirely by the fields below.
    */
   struct d12_resource *ds; /* NULL when the pass has no depth/stencil */
   /* DSV-selected first array slices. Depth and stencil use independent
    * allocations/layouts, so retaining only the resource loses both view
    * offsets when the render command is packed. */
   uint64_t ds_address;
   uint64_t stencil_address;
   uint8_t ds_level;
   uint8_t dsv_flags;       /* D3D12_DSV_FLAGS snapshot for this pass */
   bool depth_cleared, stencil_cleared;
   uint32_t isp_bgobjdepth; /* clear value in the depth plane's encoding */
   uint8_t stencil_clear;

   /*
    * Render-pass tile traffic. Outside a render pass the driver has to infer
    * load and store intent from clear tracking and assume every attachment
    * must be both seeded from memory and written back. Inside one the
    * application states it per attachment, which is where a tiler stops
    * paying for traffic nobody asked for.
    *
    * `render_pass` selects between the two: when false the plans are unset
    * and every path behaves exactly as it did before.
    */
   /* PPP_MULTISAMPLECTL for this render, or zero for the standard pattern.
    * Snapshotted per segment because the positions are fixed for a whole
    * hardware render and the application may change them between passes. */
   uint32_t sample_positions;

   bool render_pass;
   struct d12_render_pass_plan rt_plan[AGX_MAX_RENDER_TARGETS];
   /* Per-attachment clear values. A render pass may clear each target to a
    * different value in one pass, which the single colour a legacy
    * ClearRenderTargetView carries cannot express. */
   float rt_clear[AGX_MAX_RENDER_TARGETS][4];
   struct d12_render_pass_plan depth_plan;
   struct d12_render_pass_plan stencil_plan;

   /* Hardware scissor array (ISP_SCISSOR_BASE). Each draw indexes one
    * entry. This is not optional state: alongside the XY bounds each entry
    * carries the depth CLIP range, so a pass without it rasterizes nothing
    * outside z == 0.
    */
   uint64_t scissor_base;
   unsigned scissor_count;
   uint64_t depth_bias_base;
   unsigned depth_bias_count;
   /* Last viewport/scissor PPP state emitted in this render stream. PPP
    * state is persistent, so repeating an identical update on every draw is
    * pure command/data growth. */
   bool viewport_ppp_valid;
   uint8_t viewport_count;
   uint16_t viewport_scissor_index;
   uint16_t viewport_depth_bias_index;
   uint16_t
   viewport_clip[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE][4];
   float viewport_transform
      [D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE][6];
};

/* Per-render-pass array chunks. Recording rolls to a load-preserving pass
 * before either fills, which bounds storage without limiting command lists. */
#define D12_SCISSORS_PER_SEGMENT 256
#define D12_DEPTH_BIASES_PER_SEGMENT 64

/* A command list may legally contain a large sequence of independent render
 * passes.  Exhaustive descriptor tests use one clear-only pass per array
 * layer, so keep enough inline metadata for those passes while the control
 * and USC storage remain independently bounded. */
#define D12_MAX_SEGMENTS 2048

struct d12_binding_upload_cache {
   bool valid;
   /* Root-table generation this upload was built from; equal generations
    * mean no root parameter was written since, so the hit needs no compare. */
   uint32_t generation;
   const struct d12_root_signature *rs;
   const struct d12_pipeline_state *pso;
   const struct d12_descriptor_heap *resource_heap;
   const struct d12_descriptor_heap *sampler_heap;
   /* The uploaded dense root planes, kept so a root constant or root
    * descriptor change re-roots only the affected slots instead of
    * rebuilding texture and PBE heaps. */
   uint64_t compact[D12_ROOT_TABLE_ENTRIES];
   uint64_t pbe_table;
   struct d12_descriptor table[D12_MAX_RESOURCE_BINDINGS];
   struct d12_descriptor samplers[D12_MAX_RESOURCE_BINDINGS];
   const struct d12_descriptor *sources[D12_MAX_RESOURCE_BINDINGS];
   uint8_t texture_shapes[D12_MAX_RESOURCE_BINDINGS];
   uint64_t root, textures, pbes;
   uint64_t texture_table, texture_table_size_B;
   uint64_t heap, heap_textures, heap_pbes, heap_samplers;
   uint64_t tables;
   uint32_t table_count;
};

#define D12_GFX_UPLOAD_CACHE_SIZE 1024

struct d12_gfx_upload_cache_entry {
   bool valid;
   const struct d12_pipeline_state *pso;
   struct agx_tilebuffer_layout tib;
   uint64_t root, textures, pbes;
   uint64_t heap, heap_textures, heap_pbes, heap_samplers;
   uint64_t draw, blend;
   uint32_t vs_usc, fs_usc, varyings;
};

#define D12_FRAGMENT_STATE_CACHE_SIZE 2048

struct d12_fragment_state_cache_entry {
   bool valid;
   const struct d12_pipeline_state *pso;
   uint32_t fs_usc, varyings;
   enum mesa_prim topology;
   uint8_t dsv_flags;
   uint8_t stencil_ref_front, stencil_ref_back;
   bool has_ds, has_stencil, depth_bias_enable;
   bool occlusion_counting;
   uint64_t gpu;
   uint16_t size;
};

/* Native PPP uploads keep their low VDM alias until the last submitted
 * operation retires, independently of command-list Reset/release. */
struct d12_g17p_vdm_storage {
   uint32_t refcount;
   struct agx_device *dev;
   struct agx_bo *bo;
   uint64_t alias;
   size_t used;
};

struct d12_cmd_list {
   ID3D12GraphicsCommandList iface; /* must be first */
   struct util_dynarray g17p_vdm_storage;
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_COMMAND_LIST_TYPE type;
   enum d12_list_state state;

   struct agx_bo *stream_bo; /* control streams (bottom-up) + data (top-down) */
   struct agx_bo *usc_bo;    /* USC words, AGX_BO_LOW_VA */

   /* Complete GPU allocation ownership for the recorded command stream.
    * Unlike write attachments this spans the whole command list, because the
    * KMD consumes one canonical read vector for each sliced submission. */
   struct d12_read_tracker reads;

   uint8_t *stream_cur;   /* next control word write */
   size_t data_floor;     /* offset of lowest data byte (bumps down) */
   size_t usc_offset;     /* next USC byte (bumps up) */

   struct d12_segment segments[D12_MAX_SEGMENTS];
   unsigned segment_count;
   bool cdm_open;    /* last segment is an unterminated CDM */
   bool render_open; /* last segment is an unterminated RENDER */

   /* THE PRE-GRAPHICS COMPUTE STREAM.
    *
    * Geometry, tessellation, mesh and work-graph stages have no hardware
    * stage on AGX: they run as compute whose output the following DRAW
    * consumes. So those dispatches must be ordered BEFORE a draw whose
    * render pass is already open — and `d12_list_get_cdm` cannot do it,
    * because it resolves the collision by ending the render pass, and a
    * render pass can only be opened by a clear (see d12_list_get_render).
    *
    * Honeykrisp keeps two control streams per command buffer and splices
    * the compute one in ahead of the graphics one (hk_cmd_buffer.h:615).
    * This is that, using a property of our segment array: a segment is
    * just a [stream_start, stream_end) range, and segments are submitted
    * in ARRAY order, so the bytes need not be laid out in that order.
    * The pre-gfx stream is therefore carved out of the data area and its
    * segment inserted before the open render segment.
    *
    * pregfx_seg is only meaningful while pregfx_open.
    */
   uint8_t *pregfx_cur;  /* next control word write, or NULL */
   uint8_t *pregfx_end;  /* one past the region: overflow check */
   unsigned pregfx_seg;  /* index of the inserted segment */
   bool pregfx_open;

   /* Sticky: set when recording hit an error, and NOT cleared by Close.
    * Close must still transition the list to CLOSED so the allocator can
    * reset, but an errored list's command stream is incomplete and must
    * never reach the GPU -- submitting one is how the bench wedged.
    */
   bool had_error;
   /* HRESULT reported by Close for recording-time validation failures.
    * Most internal failures retain E_FAIL; API argument validation can set
    * the more specific code required by D3D12. */
   HRESULT record_error;

   /* Set when a draw in this list bump-allocates from the device
    * tessellation heap. Close then appends a GPU-side write zeroing
    * poly_heap::bottom, so the heap is freed once per EXECUTION rather
    * than once per Reset -- a closed list may be executed repeatedly with
    * no Reset between, and each execution allocates again.
    */
   bool uses_heap;

   /* Set once this recording has been handed to the GPU. Reset must then
    * take fresh storage rather than rewinding what the GPU may still be
    * reading. Cleared when fresh storage is taken.
    */
   bool submitted;

   /*
    * AGX exposes one occlusion counter selector per draw, while D3D12
    * permits overlapping virtual queries.  All active queries therefore
    * share device counter 0 and store Begin/End snapshots in their heaps.
    * The draw only needs to know whether that continuous counter is live.
    */
   unsigned active_occlusion_queries;
   struct d12_query_heap *active_pipeline_stats_heap;
   UINT active_pipeline_stats_index;
   D3D12_QUERY_DATA_PIPELINE_STATISTICS pipeline_stats;

   /* One transform-feedback statistics query may be active per stream.  Poly
    * writes the two low 32-bit counters directly into the 64-bit D3D12 result
    * fields; BeginQuery zeroes the complete result first, so the upper halves
    * remain canonical zeroes.
    */
   struct d12_query_heap *active_so_stats_heap[4];
   UINT active_so_stats_index[4];

   /* Bound state */
   /* IASetVertexBuffers. Recorded per slot; the draw path turns these,
    * plus the PSO's resolved attributes, into the fetch parameter block.
    */
   struct {
      uint64_t addr;
      uint32_t size_B;
      uint32_t stride_B;
   } vb[D12_MAX_VB_SLOTS];

   /* IASetIndexBuffer. size_B of zero means nothing is bound, which is
    * distinct from a bound buffer of length zero only in that the latter
    * cannot happen -- D3D12 rejects it.
    */
   struct {
      uint64_t addr;
      uint32_t size_B;
      enum agx_index_size index_size;
   } ib;

   /* SOSetTargets state. The filled-size address names the low 32 bits of
    * D3D12's 64-bit byte counter; stream-output sizes are bounded below
    * 4 GiB by the API and poly's native offset ABI is uint32_t. */
   D3D12_STREAM_OUTPUT_BUFFER_VIEW so[4];

   enum mesa_prim topology; /* IASetPrimitiveTopology; TRIANGLES by default */
   /* Control points per patch, from a *_CONTROL_POINT_PATCHLIST topology.
    * Zero when the bound topology is not a patch list -- the only thing
    * API state contributes to tessellation.
    */
   unsigned patch_control_points;
   struct d12_cmd_allocator *allocator; /* the one it is recording into */
   struct d12_pipeline_state *pso; /* not referenced; app keeps alive */

   /* SetPredication snapshots the API buffer's 64-bit value into command-list
    * storage at the exact point where the call is recorded. Later draw and
    * dispatch argument patching reads this GPU address, preserving D3D12's
    * latch-before-subsequent-writes semantics. Zero means disabled. */
   uint64_t predicate_gpu;
   D3D12_PREDICATION_OP predicate_op;

   /* Resource addresses indexed by binding slot, one set per pipeline
    * type: D3D12 keeps graphics and compute root bindings completely
    * separate, and sharing one table lets a dispatch inherit a draw's
    * resources.
    */
   /* Whole descriptors, not just addresses: a texture needs its layout
    * and format to pack a hardware descriptor at draw time, which an
    * address alone cannot give.
    */
   struct d12_descriptor compute_root_table[D12_MAX_RESOURCE_BINDINGS];
   struct d12_descriptor graphics_root_table[D12_MAX_RESOURCE_BINDINGS];
   const struct d12_descriptor
      *compute_root_source[D12_MAX_RESOURCE_BINDINGS];
   const struct d12_descriptor
      *graphics_root_source[D12_MAX_RESOURCE_BINDINGS];
   /* Samplers use a separate dense binding namespace. AGX sampler state 0
    * is reserved for texture Load; API sampler S is uploaded at S + 1. */
   struct d12_descriptor compute_sampler_table[D12_MAX_RESOURCE_BINDINGS];
   struct d12_descriptor graphics_sampler_table[D12_MAX_RESOURCE_BINDINGS];
   /* Bumped by every root parameter write (compute, graphics); the binding
    * upload cache compares this instead of 33 KiB of table contents. */
   uint32_t root_generation[2];
   struct util_dynarray descriptor_fixups;
   struct util_dynarray indirect_constant_fixups;
   struct util_dynarray indirect_argument_fixups;
   struct util_dynarray bundle_draws;

   /* ExecuteIndirect reads GPU-produced root/VBV/IBV addresses. A fixed view
    * cell lives in this recording's stream BO; immediately before submission
    * it points at a dense, base-sorted snapshot of all live reserved buffers.
    * The queue operation retains the snapshot BO with the stream. */
   uint64_t sparse_registry_view_gpu;
   struct agx_bo *sparse_registry_bo;
   uint32_t sparse_registry_capacity;
   bool uses_sparse_registry;

   /* Immutable recording-time uploads are shared until their complete key
    * changes.  Exhaustive PSO-mix tests revisit the same 512 pipelines
    * hundreds of times; rebuilding 6 KiB root planes and USC programs per
    * draw is both needless and larger than the command-list arena. */
   struct d12_binding_upload_cache binding_upload_cache[2];
   struct d12_gfx_upload_cache_entry gfx_upload_cache[D12_GFX_UPLOAD_CACHE_SIZE];
   struct d12_fragment_state_cache_entry
      fragment_state_cache[D12_FRAGMENT_STATE_CACHE_SIZE];
   bool direct_draw_params_valid;
   uint32_t direct_draw_params[2];
   uint64_t direct_draw_params_gpu;
   bool blend_upload_valid;
   float blend_upload[4];
   uint64_t blend_upload_gpu;

   /* Explicit state written while recording a bundle.  The final values are
    * inherited back into its caller even when the bundle contains no draw.
    */
   bool bundle_graphics_rs_set;
   bool bundle_topology_set;
   struct d12_bundle_dynamic_state bundle_dynamic_state;

   /* SetDescriptorHeaps state.  GPU descriptor handles remain CPU pointers
    * for application arithmetic, so the bound object is also how an
    * unbounded table resolves that pointer to its GPU-visible BO address. */
   struct d12_descriptor_heap *resource_descriptor_heap;
   struct d12_descriptor_heap *sampler_descriptor_heap;

   /* Zeroed backing for UNBOUND root slots, allocated lazily and reused by
    * every binding upload in this recording. See d12_list_null_buffer().
    */
   uint64_t null_buffer;
   /* Writable sink for robust out-of-view stores.  It must not alias the
    * zero source above: an invalid store followed by an invalid load still
    * has to read zero. */
   uint64_t discard_buffer;
   /* One private zeroed 32-bit counter per binding slot.  Missing UAV
    * counters must remain mapped for robustness without aliasing each other
    * or modifying the null resource's read-as-zero backing.
    */
   uint64_t null_counters;

   /* The bound root signatures, which are what map a root parameter index
    * to a shader register. Not referenced; the app keeps them alive.
    */
   struct d12_root_signature *compute_rs;
   struct d12_root_signature *graphics_rs;
   /* Rasterizer state, actually consumed by the draw path. */
   struct d12_viewport_desc viewports[D12_VIEWPORT_STATE_MAX];
   uint8_t viewport_count;
   struct d12_scissor_desc scissors[D12_VIEWPORT_STATE_MAX];
   uint8_t scissor_count;

   /* OM bindings are descriptor snapshots, not just resource pointers: RTV
    * array slices share a resource but have different addresses, and typeless
    * resources acquire their interpretation from the view.
    */
   struct d12_descriptor bound_rtv[AGX_MAX_RENDER_TARGETS];
   struct d12_descriptor bound_dsv;
   uint8_t bound_rtv_count;
   bool has_bound_dsv;

   /* Render-pass tile traffic for the attachments bound above. Carried on
    * the list because the pass is declared before the first draw opens a
    * hardware segment; d12_list_begin_render copies it across. */
   /* Programmable sample positions. Unset means the standard pattern for the
    * render target's sample count. */
   bool sample_positions_set;
   uint32_t sample_positions;

   bool render_pass_active;
   struct d12_render_pass_plan render_pass_rt_plan[AGX_MAX_RENDER_TARGETS];
   float render_pass_rt_clear[AGX_MAX_RENDER_TARGETS][4];
   struct d12_render_pass_plan render_pass_depth_plan;
   struct d12_render_pass_plan render_pass_stencil_plan;
   float render_pass_depth_clear;
   uint8_t render_pass_stencil_clear;

   /* OMSetStencilRef/OMSetFrontAndBackStencilRef. Separate from the PSO
    * because D3D12 makes the reference values dynamic state while the ops
    * live in the PSO.
    */
   uint8_t stencil_ref_front, stencil_ref_back;

   /* RSSetDepthBias state. It is consumed only by a PSO created with the
    * DYNAMIC_DEPTH_BIAS flag; static PSOs retain their rasterizer values.
    */
   float depth_bias;
   float depth_bias_clamp;
   float slope_scaled_depth_bias;

   /* IASetIndexBufferStripCutValue state. Static PSOs use their creation
    * value; a PSO carrying the dynamic flag consumes this command-list value.
    */
   D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut;

   /* OMSetBlendFactor. AGX fragment uniforms 4..11 are the four 32-bit
    * constants. Reset/ClearState use zero; OMSetBlendFactor(NULL) uses one. */
   float blend_factor[4];
};

static inline struct d12_resource *
d12_resource_from_iface(ID3D12Resource *iface)
{
   return (struct d12_resource *)iface;
}

static inline struct d12_cmd_list *
d12_cmd_list_from_iface(void *iface)
{
   return (struct d12_cmd_list *)iface;
}

struct d12_fence_event_waiter;
struct d12_fence_ticket {
   struct d12_fence_ticket *next;
   struct d12_fence *fence;
   uint64_t value;
   bool registered;
   uint32_t satisfied;
};

struct d12_fence {
   ID3D12Fence iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_FENCE_FLAGS flags;
   uint32_t shared_syncobj;

   /* A D3D12 fence is not a Linux timeline syncobj.  In particular, CPU
    * Signal() may move its completed value backwards, while DRM timeline
    * points are permanently signaled and monotonic.  Keep the public value
    * in userspace and let command-queue workers turn ordered GPU completion
    * into calls to d12_fence_signal_value(). */
   mtx_t lock;
   cnd_t changed;
   uint64_t completed_value;
   bool destroyed;
   struct d12_fence_event_waiter *event_waiters;
   struct d12_fence_ticket *tickets;

   /* Queue operations and asynchronous event waiters must keep the storage
    * alive without changing the public COM reference count. */
   uint32_t lifetime_refcount;
};

struct d12_placement_fence_entry {
   uint64_t current_gpu_address;
   uint64_t monitored_gpu_address;
   volatile uint64_t *current_cpu_address;
   uint32_t flags;
   void *lifetime_cookie;
   void (*release_lifetime)(void *cookie);
};

/* Renderer-owned view of the DDI runtime's fence-value placements.  Queue
 * operations retain this independently of the runtime-owned private object,
 * just as public ID3D12Fence queue operations retain d12_fence. */
struct d12_placement_fence {
   struct d12_device *device;
   mtx_t lock;
   cnd_t changed;
   uint32_t lifetime_refcount;
   uint32_t count;
   struct d12_placement_fence_entry entries[1];
};

struct d12_queue_op;

#define D12_QUEUE_MAX_PLACEMENT_DEPENDENCIES 64

struct d12_queue_placement_wait {
   struct d12_placement_fence *fence;
   uint64_t value;
};

struct d12_queue {
   ID3D12CommandQueue iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;

   struct d12_device *device; /* holds a device ref */
   D3D12_COMMAND_QUEUE_DESC desc;

   /* Hardware queue on the asahi device */
   uint32_t agx_queue_id;

   /* Queue-progress timeline used only for real AGX submissions.  D3D12
    * Wait/Signal operations are ordered by the worker below because Linux
    * syncobjs cannot natively represent wait-before-signal. */
   uint32_t syncobj;
   struct d12_queue_timeline_state timeline;

   mtx_t lock;
   /* DDI callbacks submit on the entering runtime thread. This lock keeps
    * concurrent callbacks in queue order without involving the COM submit
    * worker. */
   mtx_t ddi_submit_lock;
   cnd_t work_available;
   cnd_t retire_available;
   thrd_t worker;
   thrd_t retire_worker;
   struct d12_queue_op *op_head;
   struct d12_queue_op *op_tail;
   struct d12_queue_op *retire_head;
   struct d12_queue_op *retire_tail;
   struct agx_bo *signal_stream_bo;
   struct agx_bo *signal_usc_bo;
   struct agx_bo *signal_program_bo;
   struct agx_bo *dependency_scratch_bo;
   uint32_t signal_control_size;
   /* USC records on AGX2; stable full Ioto program slots on G17P. */
   uint32_t signal_program_stride;
   uint32_t signal_program_size;
   uint64_t signal_slots;
   uint32_t stop;
   uint32_t retire_stop;
   uint32_t pending_placement_wait_count;
   struct d12_queue_placement_wait
      pending_placement_waits[D12_QUEUE_MAX_PLACEMENT_DEPENDENCIES];
   bool ddi_immediate_submit;
   bool worker_started;
   bool retire_worker_started;
};

struct d12_command_signature {
   ID3D12CommandSignature iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;
   struct d12_device *device; /* holds a device ref */
   D3D12_COMMAND_SIGNATURE_DESC desc;
   D3D12_INDIRECT_ARGUMENT_DESC *arguments;
};

/* Direct interface-to-object casts rely on every COM interface being first.
 * Protect all shared frontend objects in every C/C++ translation unit. */
#ifdef __cplusplus
#define D12_ASSERT_IFACE_FIRST(type) static_assert(offsetof(struct type, iface) == 0, #type " COM interface must be first")
#else
#define D12_ASSERT_IFACE_FIRST(type) _Static_assert(offsetof(struct type, iface) == 0, #type " COM interface must be first")
#endif
D12_ASSERT_IFACE_FIRST(d12_device);
D12_ASSERT_IFACE_FIRST(d12_resource);
D12_ASSERT_IFACE_FIRST(d12_heap);
D12_ASSERT_IFACE_FIRST(d12_descriptor_heap);
D12_ASSERT_IFACE_FIRST(d12_query_heap);
D12_ASSERT_IFACE_FIRST(d12_root_signature);
D12_ASSERT_IFACE_FIRST(d12_pipeline_state);
D12_ASSERT_IFACE_FIRST(d12_cmd_allocator);
D12_ASSERT_IFACE_FIRST(d12_cmd_list);
D12_ASSERT_IFACE_FIRST(d12_fence);
D12_ASSERT_IFACE_FIRST(d12_queue);
D12_ASSERT_IFACE_FIRST(d12_command_signature);
#undef D12_ASSERT_IFACE_FIRST

static inline struct d12_device *
d12_device_from_iface(void *iface)
{
   return (struct d12_device *)iface;
}

static inline struct d12_fence *
d12_fence_from_iface(ID3D12Fence *iface)
{
   return (struct d12_fence *)iface;
}

static inline struct d12_queue *
d12_queue_from_iface(ID3D12CommandQueue *iface)
{
   return (struct d12_queue *)iface;
}

/* Object constructors (Create* device methods land here) */
HRESULT d12_fence_create(struct d12_device *device, UINT64 initial_value,
                         D3D12_FENCE_FLAGS flags, REFIID riid, void **out);
void d12_fence_internal_ref(struct d12_fence *fence);
void d12_fence_internal_unref(struct d12_fence *fence);
HRESULT d12_fence_signal_value(struct d12_fence *fence, uint64_t value);
bool d12_fence_wait_value(struct d12_fence *fence, uint64_t value,
                          const uint32_t *cancel);
void d12_fence_ticket_init(struct d12_fence *fence,
                           struct d12_fence_ticket *ticket, uint64_t value);
void d12_fence_ticket_fini(struct d12_fence_ticket *ticket);
bool d12_fence_ticket_wait(struct d12_fence_ticket *ticket,
                           const uint32_t *cancel);
HRESULT d12_fence_set_event_multiple(
   struct d12_device *device, ID3D12Fence *const *fences,
   const UINT64 *values, UINT count,
   D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event);
HRESULT d12_fence_create_shared_handle(
   struct d12_device *device, ID3D12DeviceChild *object,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle);
HRESULT d12_fence_open_shared_handle(struct d12_device *device, HANDLE handle,
                                     REFIID riid, void **out);
HRESULT d12_placement_fence_create(
   struct d12_device *device, uint32_t count,
   const struct d12_placement_fence_entry *entries,
   struct d12_placement_fence **out_fence);
void d12_placement_fence_destroy(struct d12_placement_fence *fence);
void d12_placement_fence_ref(struct d12_placement_fence *fence);
void d12_placement_fence_unref(struct d12_placement_fence *fence);
bool d12_placement_fence_wait(struct d12_placement_fence *fence,
                              uint64_t value, const uint32_t *cancel);
HRESULT d12_queue_signal_placement_fence(
   struct d12_queue *queue, struct d12_placement_fence *fence,
   uint64_t value);
HRESULT d12_queue_wait_placement_fence(
   struct d12_queue *queue, struct d12_placement_fence *fence,
   uint64_t value);
HRESULT d12_queue_create(struct d12_device *device,
                         const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid,
                         void **out);
HRESULT d12_queue_create_ddi(struct d12_device *device,
                             const D3D12_COMMAND_QUEUE_DESC *desc,
                             REFIID riid, void **out);
HRESULT d12_resource_create_committed(struct d12_device *device,
                                      const D3D12_HEAP_PROPERTIES *heap_props,
                                      D3D12_HEAP_FLAGS heap_flags,
                                      const D3D12_RESOURCE_DESC *desc,
                                      D3D12_RESOURCE_STATES initial_state,
                                      const D3D12_CLEAR_VALUE *clear_value,
                                      REFIID riid, void **out);
HRESULT d12_cmd_allocator_create(struct d12_device *device,
                                 D3D12_COMMAND_LIST_TYPE type, REFIID riid,
                                 void **out);
HRESULT d12_cmd_list_create(struct d12_device *device,
                            D3D12_COMMAND_LIST_TYPE type,
                            struct d12_cmd_allocator *allocator,
                            struct d12_pipeline_state *initial_state,
                            REFIID riid, void **out);
HRESULT d12_cmd_present_blt(
   struct d12_cmd_list *list, struct d12_resource *destination,
   struct d12_resource *source,
   const struct d12_frontend_present_blt_desc *desc);

static inline struct d12_cmd_allocator *
d12_cmd_allocator_from_iface(ID3D12CommandAllocator *iface)
{
   return (struct d12_cmd_allocator *)iface;
}
HRESULT d12_descriptor_heap_create(struct d12_device *device,
                                   const D3D12_DESCRIPTOR_HEAP_DESC *desc,
                            REFIID riid, void **out);
void d12_descriptor_heap_update_slot(struct d12_device *device,
                                     struct d12_descriptor *slot);
void d12_device_create_srv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_SHADER_RESOURCE_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_create_uav_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   ID3D12Resource *counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_create_rtv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_RENDER_TARGET_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_create_dsv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_DEPTH_STENCIL_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_create_cbv_runtime(
   struct d12_device *device, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_create_sampler_runtime(
   struct d12_device *device, const D3D12_SAMPLER_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination);
void d12_device_copy_descriptors_simple_runtime(
   struct d12_device *device, UINT count,
   D3D12_CPU_DESCRIPTOR_HANDLE destination,
   D3D12_CPU_DESCRIPTOR_HANDLE source,
   D3D12_DESCRIPTOR_HEAP_TYPE type);
bool d12_cmd_list_refresh_descriptors(struct d12_cmd_list *list);
bool d12_cmd_list_refresh_sparse_registry(struct d12_cmd_list *list);
bool d12_descriptor_resource_span(const struct d12_descriptor *descriptor,
                                  uint64_t *base, uint64_t *size);
/* Render passes. The bindings are recorded exactly as OMSetRenderTargets
 * records them; only the per-attachment tile traffic differs. */
bool d12_cmd_list_begin_render_pass(
   struct d12_cmd_list *list, unsigned rt_count,
   const struct d12_rtv *const *rtvs,
   const struct d12_render_pass_plan *rt_plans,
   const float (*rt_clears)[4], const struct d12_rtv *dsv,
   const struct d12_render_pass_plan *depth_plan,
   const struct d12_render_pass_plan *stencil_plan, float depth_clear,
   uint8_t stencil_clear);
bool d12_cmd_list_end_render_pass(struct d12_cmd_list *list);

/* Programmable sample positions, tier 1: one pattern for the whole render
 * target. A zero count and zero pixels restores the standard pattern. */
bool d12_cmd_list_set_sample_positions(
   struct d12_cmd_list *list, uint32_t samples_per_pixel,
   uint32_t pixel_count, const struct d12_sample_position *positions);

void d12_pack_sampler_descriptor(struct d12_device *device,
                                 struct d12_descriptor *slot,
                                 const D3D12_SAMPLER_DESC2 *desc);
bool d12_set_null_texture_descriptor(struct d12_device *device, void *out);
bool d12_set_null_pbe_descriptor(struct d12_device *device, void *out);
bool d12_refresh_texture_descriptor(struct d12_device *device, void *out,
                                    const struct d12_descriptor *d);
bool d12_refresh_pbe_descriptor(struct d12_device *device, void *out,
                                const struct d12_descriptor *d);
/* DXGI format support (d12_format.c) */
enum pipe_format d12_format_to_pipe(DXGI_FORMAT format);
bool d12_format_is_renderable(DXGI_FORMAT format);
unsigned d12_format_bytes_per_pixel(DXGI_FORMAT format);
uint64_t d12_format_frontend_support(DXGI_FORMAT format);
/* False when a format D3D12 requires a typed UAV load for does not report
 * one; `missing` then names the first such format. */
bool d12_format_typed_uav_load_complete(bool additional, DXGI_FORMAT *missing);

/* Depth/stencil formats. AGX keeps stencil in a separate plane from depth,
 * so these are asked for independently rather than as one pipe format.
 */
bool d12_format_is_depth_stencil(DXGI_FORMAT format);
enum pipe_format d12_format_depth_plane(DXGI_FORMAT format);
enum pipe_format d12_format_stencil_plane(DXGI_FORMAT format);
bool d12_format_depth_is_substituted(DXGI_FORMAT format);

HRESULT d12_root_signature_create(struct d12_device *device,
                                  const void *blob, size_t blob_size,
                                  REFIID riid, void **out);
HRESULT d12_pipeline_state_create_compute(struct d12_device *device,
                                          const void *dxil, size_t dxil_size,
                                          const struct d12_binding_map *map,
                                          REFIID riid, void **out);
void d12_pipeline_state_library_ref(struct d12_pipeline_state *pso);
void d12_pipeline_state_library_unref(struct d12_pipeline_state *pso);
HRESULT d12_pipeline_state_create_mesh(
   struct d12_device *device, const void *ms_dxil, size_t ms_size,
   const void *as_dxil, size_t as_size, const void *ps_dxil, size_t ps_size,
   enum pipe_format rt_format, unsigned nr_rt, const DXGI_FORMAT *rt_formats,
   unsigned nr_samples, UINT sample_mask,
   const D3D12_DEPTH_STENCIL_DESC *ds_desc, DXGI_FORMAT dsv_format,
   const D3D12_RASTERIZER_DESC *rs_desc,
   const D3D12_BLEND_DESC *blend_desc,
   const struct d12_binding_map *map, REFIID riid, void **out);

HRESULT d12_pipeline_state_create_graphics(
   struct d12_device *device, const void *vs_dxil, size_t vs_size,
   const void *ps_dxil, size_t ps_size, const void *gs_dxil, size_t gs_size,
   const void *hs_dxil, size_t hs_size, const void *ds_dxil, size_t ds_size,
   enum pipe_format rt_format, unsigned nr_rt, const DXGI_FORMAT *rt_formats,
   unsigned nr_samples, UINT sample_mask,
   const D3D12_DEPTH_STENCIL_DESC *ds_desc, DXGI_FORMAT dsv_format,
   const D3D12_RASTERIZER_DESC *rs_desc, float rasterizer_depth_bias,
   bool front_stencil_enable, bool back_stencil_enable,
   uint8_t front_stencil_read_mask, uint8_t front_stencil_write_mask,
   uint8_t back_stencil_read_mask, uint8_t back_stencil_write_mask,
   const D3D12_BLEND_DESC *blend_desc,
   const D3D12_INPUT_LAYOUT_DESC *input_layout,
   const D3D12_STREAM_OUTPUT_DESC *so_desc,
   D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology_type,
   D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut,
   D3D12_PIPELINE_STATE_FLAGS flags,
   const struct d12_binding_map *map, REFIID riid, void **out);

void d12_device_add_ref(struct d12_device *device);
void d12_device_release(struct d12_device *device);
HRESULT d12_device_backend_init(struct d12_device *device,
                                D3D_FEATURE_LEVEL feature_level);
void d12_device_backend_cleanup(struct d12_device *device);

#ifndef _WIN32
bool d12_platform_is_real_asahi_render_fd(int fd);
#endif
bool d12_platform_open_device(struct d12_device *device, IUnknown *adapter);
void d12_platform_close_device(struct d12_device *device);
void d12_platform_signal_event(HANDLE event);
HRESULT d12_platform_create_shared_bo(
   struct d12_device *device, size_t size, unsigned align,
   enum agx_bo_flags flags, const void *runtime_data,
   uint32_t runtime_data_size, const char *label, struct agx_bo **bo);
HRESULT d12_platform_import_shared_bo(
   struct d12_device *device, HANDLE handle, void *runtime_data,
   uint32_t *runtime_data_size, const char *label, struct agx_bo **bo);
HRESULT d12_platform_export_shared_bo(
   struct d12_device *device, struct agx_bo *bo,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle);
HRESULT d12_platform_create_shared_sync(struct d12_device *device,
                                        uint64_t initial_value,
                                        uint32_t *syncobj);
HRESULT d12_platform_import_shared_sync(struct d12_device *device,
                                        HANDLE handle, uint32_t *syncobj);
HRESULT d12_platform_export_shared_sync(
   struct d12_device *device, uint32_t syncobj,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle);
HRESULT d12_platform_query_shared_sync(struct d12_device *device,
                                       uint32_t syncobj, uint64_t *value);
HRESULT d12_platform_set_bo_resident(struct d12_device *device,
                                     struct agx_bo *bo, bool resident);
HRESULT d12_platform_open_shared_handle_by_name(struct d12_device *device,
                                                LPCWSTR name, DWORD access,
                                                HANDLE *handle);

void d12_g17p_vdm_storage_ref(struct d12_g17p_vdm_storage *storage);
void d12_g17p_vdm_storage_unref(struct d12_g17p_vdm_storage *storage);
void d12_list_g17p_vdm_storage_reset(struct d12_cmd_list *list);
void *d12_list_alloc_g17p_vdm(struct d12_cmd_list *list, size_t size,
                            size_t alignment, uint64_t *compact_va);

uint32_t d12_build_g17p_bg_eot(
   struct d12_cmd_list *list, const struct d12_segment *seg,
   const float legacy_color[4],
   const enum agx_bg_eot_op ops[AGX_MAX_RENDER_TARGETS],
   const struct agx_tilebuffer_layout *tib,
   uint64_t *out_resources);

/* Bounded D32 qualification: registered shim metadata or separately opted-in
 * verified physical Asahi render fd. Every segment shape check is mandatory. */
bool d12_g17p_depth_enabled(const struct agx_device *dev);
bool d12_g17p_depth_segment_supported(const struct d12_segment *seg);

/* Native Texture.Load metadata producer never admits a physical DRM fd. */
bool d12_g17p_txf_metadata_enabled(const struct agx_device *dev);
bool d12_g17p_txf_physical_enabled(const struct agx_device *dev);
