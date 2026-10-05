/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12Resource storage and CPU access. G17P UMA is cached and requires
 * explicit buffer and opaque texture cache maintenance.
 */

#include "d12_private.h"
#include "d12_core_caps.h"
#include "d12_heap_desc.h"
#include "d12_sparse_residency_core.h"
#include "util/os_misc.h"
#include "util/cache_ops.h"

static_assert(AIL_PAGESIZE == D12_AGX_SPARSE_PAGE_SIZE_B,
              "portable sparse page size must match AIL");
static_assert(AIL_PAGES_PER_FOLIO == D12_AGX_SPARSE_PAGES_PER_FOLIO,
              "portable sparse folio width must match AIL");
static_assert(AIL_FOLIO_SIZE_EL == D12_AGX_SPARSE_ELEMENTS_PER_FOLIO,
              "portable sparse folio stride must match AIL");
static_assert(AIL_SPARSE_ELSIZE_B == D12_AGX_SPARSE_ELEMENT_SIZE_B,
              "portable sparse entry width must match AIL");
static_assert(D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES ==
                 D12_SPARSE_BUFFER_TILE_SIZE_B,
              "portable sparse buffer tile size must match D3D12");

/* Runtime-private metadata used by the native Windows D3D runtimes and by
 * vkd3d-proton. Keeping this wire layout compatible is what lets a D3D11
 * DXGI presentation resource and this native D3D12 driver open the same
 * D3DKMT resource without a copy or a second WDDM allocation. */
struct d12_d3dkmt_dxgi_desc {
   uint32_t size;
   uint32_t version;
   uint32_t width;
   uint32_t height;
   DXGI_FORMAT format;
   uint32_t unknown_0;
   uint32_t unknown_1;
   uint32_t keyed_mutex;
   uint32_t mutex_handle;
   uint32_t sync_handle;
   uint32_t nt_shared;
   uint32_t unknown_2;
   uint32_t unknown_3;
   uint32_t unknown_4;
};

struct d12_d3dkmt_texture2d_desc {
   uint32_t width;
   uint32_t height;
   uint32_t mip_levels;
   uint32_t array_size;
   DXGI_FORMAT format;
   DXGI_SAMPLE_DESC sample_desc;
   uint32_t usage;
   uint32_t bind_flags;
   uint32_t cpu_access_flags;
   uint32_t misc_flags;
};

struct d12_d3dkmt_d3d11_desc {
   struct d12_d3dkmt_dxgi_desc dxgi;
   uint32_t dimension;
   union {
      struct d12_d3dkmt_texture2d_desc texture2d;
      uint8_t padding[44];
   };
};

struct d12_d3dkmt_d3d12_desc {
   struct d12_d3dkmt_d3d11_desc d3d11;
   uint32_t unknown_5[4];
   uint32_t resource_size;
   uint32_t unknown_6[7];
   uint32_t resource_align;
   uint32_t unknown_7[9];
   union {
      D3D12_RESOURCE_DESC1 desc1;
      uint32_t padding[16];
   };
   uint64_t unknown_8;
};

static_assert(sizeof(struct d12_d3dkmt_dxgi_desc) == 0x38,
              "D3DKMT DXGI descriptor ABI");
static_assert(sizeof(struct d12_d3dkmt_d3d11_desc) == 0x68,
              "D3DKMT D3D11 descriptor ABI");
static_assert(sizeof(D3D12_RESOURCE_DESC1) == 0x40,
              "D3D12_RESOURCE_DESC1 ABI");
static_assert(sizeof(struct d12_d3dkmt_d3d12_desc) == 0x108,
              "D3DKMT D3D12 descriptor ABI");

/* DXVK's D3D11 OpenSharedResource1 path recognizes the version-4 prefix by
 * these exact offsets, even when the runtime returns the complete 0x108-byte
 * D3D12 blob. Keep more than the aggregate sizes pinned so an otherwise
 * innocuous field or alignment change cannot silently break zero-copy DXGI
 * presentation. */
static_assert(offsetof(struct d12_d3dkmt_dxgi_desc, version) == 0x04,
              "D3DKMT DXGI version offset");
static_assert(offsetof(struct d12_d3dkmt_dxgi_desc, keyed_mutex) == 0x1c,
              "D3DKMT DXGI keyed-mutex offset");
static_assert(offsetof(struct d12_d3dkmt_dxgi_desc, nt_shared) == 0x28,
              "D3DKMT DXGI NT-sharing offset");
static_assert(offsetof(struct d12_d3dkmt_d3d11_desc, dimension) == 0x38,
              "D3DKMT D3D11 dimension offset");
static_assert(offsetof(struct d12_d3dkmt_d3d11_desc, texture2d) == 0x3c,
              "D3DKMT D3D11 texture descriptor offset");
static_assert(offsetof(struct d12_d3dkmt_d3d11_desc,
                       texture2d.misc_flags) == 0x64,
              "D3DKMT D3D11 texture flags offset");
static_assert(offsetof(struct d12_d3dkmt_d3d12_desc, resource_size) == 0x78,
              "D3DKMT D3D12 resource size offset");
static_assert(offsetof(struct d12_d3dkmt_d3d12_desc, resource_align) == 0x98,
              "D3DKMT D3D12 resource alignment offset");
static_assert(offsetof(struct d12_d3dkmt_d3d12_desc, desc1) == 0xc0,
              "D3DKMT D3D12 resource descriptor offset");

enum {
   D12_D3D11_RESOURCE_DIMENSION_TEXTURE2D = 3,
   D12_D3D11_BIND_SHADER_RESOURCE = 0x8,
   D12_D3D11_BIND_RENDER_TARGET = 0x20,
   D12_D3D11_BIND_DEPTH_STENCIL = 0x40,
   D12_D3D11_BIND_UNORDERED_ACCESS = 0x80,
   D12_D3D11_RESOURCE_MISC_SHARED = 0x2,
   D12_D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800,
   D12_D3DKMT_HEAP_MAGIC = 0x48584741, /* "AGXH" */
};

struct d12_d3dkmt_heap_desc {
   uint32_t magic;
   uint32_t version;
   uint32_t struct_size;
   uint32_t reserved;
   uint64_t size;
   uint64_t alignment;
   uint32_t type;
   uint32_t cpu_page_property;
   uint32_t memory_pool;
   uint32_t creation_node_mask;
   uint32_t visible_node_mask;
   uint32_t flags;
};

static void
d12_shared_heap_metadata_init(struct d12_d3dkmt_heap_desc *metadata,
                              const D3D12_HEAP_DESC *desc)
{
   *metadata = (struct d12_d3dkmt_heap_desc){
      .magic = D12_D3DKMT_HEAP_MAGIC,
      .version = 1,
      .struct_size = sizeof(*metadata),
      .size = desc->SizeInBytes,
      .alignment = desc->Alignment,
      .type = desc->Properties.Type,
      .cpu_page_property = desc->Properties.CPUPageProperty,
      .memory_pool = desc->Properties.MemoryPoolPreference,
      .creation_node_mask = desc->Properties.CreationNodeMask,
      .visible_node_mask = desc->Properties.VisibleNodeMask,
      .flags = desc->Flags,
   };
}

static bool
d12_shared_heap_metadata_parse(const void *data, uint32_t size,
                               D3D12_HEAP_DESC *desc)
{
   if (!data || !desc || size != sizeof(struct d12_d3dkmt_heap_desc))
      return false;
   const struct d12_d3dkmt_heap_desc *metadata = data;
   if (metadata->magic != D12_D3DKMT_HEAP_MAGIC || metadata->version != 1 ||
       metadata->struct_size != sizeof(*metadata))
      return false;

   *desc = (D3D12_HEAP_DESC){
      .SizeInBytes = metadata->size,
      .Properties =
         {
            .Type = metadata->type,
            .CPUPageProperty = metadata->cpu_page_property,
            .MemoryPoolPreference = metadata->memory_pool,
            .CreationNodeMask = metadata->creation_node_mask,
            .VisibleNodeMask = metadata->visible_node_mask,
         },
      .Alignment = metadata->alignment,
      .Flags = metadata->flags,
   };
   return true;
}

static void
d12_shared_metadata_init(struct d12_d3dkmt_d3d12_desc *metadata,
                         const D3D12_RESOURCE_DESC *desc, uint64_t size_B)
{
   memset(metadata, 0, sizeof(*metadata));
   metadata->d3d11.dxgi.size = sizeof(metadata->d3d11);
   metadata->d3d11.dxgi.nt_shared = 1;
   metadata->resource_size = MIN2(size_B, UINT32_MAX);
   metadata->resource_align = MIN2(desc->Alignment, UINT32_MAX);

   metadata->desc1 = (D3D12_RESOURCE_DESC1){
      .Dimension = desc->Dimension,
      .Alignment = desc->Alignment,
      .Width = desc->Width,
      .Height = desc->Height,
      .DepthOrArraySize = desc->DepthOrArraySize,
      .MipLevels = desc->MipLevels,
      .Format = desc->Format,
      .SampleDesc = desc->SampleDesc,
      .Layout = desc->Layout,
      .Flags = desc->Flags,
   };

   if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
      return;

   struct d12_d3dkmt_d3d11_desc *d3d11 = &metadata->d3d11;
   d3d11->dxgi.width = MIN2(desc->Width, UINT32_MAX);
   d3d11->dxgi.height = desc->Height;
   d3d11->dxgi.format = desc->Format;
   d3d11->dimension = D12_D3D11_RESOURCE_DIMENSION_TEXTURE2D;
   d3d11->texture2d = (struct d12_d3dkmt_texture2d_desc){
      .width = MIN2(desc->Width, UINT32_MAX),
      .height = desc->Height,
      .mip_levels = desc->MipLevels,
      .array_size = desc->DepthOrArraySize,
      .format = desc->Format,
      .sample_desc = desc->SampleDesc,
      .usage = 0, /* D3D11_USAGE_DEFAULT */
      .bind_flags = D12_D3D11_BIND_SHADER_RESOURCE,
      .misc_flags = D12_D3D11_RESOURCE_MISC_SHARED |
                    D12_D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
   };

   if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
      d3d11->texture2d.bind_flags |= D12_D3D11_BIND_RENDER_TARGET;
   if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
      d3d11->texture2d.bind_flags |= D12_D3D11_BIND_DEPTH_STENCIL;
   if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
      d3d11->texture2d.bind_flags |= D12_D3D11_BIND_UNORDERED_ACCESS;
   if (desc->Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
      d3d11->texture2d.bind_flags &= ~D12_D3D11_BIND_SHADER_RESOURCE;

   /* Version 4 exposes the D3D11 prefix to DXGI/D3D11 consumers. The full
    * 0x108-byte blob still carries desc1 for D3D12 consumers. */
   if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
      d3d11->dxgi.version = 4;
}

static bool
d12_shared_metadata_parse(const void *data, uint32_t size,
                          D3D12_RESOURCE_DESC *desc)
{
   const struct d12_d3dkmt_dxgi_desc *dxgi = data;
   if (!data || !desc || size < sizeof(*dxgi))
      return false;

   if (size == sizeof(struct d12_d3dkmt_d3d12_desc) &&
       dxgi->size == sizeof(struct d12_d3dkmt_d3d11_desc) &&
       (dxgi->version == 0 || dxgi->version == 4)) {
      const struct d12_d3dkmt_d3d12_desc *metadata = data;
      *desc = (D3D12_RESOURCE_DESC){
         .Dimension = metadata->desc1.Dimension,
         .Alignment = metadata->desc1.Alignment,
         .Width = metadata->desc1.Width,
         .Height = metadata->desc1.Height,
         .DepthOrArraySize = metadata->desc1.DepthOrArraySize,
         .MipLevels = metadata->desc1.MipLevels,
         .Format = metadata->desc1.Format,
         .SampleDesc = metadata->desc1.SampleDesc,
         .Layout = metadata->desc1.Layout,
         .Flags = metadata->desc1.Flags,
      };
      return true;
   }

   if (size == sizeof(struct d12_d3dkmt_d3d11_desc) &&
       dxgi->size == sizeof(struct d12_d3dkmt_d3d11_desc) &&
       dxgi->version == 4) {
      const struct d12_d3dkmt_d3d11_desc *metadata = data;
      if (metadata->dimension != D12_D3D11_RESOURCE_DIMENSION_TEXTURE2D)
         return false;

      const struct d12_d3dkmt_texture2d_desc *texture =
         &metadata->texture2d;
      *desc = (D3D12_RESOURCE_DESC){
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = texture->width,
         .Height = texture->height,
         .DepthOrArraySize = texture->array_size,
         .MipLevels = texture->mip_levels,
         .Format = texture->format,
         .SampleDesc = texture->sample_desc,
         .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
         .Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
      };
      if (texture->bind_flags & D12_D3D11_BIND_RENDER_TARGET)
         desc->Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      if (texture->bind_flags & D12_D3D11_BIND_DEPTH_STENCIL)
         desc->Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      if (texture->bind_flags & D12_D3D11_BIND_UNORDERED_ACCESS)
         desc->Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      if ((texture->bind_flags & D12_D3D11_BIND_DEPTH_STENCIL) &&
          !(texture->bind_flags & D12_D3D11_BIND_SHADER_RESOURCE))
         desc->Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
      return true;
   }

   return false;
}

static ULONG STDMETHODCALLTYPE
d12_resource_AddRef(ID3D12Resource *iface);

static const ID3D12ResourceVtbl d12_resource_vtbl;
static const ID3D12HeapVtbl d12_heap_vtbl;

static HRESULT STDMETHODCALLTYPE
d12_resource_QueryInterface(ID3D12Resource *iface, REFIID riid, void **out)
{
   struct d12_resource *res = d12_resource_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12Resource)) {
      p_atomic_inc(&res->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&res->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12Resource::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_resource_AddRef(ID3D12Resource *iface)
{
   struct d12_resource *res = d12_resource_from_iface(iface);
   return p_atomic_inc_return(&res->refcount);
}

static void
d12_resource_cleanup_sparse_storage(struct d12_resource *res)
{
   for (uint32_t i = 0; i < res->sparse_tile_count; i++) {
      if (res->sparse_mappings && res->sparse_mappings[i].bo)
         agx_bo_unreference(&res->device->dev,
                            res->sparse_mappings[i].bo);
   }
   if (res->sparse_map)
      agx_bo_unreference(&res->device->dev, res->sparse_map);
   free(res->sparse_mappings);
   free(res->sparse_tilings);
   simple_mtx_destroy(&res->sparse_lock);
}

static ULONG STDMETHODCALLTYPE
d12_resource_Release(ID3D12Resource *iface)
{
   struct d12_resource *res = d12_resource_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&res->refcount);

   if (refs == 0) {
      if (res->reserved) {
         simple_mtx_lock(&res->device->sparse_resource_lock);
         list_del(&res->sparse_link);
         simple_mtx_unlock(&res->device->sparse_resource_lock);
         d12_resource_cleanup_sparse_storage(res);
      }

      if (res->va)
         agx_va_free(&res->device->dev, res->va, true);

      /* A PLACED resource borrows the heap's BO -- unreferencing it here
       * would free storage the heap still owns and other placed resources
       * are still using. Drop the heap reference instead. */
      if (res->placed) {
         /* Drop the BO reference taken at creation; the heap object's own
          * refcount was never touched. */
         agx_bo_unreference(&res->device->dev, res->bo);
      } else if (!res->reserved) {
         agx_bo_unreference(&res->device->dev, res->bo);
         if (res->stencil_bo && res->stencil_bo != res->bo)
            agx_bo_unreference(&res->device->dev, res->stencil_bo);
      }

      d12_private_data_store_destroy(&res->private_data);
      d12_device_release(res->device);
      free(res);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_resource, ID3D12Resource,
                                d12_resource_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_resource_GetDevice(ID3D12Resource *iface, REFIID riid, void **out)
{
   struct d12_resource *res = d12_resource_from_iface(iface);
   return ID3D12Device_QueryInterface(&res->device->iface, riid, out);
}

/* G17P CPU-visible buffers are cached but CacheCoherentUMA is false. Keep
 * cache operations within the resource's BO window, including placed offsets.
 * Callers still synchronize GPU access with a fence before a reading Map. */
static bool
d12_resource_buffer_cache_range(const struct d12_resource *res,
                                const D3D12_RANGE *range,
                                size_t *begin, size_t *size)
{
   uint64_t width = res->desc.Width;
   if (!res->bo || width > SIZE_MAX || res->offset_B > res->bo->size ||
       width > res->bo->size - res->offset_B)
      return false;

   uint64_t first = range ? range->Begin : 0;
   uint64_t last = range ? range->End : width;
   if (first > width || last > width)
      return false;
   *begin = first;
   *size = first < last ? last - first : 0;
   if (*size) {
      /* The platform helper rounds to cache lines. Its maximum granule
       * must remain inside the owned BO, even for an unaligned range. */
      size_t granule = util_cache_granularity();
      if (!granule || (granule & (granule - 1)))
         return false;
      uint64_t final_line = (res->offset_B + last - 1) &
                            ~((uint64_t)granule - 1);
      if (final_line > res->bo->size ||
          granule > res->bo->size - final_line)
         return false;
   }
   return true;
}

static bool
d12_resource_needs_buffer_cache_ops(const struct d12_resource *res)
{
#ifndef _WIN32
   return res->device->dev.chip == AGX_CHIP_G17P &&
          res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
#else
   return false;
#endif
}

static HRESULT STDMETHODCALLTYPE
d12_resource_Map(ID3D12Resource *iface, UINT subresource,
                 const D3D12_RANGE *read_range, void **data)
{
   struct d12_resource *res = d12_resource_from_iface(iface);

   if (subresource != 0)
      return E_INVALIDARG;

   if (res->reserved)
      return E_INVALIDARG;

   /* DEFAULT heaps have no application-visible CPU mapping even on UMA.
    * For an UNKNOWN-layout texture on a CPU-visible custom heap, Map(NULL)
    * is still legal (it establishes the mapping without exposing the VA),
    * but returning a pointer would falsely describe the opaque layout as a
    * linear byte array.  Rejections must leave the caller's sentinel alone. */
   /* GPU_UPLOAD is the one non-staging heap type the CPU may write: that is
    * the entire point of it. */
   if ((res->heap_props.Type == D3D12_HEAP_TYPE_DEFAULT) ||
       res->heap_props.CPUPageProperty ==
          D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE ||
       (data && res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER &&
        res->desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN))
      return E_INVALIDARG;

   /* Persistent mapping; a noncoherent G17P buffer reads through explicit
    * cache maintenance of the caller's read range.
    * offset_B is nonzero for a PLACED resource, which is a window into a
    * heap -- mapping the BO base would hand back the start of the heap.
    *
    * agx_bo_map CAN fail: on WDDM it is a Lock2 callback, which dxgkrnl
    * refuses under allocation pressure and after device loss.  Doing pointer
    * arithmetic on the NULL result and returning S_OK hands the application
    * a near-null pointer it will fault on, with the fault landing in the app
    * rather than here.  Report the failure instead. */
   void *mapping = agx_bo_map(res->bo);
   if (!mapping)
      return E_OUTOFMEMORY;

   if (d12_resource_needs_buffer_cache_ops(res)) {
      size_t begin, size;
      if (!util_has_cache_ops())
         return E_FAIL;
      if (!d12_resource_buffer_cache_range(res, read_range, &begin, &size))
         return E_INVALIDARG;
      if (size)
         util_flush_inval_range((uint8_t *)mapping + res->offset_B + begin,
                               size);
   }

   if (data)
      *data = (uint8_t *)mapping + res->offset_B;
   return S_OK;
}

static void STDMETHODCALLTYPE
d12_resource_Unmap(ID3D12Resource *iface, UINT subresource,
                   const D3D12_RANGE *written_range)
{
   struct d12_resource *res = d12_resource_from_iface(iface);
   if (subresource || res->reserved ||
       res->heap_props.Type == D3D12_HEAP_TYPE_DEFAULT ||
       res->heap_props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE ||
       !d12_resource_needs_buffer_cache_ops(res) || !util_has_cache_ops())
      return;

   size_t begin, size;
   if (!d12_resource_buffer_cache_range(res, written_range, &begin, &size) ||
       !size)
      return;
   void *mapping = agx_bo_map(res->bo);
   if (mapping)
      util_flush_range((uint8_t *)mapping + res->offset_B + begin, size);
}

static D12_AGGREGATE_RETURN(D3D12_RESOURCE_DESC) STDMETHODCALLTYPE
d12_resource_GetDesc(ID3D12Resource *iface
                     D12_AGGREGATE_OUT(D3D12_RESOURCE_DESC))
{
   struct d12_resource *res = d12_resource_from_iface(iface);
   D12_RETURN_AGGREGATE(res->desc);
}

static D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE
d12_resource_GetGPUVirtualAddress(ID3D12Resource *iface)
{
   struct d12_resource *res = d12_resource_from_iface(iface);

   /* Per the D3D12 contract this is only valid for buffers. */
   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
      return 0;

   return res->addr;
}

struct d12_cpu_subresource {
   uint8_t *map;
   uint32_t x_el;
   uint32_t y_el;
   uint32_t width_el;
   uint32_t height_el;
   uint32_t blocksize_B;
   uint32_t texture_pitch_B;
   bool empty;
   bool tiled;
};

static HRESULT
d12_resource_cpu_subresource(struct d12_resource *res, UINT subresource,
                             const D3D12_BOX *box,
                             struct d12_cpu_subresource *out)
{
   memset(out, 0, sizeof(*out));

   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       res->heap_props.Type != D3D12_HEAP_TYPE_CUSTOM ||
       res->heap_props.CPUPageProperty ==
          D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE ||
       subresource != 0)
      return E_INVALIDARG;

   const enum pipe_format format = res->layout.format;
   const uint32_t block_width = util_format_get_blockwidth(format);
   const uint32_t block_height = util_format_get_blockheight(format);
   const uint32_t blocksize = util_format_get_blocksize(format);
   const uint32_t width = res->layout.width_px;
   const uint32_t height = res->layout.height_px;
   const D3D12_BOX whole = {0, 0, 0, width, height, 1};
   const D3D12_BOX *b = box ? box : &whole;

   /* Coordinates must still name the resource even when the box is empty:
    * {width,height,depth,width,height,depth} is a valid no-op, while an empty
    * box at depth + 1 is invalid. This distinction is covered explicitly by
    * the conformance test.
    */
   if (b->left > width || b->right > width || b->top > height ||
       b->bottom > height || b->front > 1 || b->back > 1)
      return E_INVALIDARG;

   if (b->left >= b->right || b->top >= b->bottom || b->front >= b->back) {
      out->empty = true;
      return S_OK;
   }

   if (b->front != 0 || b->back != 1)
      return E_INVALIDARG;

   /* BC regions are expressed in pixels but copied in whole format blocks.
    * The right/bottom edge may be the image edge; interior coordinates must
    * be block aligned.
    */
   if (b->left % block_width || b->top % block_height ||
       (b->right % block_width && b->right != width) ||
       (b->bottom % block_height && b->bottom != height))
      return E_INVALIDARG;

   const bool tiled = res->layout.tiling != AIL_TILING_LINEAR;
   const enum agx_chip chip = res->device->dev.chip;
   const bool cpu_tiling_supported = chip == AGX_CHIP_G17P ||
      chip == AGX_CHIP_G13G || chip == AGX_CHIP_G13X ||
      chip == AGX_CHIP_G14G || chip == AGX_CHIP_G14X;
   if (tiled &&
       (!cpu_tiling_supported ||
        (res->layout.tiling != AIL_TILING_GPU && res->layout.tiling != AIL_TILING_TWIDDLED) ||
        res->heap_props.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
        res->heap_props.MemoryPoolPreference != D3D12_MEMORY_POOL_L0 ||
        res->desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || res->reserved ||
        res->layout.levels != 1 || res->desc.DepthOrArraySize != 1 ||
        res->layout.sample_count_sa != 1 || res->layout.compressed ||
        !util_has_cache_ops() ||
        (format != PIPE_FORMAT_R8G8B8A8_UNORM && format != PIPE_FORMAT_B8G8R8A8_UNORM) ||
        res->offset_B > res->bo->size ||
        res->layout.size_B > res->bo->size - res->offset_B))
      return E_INVALIDARG;

   void *mapping = agx_bo_map(res->bo);
   if (!mapping)
      return E_OUTOFMEMORY;

   const uint32_t right_el = DIV_ROUND_UP(b->right, block_width);
   const uint32_t bottom_el = DIV_ROUND_UP(b->bottom, block_height);
   *out = (struct d12_cpu_subresource){
      .map = (uint8_t *)mapping + res->offset_B,
      .x_el = b->left / block_width,
      .y_el = b->top / block_height,
      .width_el = right_el - b->left / block_width,
      .height_el = bottom_el - b->top / block_height,
      .blocksize_B = blocksize,
      .texture_pitch_B = tiled ? 0 : ail_get_linear_stride_B(&res->layout, 0),
      .tiled = tiled,
   };

   if (tiled)
      return S_OK;

   const uint64_t last =
      (uint64_t)(out->y_el + out->height_el - 1) * out->texture_pitch_B +
      (uint64_t)(out->x_el + out->width_el) * blocksize;
   if (last > res->layout.size_B ||
       res->offset_B > res->bo->size ||
       last > res->bo->size - res->offset_B)
      return E_INVALIDARG;

   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_resource_WriteToSubresource(ID3D12Resource *iface, UINT dst_subresource,
                                const D3D12_BOX *dst_box, const void *src,
                                UINT src_row_pitch, UINT src_depth_pitch)
{
   (void)src_depth_pitch;

   struct d12_resource *res = d12_resource_from_iface(iface);
   struct d12_cpu_subresource surface;
   HRESULT hr = d12_resource_cpu_subresource(res, dst_subresource, dst_box,
                                             &surface);
   if (FAILED(hr) || surface.empty)
      return hr;
   if (!src)
      return E_INVALIDARG;

   const uint64_t row_bytes =
      (uint64_t)surface.width_el * surface.blocksize_B;
   if (src_row_pitch < row_bytes ||
       (surface.tiled && src_row_pitch % surface.blocksize_B))
      return E_INVALIDARG;

   if (surface.tiled) {
      /* UNKNOWN-layout UMA textures expose no CPU pointer. AIL owns the
       * resource's actual tiled layout; only caller-provided bytes are stored. */
      ail_tile(surface.map, (void *)src, &res->layout, 0, src_row_pitch,
               surface.x_el, surface.y_el, surface.width_el, surface.height_el);
      /* Non-coherent UMA is intentionally advertised. Publish dirty CPU
       * cache lines before the resource can return to GPU ownership. */
      util_flush_range(surface.map, res->layout.size_B);
      if (agx_bo_sync_for_device(res->bo, res->offset_B, res->layout.size_B))
         return E_FAIL;
      return S_OK;
   }

   uint8_t *dst = surface.map +
                  (uint64_t)surface.y_el * surface.texture_pitch_B +
                  (uint64_t)surface.x_el * surface.blocksize_B;
   for (uint32_t y = 0; y < surface.height_el; ++y)
      memcpy(dst + (uint64_t)y * surface.texture_pitch_B,
             (const uint8_t *)src + (uint64_t)y * src_row_pitch, row_bytes);

   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_resource_ReadFromSubresource(ID3D12Resource *iface, void *dst,
                                 UINT dst_row_pitch, UINT dst_depth_pitch,
                                 UINT src_subresource,
                                 const D3D12_BOX *src_box)
{
   (void)dst_depth_pitch;

   struct d12_resource *res = d12_resource_from_iface(iface);
   struct d12_cpu_subresource surface;
   HRESULT hr = d12_resource_cpu_subresource(res, src_subresource, src_box,
                                             &surface);
   if (FAILED(hr) || surface.empty)
      return hr;
   if (!dst)
      return E_INVALIDARG;

   const uint64_t row_bytes =
      (uint64_t)surface.width_el * surface.blocksize_B;
   if (dst_row_pitch < row_bytes ||
       (surface.tiled && dst_row_pitch % surface.blocksize_B))
      return E_INVALIDARG;

   if (surface.tiled) {
      /* Copy the real EOT-written resource bytes after the caller's fence and
       * COMMON transition; this does not invoke a shader or synthesize pixels. */
      /* The caller synchronized GPU completion before this CPU access.
       * Drop clean/stale CPU lines before reading actual EOT writes. */
      util_flush_inval_range(surface.map, res->layout.size_B);
      ail_detile(surface.map, dst, &res->layout, 0, dst_row_pitch,
                 surface.x_el, surface.y_el, surface.width_el, surface.height_el);
      return S_OK;
   }

   const uint8_t *src = surface.map +
                        (uint64_t)surface.y_el * surface.texture_pitch_B +
                        (uint64_t)surface.x_el * surface.blocksize_B;
   for (uint32_t y = 0; y < surface.height_el; ++y)
      memcpy((uint8_t *)dst + (uint64_t)y * dst_row_pitch,
             src + (uint64_t)y * surface.texture_pitch_B, row_bytes);

   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_resource_GetHeapProperties(ID3D12Resource *iface,
                               D3D12_HEAP_PROPERTIES *heap_props,
                               D3D12_HEAP_FLAGS *heap_flags)
{
   struct d12_resource *res = d12_resource_from_iface(iface);

   /* Reserved resources do not have heap properties.  Leave caller-owned
    * outputs untouched on failure, matching the D3D12 contract. */
   if (res->reserved)
      return E_INVALIDARG;

   if (heap_props)
      *heap_props = res->heap_props;
   if (heap_flags)
      *heap_flags = res->heap_flags;
   return S_OK;
}

static const ID3D12ResourceVtbl d12_resource_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12Resource_DEFAULTS,
   .QueryInterface = d12_resource_QueryInterface,
   .AddRef = d12_resource_AddRef,
   .Release = d12_resource_Release,
   .GetPrivateData = d12_resource_GetPrivateData,
   .SetPrivateData = d12_resource_SetPrivateData,
   .SetPrivateDataInterface = d12_resource_SetPrivateDataInterface,
   .SetName = d12_resource_SetName,
   .GetDevice = d12_resource_GetDevice,
   .Map = d12_resource_Map,
   .Unmap = d12_resource_Unmap,
   .GetDesc = d12_resource_GetDesc,
   .GetGPUVirtualAddress = d12_resource_GetGPUVirtualAddress,
   .WriteToSubresource = d12_resource_WriteToSubresource,
   .ReadFromSubresource = d12_resource_ReadFromSubresource,
   .GetHeapProperties = d12_resource_GetHeapProperties,
};

static bool
d12_resource_init_sparse_metadata(struct d12_resource *res,
                                  const D3D12_RESOURCE_DESC *desc)
{
   const uint32_t tile_B = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
   simple_mtx_init(&res->sparse_lock, mtx_plain);

   if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      res->sparse_tile_count = DIV_ROUND_UP(desc->Width, tile_B);
      res->sparse_subresource_count = 1;
      res->sparse_tile_shape = (D3D12_TILE_SHAPE){tile_B, 1, 1};
      res->sparse_tilings = calloc(1, sizeof(*res->sparse_tilings));
      if (!res->sparse_tilings)
         goto fail;
      res->sparse_tilings[0] = (D3D12_SUBRESOURCE_TILING){
         .WidthInTiles = res->sparse_tile_count,
         .HeightInTiles = 1,
         .DepthInTiles = 1,
         .StartTileIndexInOverallResource = 0,
      };
   } else {
      const struct ail_layout *layout = &res->layout;
      const unsigned levels = desc->MipLevels ? desc->MipLevels
                                               : layout->levels;
      const unsigned layers = MAX2(desc->DepthOrArraySize, 1);
      const unsigned tail_first = MIN2(layout->mip_tail_first_lod, levels);
      const unsigned bw = util_format_get_blockwidth(layout->format);
      const unsigned bh = util_format_get_blockheight(layout->format);
      const unsigned block_B =
         util_format_get_blocksize(layout->format) * layout->sample_count_sa;
      /* One shared derivation with the Tier 2 packing argument: AGX packs
       * from the first level below one hardware tile, and the D3D standard
       * tile is two hardware tiles on each axis, so a level that is a whole
       * standard tile is always nonpacked. See d12_sparse_residency_core.h. */
      uint32_t std_w_el = 0, std_h_el = 0;
      if (!d12_sparse_standard_tile_el(block_B, &std_w_el, &std_h_el)) {
         mesa_loge("reserved resource block size %u has no AGX tile shape",
                   block_B);
         goto fail;
      }
      assert(std_w_el == ail_get_max_tile_size(block_B).width_el *
                            D12_SPARSE_STANDARD_TILE_SCALE &&
             std_h_el == ail_get_max_tile_size(block_B).height_el *
                            D12_SPARSE_STANDARD_TILE_SCALE &&
             "standard tile shape must stay two hardware tiles per axis");

      res->sparse_tile_shape = (D3D12_TILE_SHAPE){
         .WidthInTexels = std_w_el * bw,
         .HeightInTexels = std_h_el * bh,
         .DepthInTexels = 1,
      };
      res->sparse_subresource_count = levels * layers;
      res->sparse_mip_count = levels;
      res->sparse_tilings =
         calloc(res->sparse_subresource_count, sizeof(*res->sparse_tilings));
      if (!res->sparse_tilings)
         goto fail;

      uint32_t tile = 0;
      for (unsigned layer = 0; layer < layers; layer++) {
         for (unsigned level = 0; level < tail_first; level++) {
            const unsigned subresource = layer * levels + level;
            const unsigned width_el = util_format_get_nblocksx(
               layout->format, u_minify(layout->width_px, level));
            const unsigned height_el = util_format_get_nblocksy(
               layout->format, u_minify(layout->height_px, level));
            D3D12_SUBRESOURCE_TILING *tiling =
               &res->sparse_tilings[subresource];
            tiling->WidthInTiles = DIV_ROUND_UP(width_el, std_w_el);
            tiling->HeightInTiles = DIV_ROUND_UP(height_el, std_h_el);
            tiling->DepthInTiles = 1;
            tiling->StartTileIndexInOverallResource = tile;
            tile += tiling->WidthInTiles * tiling->HeightInTiles;
         }
         for (unsigned level = tail_first; level < levels; level++) {
            res->sparse_tilings[layer * levels + level]
               .StartTileIndexInOverallResource = UINT32_MAX;
         }
      }

      const uint32_t packed_start = tile;
      uint32_t packed_tiles = 0;
      if (tail_first < levels) {
         uint64_t tail_size_B =
            (layout->level_offsets_B[levels] -
             layout->level_offsets_B[tail_first]) * layers;
         packed_tiles = DIV_ROUND_UP(tail_size_B, tile_B);
         tile += packed_tiles;
      }
      res->sparse_packed_mips = (D3D12_PACKED_MIP_INFO){
         .NumStandardMips = tail_first,
         .NumPackedMips = levels - tail_first,
         .NumTilesForPackedMips = packed_tiles,
         .StartTileIndexInOverallResource = packed_tiles ? packed_start : 0,
      };
      res->sparse_tile_count = tile;
   }

   res->sparse_mappings =
      calloc(res->sparse_tile_count, sizeof(*res->sparse_mappings));
   if (!res->sparse_mappings)
      goto fail;
   return true;

fail:
   free(res->sparse_tilings);
   res->sparse_tilings = NULL;
   simple_mtx_destroy(&res->sparse_lock);
   return false;
}

struct d12_sparse_update {
   uint32_t tile;
   struct agx_bo *bo;
   uint64_t heap_offset_B;
   bool skip;
};

static bool
d12_sparse_coordinate_tile(const struct d12_resource *res,
                           const D3D12_TILED_RESOURCE_COORDINATE *coord,
                           uint32_t *tile)
{
   if (coord->Subresource >= res->sparse_subresource_count)
      return false;

   const D3D12_SUBRESOURCE_TILING *tiling =
      &res->sparse_tilings[coord->Subresource];
   if (!tiling->WidthInTiles) {
      if (coord->Y || coord->Z ||
          coord->X >= res->sparse_packed_mips.NumTilesForPackedMips)
         return false;
      *tile = res->sparse_packed_mips.StartTileIndexInOverallResource +
              coord->X;
      return true;
   }

   if (coord->X >= tiling->WidthInTiles ||
       coord->Y >= tiling->HeightInTiles ||
       coord->Z >= tiling->DepthInTiles)
      return false;

   *tile = tiling->StartTileIndexInOverallResource +
           ((coord->Z * tiling->HeightInTiles + coord->Y) *
               tiling->WidthInTiles +
            coord->X);
   return true;
}

HRESULT
d12_sparse_collect_tiles(
   const struct d12_resource *res, UINT num_regions,
   const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, uint32_t **out_tiles,
   uint32_t *out_count)
{
   if (!out_tiles || !out_count)
      return E_INVALIDARG;
   *out_tiles = NULL;
   *out_count = 0;
   if (!num_regions)
      return S_OK;
   if (!res || !res->reserved || (!starts && num_regions != 1))
      return E_INVALIDARG;

   /* A fully omitted one-region description means the complete resource. */
   if (!starts && !sizes) {
      uint32_t *tiles = malloc((size_t)res->sparse_tile_count * sizeof(*tiles));
      if (!tiles)
         return E_OUTOFMEMORY;
      for (uint32_t i = 0; i < res->sparse_tile_count; i++)
         tiles[i] = i;
      *out_tiles = tiles;
      *out_count = res->sparse_tile_count;
      return S_OK;
   }

   uint64_t total = 0;
   for (UINT r = 0; r < num_regions; r++) {
      if (!sizes) {
         total++;
      } else if (sizes[r].UseBox) {
         uint64_t count = (uint64_t)sizes[r].Width * sizes[r].Height *
                          sizes[r].Depth;
         if (!count || count != sizes[r].NumTiles)
            return E_INVALIDARG;
         total += count;
      } else {
         total += sizes[r].NumTiles;
      }
      if (total > UINT32_MAX)
         return E_INVALIDARG;
   }

   uint32_t *tiles = malloc((size_t)total * sizeof(*tiles));
   if (!tiles && total)
      return E_OUTOFMEMORY;

   uint32_t cursor = 0;
   const D3D12_TILED_RESOURCE_COORDINATE zero = {0};
   for (UINT r = 0; r < num_regions; r++) {
      const D3D12_TILED_RESOURCE_COORDINATE *start = starts ? &starts[r] : &zero;
      uint32_t first;
      if (!d12_sparse_coordinate_tile(res, start, &first))
         goto invalid;

      if (!sizes) {
         tiles[cursor++] = first;
         continue;
      }

      const D3D12_TILE_REGION_SIZE *size = &sizes[r];
      if (!size->UseBox) {
         if ((uint64_t)first + size->NumTiles > res->sparse_tile_count)
            goto invalid;
         for (UINT i = 0; i < size->NumTiles; i++)
            tiles[cursor++] = first + i;
         continue;
      }

      const D3D12_SUBRESOURCE_TILING *tiling =
         &res->sparse_tilings[start->Subresource];
      if (!tiling->WidthInTiles ||
          (uint64_t)start->X + size->Width > tiling->WidthInTiles ||
          (uint64_t)start->Y + size->Height > tiling->HeightInTiles ||
          (uint64_t)start->Z + size->Depth > tiling->DepthInTiles)
         goto invalid;

      for (UINT z = 0; z < size->Depth; z++) {
         for (UINT y = 0; y < size->Height; y++) {
            for (UINT x = 0; x < size->Width; x++) {
               tiles[cursor++] =
                  tiling->StartTileIndexInOverallResource +
                  (((start->Z + z) * tiling->HeightInTiles + start->Y + y) *
                      tiling->WidthInTiles +
                   start->X + x);
            }
         }
      }
   }

   *out_tiles = tiles;
   *out_count = cursor;
   return S_OK;

invalid:
   free(tiles);
   return E_INVALIDARG;
}

bool
d12_sparse_get_tile_copy_info(const struct d12_resource *res, uint32_t tile,
                              struct d12_sparse_tile_copy_info *info)
{
   if (!res || !res->reserved || !info || tile >= res->sparse_tile_count)
      return false;

   memset(info, 0, sizeof(*info));
   if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      info->offset_B =
         (uint64_t)tile * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
      return true;
   }

   if (res->sparse_packed_mips.NumPackedMips &&
       tile >= res->sparse_packed_mips.StartTileIndexInOverallResource) {
      const unsigned tail = res->layout.mip_tail_first_lod;
      info->offset_B =
         res->layout.level_offsets_B[tail] +
         (uint64_t)(tile -
                    res->sparse_packed_mips.StartTileIndexInOverallResource) *
            D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
      return true;
   }

   for (uint32_t subresource = 0;
        subresource < res->sparse_subresource_count; subresource++) {
      const D3D12_SUBRESOURCE_TILING *tiling =
         &res->sparse_tilings[subresource];
      uint64_t count = (uint64_t)tiling->WidthInTiles *
                       tiling->HeightInTiles * tiling->DepthInTiles;
      uint64_t end = tiling->StartTileIndexInOverallResource + count;
      if (!count || tile < tiling->StartTileIndexInOverallResource ||
          tile >= end)
         continue;

      uint32_t local = tile - tiling->StartTileIndexInOverallResource;
      info->standard = true;
      info->subresource = subresource;
      info->x = local % tiling->WidthInTiles;
      local /= tiling->WidthInTiles;
      info->y = local % tiling->HeightInTiles;
      info->z = local / tiling->HeightInTiles;
      return true;
   }

   return false;
}

static unsigned
d12_sparse_tile_pages(const struct d12_resource *res, uint32_t tile,
                      uint64_t page_offset_B[4], uint8_t heap_page[4])
{
   const uint64_t page_B = AIL_PAGESIZE;
   unsigned count = 0;

   struct d12_sparse_tile_copy_info info;
   if (!d12_sparse_get_tile_copy_info(res, tile, &info))
      return 0;

   if (!info.standard) {
      uint64_t base = info.offset_B;

      for (unsigned q = 0; q < 4; q++) {
         uint64_t offset = base + q * page_B;
         if (offset >= res->sparse_size_B)
            break;
         page_offset_B[count] = offset;
         heap_page[count++] = q;
      }
      return count;
   }

   const struct ail_layout *layout = &res->layout;
   const unsigned levels = res->sparse_mip_count;
   const unsigned level = info.subresource % levels;
   const unsigned layer = info.subresource / levels;
   const unsigned tx = info.x;
   const unsigned ty = info.y;
   const struct ail_tile hw = layout->tilesize_el[level];
   const unsigned std_w = hw.width_el * 2;
   const unsigned std_h = hw.height_el * 2;
   const unsigned width_el = util_format_get_nblocksx(
      layout->format, u_minify(layout->width_px, level));
   const unsigned height_el = util_format_get_nblocksy(
      layout->format, u_minify(layout->height_px, level));

   for (unsigned hy = 0; hy < 2; hy++) {
      for (unsigned hx = 0; hx < 2; hx++) {
         unsigned x = tx * std_w + hx * hw.width_el;
         unsigned y = ty * std_h + hy * hw.height_el;
         if (x >= width_el || y >= height_el)
            continue;

         uint64_t offset =
            ail_get_twiddled_block_B(layout, level, x, y, layer);
         if (offset >= res->sparse_size_B || (offset % page_B))
            return 0;
         page_offset_B[count] = offset;
         heap_page[count++] = hy * 2 + hx;
      }
   }
   return count;
}

static HRESULT
d12_sparse_apply_updates(struct d12_resource *res,
                         const struct d12_sparse_update *updates,
                         uint32_t update_count,
                         struct drm_asahi_gem_bind_op **out_ops,
                         uint32_t *out_count,
                         struct d12_sparse_map_update **out_map_updates,
                         uint32_t *out_map_count)
{
   if (!out_ops || !out_count || !out_map_updates || !out_map_count)
      return E_INVALIDARG;
   *out_ops = NULL;
   *out_count = 0;
   *out_map_updates = NULL;
   *out_map_count = 0;

   if (update_count > UINT32_MAX / 8)
      return E_INVALIDARG;
   struct drm_asahi_gem_bind_op *ops =
      calloc((size_t)update_count * 8, sizeof(*ops));
   if (!ops && update_count)
      return E_OUTOFMEMORY;

   struct d12_sparse_map_update *map_updates = NULL;
   if (res->sparse_map) {
      if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER &&
          (!res->layout.depth_px ||
           res->layout.layer_stride_B >
              UINT64_MAX / res->layout.depth_px)) {
         free(ops);
         return E_INVALIDARG;
      }
      map_updates = calloc((size_t)update_count * 4,
                           sizeof(*map_updates));
      if (!map_updates && update_count) {
         free(ops);
         return E_OUTOFMEMORY;
      }
   }

   uint32_t count = 0;
   uint32_t map_count = 0;
   for (uint32_t i = 0; i < update_count; i++) {
      const struct d12_sparse_update *update = &updates[i];
      if (update->skip)
         continue;

      uint64_t page_offset_B[4];
      uint8_t heap_page[4];
      unsigned pages =
         d12_sparse_tile_pages(res, update->tile, page_offset_B, heap_page);
      if (!pages) {
         free(map_updates);
         free(ops);
         return E_INVALIDARG;
      }

      if (res->sparse_map &&
          res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         map_updates[map_count++] = (struct d12_sparse_map_update){
            .index_el = update->tile,
            .enabled = update->bo != NULL,
         };
      }

      for (unsigned p = 0; p < pages; p++) {
         if (res->sparse_map &&
             res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
            uint32_t index_el;
            const uint64_t logical_span =
               res->layout.layer_stride_B * res->layout.depth_px;

            /* A D3D12 tile is four AGX pages, so the final tile may cover
             * harmless VA padding beyond the final layer. It has no hardware
             * sparse-table word and is never texture-addressable. Every page
             * inside the logical image must resolve before anything mutates. */
            if (page_offset_B[p] < logical_span) {
               if (!d12_sparse_residency_index(
                      page_offset_B[p], res->layout.layer_stride_B,
                      res->layout.depth_px,
                      res->layout.sparse_folios_per_layer,
                      res->layout.sparse_table_size_B, &index_el)) {
                  free(map_updates);
                  free(ops);
                  return E_INVALIDARG;
               }
               map_updates[map_count++] = (struct d12_sparse_map_update){
                  .read_address = res->addr +
                                  res->device->dev.sparse_ro_offset +
                                  page_offset_B[p],
                  .index_el = index_el,
                  .enabled = update->bo != NULL,
               };
            }
         }

         struct drm_asahi_gem_bind_op op = {
            .handle = update->bo ? update->bo->uapi_handle
                                 : res->device->dev.scratch_bo->uapi_handle,
            .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
            .offset = update->bo
                         ? update->heap_offset_B +
                              (uint64_t)heap_page[p] * AIL_PAGESIZE
                         : 0,
            .range = AIL_PAGESIZE,
            .addr = res->addr + page_offset_B[p],
         };
         if (!update->bo)
            op.flags |= DRM_ASAHI_BIND_SINGLE_PAGE;
         ops[count++] = op;

         op.addr += res->device->dev.sparse_ro_offset;
         op.flags &= ~DRM_ASAHI_BIND_WRITE;
         if (!update->bo)
            op.handle = res->device->dev.zero_bo->uapi_handle;
         ops[count++] = op;
      }

      if (getenv("D3D12AGX_DUMP_SPARSE")) {
         mesa_logi("sparse tile %u pages=%u bo=%u heap=0x%llx "
                   "va=0x%llx..0x%llx",
                   update->tile, pages,
                   update->bo ? update->bo->uapi_handle : 0,
                   (unsigned long long)update->heap_offset_B,
                   (unsigned long long)ops[count - pages * 2].addr,
                   (unsigned long long)(ops[count - 2].addr + AIL_PAGESIZE));
      }
   }

   /* Hold every replacement alive before dropping any old tile reference.
    * This is required for overlapping CopyTileMappings within one resource:
    * a later source tile may be the last owner of a BO overwritten earlier in
    * the same batch.  These references become the destination tiles' retained
    * references below. */
   for (uint32_t i = 0; i < update_count; i++) {
      if (!updates[i].skip && updates[i].bo)
         agx_bo_reference(updates[i].bo);
   }

   /* Mutate the software mapping only after every address and heap range has
    * validated and the complete kernel batch has been allocated. */
   for (uint32_t i = 0; i < update_count; i++) {
      const struct d12_sparse_update *update = &updates[i];
      if (update->skip)
         continue;
      struct d12_sparse_mapping *mapping =
         &res->sparse_mappings[update->tile];
      if (mapping->bo)
         agx_bo_unreference(&res->device->dev, mapping->bo);
      mapping->bo = update->bo;
      mapping->heap_offset_B = update->heap_offset_B;
   }

   *out_ops = ops;
   *out_count = count;
   *out_map_updates = map_updates;
   *out_map_count = map_count;
   return S_OK;
}

bool
d12_sparse_commit_map_updates(
   struct d12_resource *res,
   const struct d12_sparse_map_update *updates, uint32_t update_count)
{
   if (!update_count)
      return true;
   if (!res || !res->sparse_map || !updates)
      return false;

   uint32_t *sparse_words = agx_bo_map(res->sparse_map);
   if (!sparse_words)
      return false;

   simple_mtx_lock(&res->sparse_lock);
   for (uint32_t i = 0; i < update_count; ++i) {
      const struct d12_sparse_map_update *update = &updates[i];
      if ((uint64_t)update->index_el * AIL_SPARSE_ELSIZE_B >=
             res->sparse_map_size_B ||
          (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
           update->index_el >= res->sparse_tile_count)) {
         simple_mtx_unlock(&res->sparse_lock);
         return false;
      }
   }

   for (uint32_t i = 0; i < update_count; ++i) {
      const struct d12_sparse_map_update *update = &updates[i];
      if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         sparse_words[update->index_el] =
            update->enabled ? UINT32_MAX : 0;
      } else {
         agx_pack(sparse_words + update->index_el, SPARSE_BLOCK, cfg) {
            cfg.enabled = update->enabled;
            cfg.unknown = cfg.enabled;
            if (cfg.enabled)
               cfg.address = update->read_address;
         }
      }
   }
   simple_mtx_unlock(&res->sparse_lock);
   return true;
}

HRESULT
d12_sparse_update_tile_mappings(
   struct d12_resource *res, UINT num_regions,
   const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, struct d12_heap *heap,
   UINT num_ranges, const D3D12_TILE_RANGE_FLAGS *range_flags,
   const UINT *heap_starts, const UINT *range_counts,
   struct drm_asahi_gem_bind_op **out_ops, uint32_t *out_count,
   struct d12_sparse_map_update **out_map_updates,
   uint32_t *out_map_count)
{
   if (!out_ops || !out_count || !out_map_updates || !out_map_count)
      return E_INVALIDARG;
   *out_ops = NULL;
   *out_count = 0;
   *out_map_updates = NULL;
   *out_map_count = 0;
   uint32_t *tiles = NULL;
   uint32_t tile_count = 0;
   HRESULT hr = d12_sparse_collect_tiles(res, num_regions, starts, sizes,
                                         &tiles, &tile_count);
   if (FAILED(hr))
      return hr;
   if (!tile_count) {
      free(tiles);
      return S_OK;
   }
   if (!num_ranges || (!range_counts && num_ranges != 1)) {
      free(tiles);
      return E_INVALIDARG;
   }

   struct d12_sparse_update *updates =
      calloc(tile_count, sizeof(*updates));
   if (!updates) {
      free(tiles);
      return E_OUTOFMEMORY;
   }

   uint32_t cursor = 0;
   for (UINT r = 0; r < num_ranges; r++) {
      D3D12_TILE_RANGE_FLAGS flags =
         range_flags ? range_flags[r] : D3D12_TILE_RANGE_FLAG_NONE;
      UINT declared_count = range_counts ? range_counts[r] : tile_count;
      UINT count = MIN2(declared_count, tile_count - cursor);
      UINT heap_tile = heap_starts ? heap_starts[r] : 0;
      if (flags != D3D12_TILE_RANGE_FLAG_NONE &&
           flags != D3D12_TILE_RANGE_FLAG_NULL &&
           flags != D3D12_TILE_RANGE_FLAG_SKIP &&
           flags != D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE) {
         hr = E_INVALIDARG;
         goto out;
      }

      /* The mapping ranges are a stream consumed by the resource regions.
       * D3D12 permits the final range to extend past that stream; its surplus
       * entries have no destination tiles and are ignored. */
      if (!count)
         break;

      if ((flags == D3D12_TILE_RANGE_FLAG_NONE ||
           flags == D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE) &&
          (!heap ||
           ((uint64_t)heap_tile +
                (flags == D3D12_TILE_RANGE_FLAG_NONE ? count : 1)) *
                 D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES >
              heap->bo->size)) {
         hr = E_INVALIDARG;
         goto out;
      }

      for (UINT i = 0; i < count; i++) {
         updates[cursor] = (struct d12_sparse_update){
            .tile = tiles[cursor],
            .bo = (flags == D3D12_TILE_RANGE_FLAG_NONE ||
                   flags == D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE)
                     ? heap->bo
                     : NULL,
            .heap_offset_B =
               (uint64_t)(heap_tile +
                          (flags == D3D12_TILE_RANGE_FLAG_NONE ? i : 0)) *
               D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES,
            .skip = flags == D3D12_TILE_RANGE_FLAG_SKIP,
         };
         cursor++;
      }
   }
   if (cursor != tile_count) {
      hr = E_INVALIDARG;
      goto out;
   }

   simple_mtx_lock(&res->sparse_lock);
   hr = d12_sparse_apply_updates(res, updates, tile_count, out_ops, out_count,
                                 out_map_updates, out_map_count);
   simple_mtx_unlock(&res->sparse_lock);

out:
   free(updates);
   free(tiles);
   return hr;
}

HRESULT
d12_sparse_copy_tile_mappings(
   struct d12_resource *dst, const D3D12_TILED_RESOURCE_COORDINATE *dst_start,
   struct d12_resource *src, const D3D12_TILED_RESOURCE_COORDINATE *src_start,
   const D3D12_TILE_REGION_SIZE *size,
   struct drm_asahi_gem_bind_op **out_ops, uint32_t *out_count,
   struct d12_sparse_map_update **out_map_updates,
   uint32_t *out_map_count)
{
   if (!out_ops || !out_count || !out_map_updates || !out_map_count)
      return E_INVALIDARG;
   *out_ops = NULL;
   *out_count = 0;
   *out_map_updates = NULL;
   *out_map_count = 0;

   if (!dst || !src || !dst_start || !src_start || !size ||
       !dst->reserved || !src->reserved)
      return E_INVALIDARG;

   uint32_t *dst_tiles = NULL, *src_tiles = NULL;
   uint32_t dst_count = 0, src_count = 0;
   HRESULT hr = d12_sparse_collect_tiles(dst, 1, dst_start, size, &dst_tiles,
                                         &dst_count);
   if (FAILED(hr))
      return hr;
   hr = d12_sparse_collect_tiles(src, 1, src_start, size, &src_tiles,
                                 &src_count);
   if (FAILED(hr) || src_count != dst_count) {
      free(dst_tiles);
      free(src_tiles);
      return FAILED(hr) ? hr : E_INVALIDARG;
   }

   struct d12_sparse_update *updates = calloc(dst_count, sizeof(*updates));
   if (!updates && dst_count) {
      free(dst_tiles);
      free(src_tiles);
      return E_OUTOFMEMORY;
   }

   struct d12_resource *first =
      (uintptr_t)dst < (uintptr_t)src ? dst : src;
   struct d12_resource *second = first == dst ? src : dst;
   simple_mtx_lock(&first->sparse_lock);
   if (second != first)
      simple_mtx_lock(&second->sparse_lock);

   for (uint32_t i = 0; i < dst_count; i++) {
      const struct d12_sparse_mapping *mapping =
         &src->sparse_mappings[src_tiles[i]];
      updates[i] = (struct d12_sparse_update){
         .tile = dst_tiles[i],
         .bo = mapping->bo,
         .heap_offset_B = mapping->heap_offset_B,
      };
   }
   hr = d12_sparse_apply_updates(dst, updates, dst_count, out_ops, out_count,
                                 out_map_updates, out_map_count);

   if (second != first)
      simple_mtx_unlock(&second->sparse_lock);
   simple_mtx_unlock(&first->sparse_lock);
   free(updates);
   free(dst_tiles);
   free(src_tiles);
   return hr;
}

bool
d12_resource_iface_is_ours(ID3D12Resource *iface)
{
   return iface && iface->lpVtbl == &d12_resource_vtbl;
}

bool
d12_resource_explicit_row_major_size(
   const D3D12_RESOURCE_DESC *desc, uint32_t row_pitch_B,
   uint32_t slice_pitch_B, uint64_t *size_B)
{
   enum pipe_format pipe;
   uint64_t minimum_pitch_B;

   if (size_B)
      *size_B = 0;
   if (!row_pitch_B && !slice_pitch_B)
      return true;
   if (!desc || !row_pitch_B || !slice_pitch_B ||
       desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR ||
       desc->Width > UINT32_MAX || !desc->Height ||
       desc->DepthOrArraySize != 1 || desc->MipLevels != 1 ||
       desc->SampleDesc.Count != 1 || desc->SampleDesc.Quality ||
       !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) ||
       (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) ||
       (row_pitch_B & 255u) || desc->Height > UINT32_MAX / row_pitch_B ||
       slice_pitch_B != row_pitch_B * desc->Height ||
       d12_format_is_depth_stencil(desc->Format))
      return false;

   pipe = d12_format_to_pipe(desc->Format);
   if (pipe == PIPE_FORMAT_NONE || util_format_get_blockwidth(pipe) != 1 ||
       util_format_get_blockheight(pipe) != 1 ||
       util_format_get_blocksize(pipe) > 16)
      return false;
   minimum_pitch_B = util_format_get_stride(pipe, (uint32_t)desc->Width);
   if (row_pitch_B < minimum_pitch_B)
      return false;
   if (size_B)
      *size_B = slice_pitch_B;
   return true;
}

/* One texture-layout decision feeds both allocation-size reflection and
 * resource creation.  The D3D12 DDI explicitly makes sizing the first half
 * of creation and promises that the resolved layout is passed back in the
 * second half, so using a second estimate here is an ABI violation rather
 * than merely wasted padding. */
static HRESULT
d12_resource_compute_texture_layout(
   const D3D12_RESOURCE_DESC *desc, uint32_t row_pitch_B,
   uint32_t slice_pitch_B, bool reserved, enum ail_layout_policy policy,
   struct ail_layout *layout, struct ail_layout *stencil_layout,
   bool *is_depth_stencil)
{
   uint64_t explicit_row_major_size_B = 0;
   const bool interop_layout =
      desc && desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR &&
      (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
   const bool gpu_tiled =
      desc &&
      (reserved || interop_layout ||
       desc->Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);

   if (!desc || !layout || !stencil_layout || !is_depth_stencil ||
       desc->Dimension < D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
       desc->Dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
       (desc->Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN &&
        desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR &&
        desc->Layout != D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) ||
       !desc->Width || desc->Width > UINT32_MAX || !desc->Height ||
       !desc->DepthOrArraySize || !desc->SampleDesc.Count ||
       !d12_resource_explicit_row_major_size(
          desc, row_pitch_B, slice_pitch_B, &explicit_row_major_size_B))
      return E_INVALIDARG;
   if (desc->Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR &&
       (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc->MipLevels != 1 || desc->DepthOrArraySize != 1 ||
        desc->SampleDesc.Count != 1 || desc->SampleDesc.Quality ||
        !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) ||
        (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))))
      return E_INVALIDARG;

   memset(layout, 0, sizeof(*layout));
   memset(stencil_layout, 0, sizeof(*stencil_layout));
   *is_depth_stencil = false;

   if ((desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
        desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) &&
       d12_format_is_depth_stencil(desc->Format)) {
      enum pipe_format z = d12_format_depth_plane(desc->Format);
      enum pipe_format s = d12_format_stencil_plane(desc->Format);
      const unsigned samples = MAX2(desc->SampleDesc.Count, 1);
      const unsigned full_levels =
         util_logbase2(MAX2((uint32_t)MAX2(desc->Width, 1),
                            MAX2(desc->Height, 1))) + 1;
      const unsigned levels = desc->MipLevels ? desc->MipLevels : full_levels;

      if (z == PIPE_FORMAT_NONE ||
          (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
           (desc->Height != 1 || samples != 1)) ||
          !d12_sample_count_supported(samples) || !desc->Width ||
          !desc->Height) {
         mesa_loge("D3D12 texture layout: depth/stencil shape unsupported "
                   "(mips %u, layers %u, samples %u)",
                   desc->MipLevels, desc->DepthOrArraySize,
                   desc->SampleDesc.Count);
         return E_NOTIMPL;
      }

      if (d12_format_depth_is_substituted(desc->Format)) {
         static bool warned = false;
         if (!warned) {
            warned = true;
            mesa_logw("depth format %d has no native equivalent on AGX; "
                      "using 32-bit float depth. Comparisons and clears "
                      "quantize as float32, not unorm24.",
                      (int)desc->Format);
         }
      }

      *is_depth_stencil = true;
      *layout = (struct ail_layout){
         .policy = policy,
         .width_px = (uint32_t)desc->Width,
         .height_px = desc->Height,
         .depth_px = MAX2(desc->DepthOrArraySize, 1),
         .sample_count_sa = samples,
         .levels = levels,
         .tiling = gpu_tiled ? AIL_TILING_GPU : AIL_TILING_TWIDDLED,
         .format = z,
         .writeable_image = interop_layout,
         .renderable = interop_layout,
      };
      ail_make_miptree(layout);

      if (s != PIPE_FORMAT_NONE) {
         *stencil_layout = (struct ail_layout){
            .policy = policy,
            .width_px = (uint32_t)desc->Width,
            .height_px = desc->Height,
            .depth_px = MAX2(desc->DepthOrArraySize, 1),
            .sample_count_sa = samples,
            .levels = levels,
            .tiling = gpu_tiled ? AIL_TILING_GPU : AIL_TILING_TWIDDLED,
            .format = s,
            .writeable_image = interop_layout,
            .renderable = interop_layout,
         };
         ail_make_miptree(stencil_layout);
      }
      return S_OK;
   }

   enum pipe_format pipe = d12_format_to_pipe(desc->Format);
   const bool is_3d =
      desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
   const bool is_1d =
      desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;

   if (pipe == PIPE_FORMAT_NONE || !desc->Width || !desc->Height ||
       !desc->DepthOrArraySize ||
       /* An unencodable sample count reached agx_translate_sample_count and
        * took the whole process with an UNREACHABLE. A driver refuses the
        * resource; it does not abort its caller. */
       !d12_sample_count_supported(MAX2(desc->SampleDesc.Count, 1u)) ||
       ((is_1d || is_3d) && desc->SampleDesc.Count != 1) ||
       (is_1d && desc->Height != 1)) {
      mesa_loge("D3D12 texture layout: shape unsupported "
                "(fmt %d, mips %u, layers %u, samples %u)",
                (int)desc->Format, desc->MipLevels,
                desc->DepthOrArraySize, desc->SampleDesc.Count);
      return E_NOTIMPL;
   }
   if (desc->Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR &&
       (util_format_get_blockwidth(pipe) != 1 ||
        util_format_get_blockheight(pipe) != 1 ||
        util_format_get_blocksize(pipe) > 16))
      return E_INVALIDARG;
   if ((desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) &&
       !d12_format_is_renderable(desc->Format))
      return E_INVALIDARG;

   unsigned major_axis =
      MAX2((uint32_t)MAX2(desc->Width, 1), MAX2(desc->Height, 1));
   if (is_3d)
      major_axis = MAX2(major_axis, desc->DepthOrArraySize);
   const unsigned full_levels = util_logbase2(major_axis) + 1;
   const unsigned levels = desc->MipLevels ? desc->MipLevels : full_levels;
   const unsigned samples = MAX2(desc->SampleDesc.Count, 1);
   const bool blocked = util_format_get_blockwidth(pipe) > 1 ||
                        util_format_get_blockheight(pipe) > 1;
   const bool writeable_image =
      interop_layout ||
      (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   const bool linear_ok =
      !gpu_tiled && !is_3d && levels == 1 && samples == 1 &&
      desc->DepthOrArraySize == 1 && !blocked && !writeable_image;
   uint32_t linear_stride_B = 0;

   if (linear_ok) {
      if (row_pitch_B) {
         linear_stride_B = row_pitch_B;
      } else {
         const uint32_t minimum_stride_B =
            util_format_get_stride(pipe, (uint32_t)desc->Width);
         const uint32_t alignment_B =
            desc->Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR ? 256u : 64u;
         if (minimum_stride_B > UINT32_MAX - (alignment_B - 1))
            return E_INVALIDARG;
         linear_stride_B = align(minimum_stride_B, alignment_B);
      }
   }

   *layout = (struct ail_layout){
      .policy = policy,
      .width_px = (uint32_t)desc->Width,
      .height_px = desc->Height,
      .depth_px = MAX2(desc->DepthOrArraySize, 1),
      .sample_count_sa = samples,
      .levels = levels,
      .mipmapped_z = is_3d,
      .tiling = gpu_tiled
                   ? AIL_TILING_GPU
                   : (linear_ok ? AIL_TILING_LINEAR
                                : AIL_TILING_TWIDDLED),
      .format = pipe,
      .writeable_image = writeable_image,
      .renderable = interop_layout,
      .linear_stride_B = linear_stride_B,
   };
   ail_make_miptree(layout);
   if (explicit_row_major_size_B &&
       layout->size_B != explicit_row_major_size_B)
      return E_INVALIDARG;
   return S_OK;
}

HRESULT
d12_resource_get_allocation_size(
   struct d12_device *device, const D3D12_RESOURCE_DESC *desc,
   uint32_t row_pitch_B,
   uint32_t slice_pitch_B, bool reserved, uint64_t *size_B)
{
   struct ail_layout layout;
   struct ail_layout stencil_layout;
   bool is_depth_stencil;
   HRESULT result;

   if (!device || !desc || !size_B ||
       ((row_pitch_B || slice_pitch_B) &&
        desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER))
      return E_INVALIDARG;
   if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (!desc->Width)
         return E_INVALIDARG;
      if (reserved) {
         uint64_t aligned_size_B;
         uint32_t tile_count;
         if (!d12_sparse_buffer_size(
                desc->Width, d12_core_get_caps()->gpu_virtual_address_bits,
                &aligned_size_B, &tile_count))
            return E_INVALIDARG;
      }
      *size_B = desc->Width;
      return S_OK;
   }

   const enum ail_layout_policy policy =
      ail_layout_policy_for_descriptor_generation(
         agx_chip_descriptor_generation(device->dev.chip));
   result = d12_resource_compute_texture_layout(
      desc, row_pitch_B, slice_pitch_B, reserved, policy, &layout,
      &stencil_layout, &is_depth_stencil);
   if (FAILED(result))
      return result;
   if (stencil_layout.size_B > UINT64_MAX - layout.size_B)
      return E_INVALIDARG;
   *size_B = layout.size_B + stencil_layout.size_B;
   return S_OK;
}

static HRESULT
d12_resource_create(struct d12_device *device,
                    const D3D12_HEAP_PROPERTIES *heap_props,
                    D3D12_HEAP_FLAGS heap_flags,
                    const D3D12_RESOURCE_DESC *desc, struct d12_heap *heap,
                    uint64_t heap_offset_B, uint32_t row_pitch_B,
                    uint32_t slice_pitch_B, bool reserved,
                    struct agx_bo *imported_bo, REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   if (!heap_props || !desc)
      return E_INVALIDARG;

   uint64_t size_B = 0;
   struct ail_layout layout = {0};
   struct ail_layout stencil_layout = {0};
   bool is_depth_stencil = false;
   const bool shared = heap_flags & D3D12_HEAP_FLAG_SHARED;

   if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (!desc->Width || row_pitch_B || slice_pitch_B)
         return E_INVALIDARG;
      size_B = desc->Width;
   } else {
      const enum ail_layout_policy policy =
         ail_layout_policy_for_descriptor_generation(
            agx_chip_descriptor_generation(device->dev.chip));
      HRESULT layout_result = d12_resource_compute_texture_layout(
         desc, row_pitch_B, slice_pitch_B, reserved, policy, &layout,
         &stencil_layout, &is_depth_stencil);
      if (FAILED(layout_result))
         return layout_result;
      size_B = layout.size_B;
   }

   D3D12_RESOURCE_DESC resolved_desc = *desc;
   if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
      resolved_desc.MipLevels =
         desc->MipLevels ? desc->MipLevels : (layout.levels ? layout.levels : 1);
      if (!resolved_desc.Alignment) {
         resolved_desc.Alignment =
            desc->SampleDesc.Count > 1
               ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
               : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
      }
   } else if (!resolved_desc.Alignment) {
      resolved_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
   }

   struct d12_resource *res = calloc(1, sizeof(*res));
   if (!res)
      return E_OUTOFMEMORY;

   res->layout = layout;
   res->stencil_layout = stencil_layout;
   res->is_depth_stencil = is_depth_stencil;

   uint64_t stencil_offset_B = 0;

   if (reserved) {
      /* D3D12 tiles are 64 KiB while the Asahi VM page and hardware tile are
       * 16 KiB.  Reserve a 64 KiB-rounded VA and initially back it with the
       * driver's repeated scratch page, exactly the safe sparse foundation
       * used by Honeykrisp.  UpdateTileMappings will replace these pages. */
      uint64_t sparse_size_B;
      uint32_t sparse_tile_count = 0;
      if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         if (!d12_sparse_buffer_size(
                size_B, d12_core_get_caps()->gpu_virtual_address_bits,
                &sparse_size_B, &sparse_tile_count)) {
            free(res);
            return E_INVALIDARG;
         }
      } else {
         if (size_B > UINT64_MAX -
                         (D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES - 1)) {
            free(res);
            return E_INVALIDARG;
         }
         sparse_size_B =
            align64(size_B, D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
      }
      res->va = agx_va_alloc(&device->dev, sparse_size_B,
                             D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, 0, 0);
      if (!res->va) {
         free(res);
         return E_OUTOFMEMORY;
      }

      const uint32_t rw_flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE |
                                DRM_ASAHI_BIND_SINGLE_PAGE;
      int ret = agx_bo_bind(&device->dev, device->dev.scratch_bo,
                            res->va->addr, sparse_size_B, 0, rw_flags);
      if (!ret) {
         const uint32_t ro_flags =
            DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_SINGLE_PAGE;
         ret = agx_bo_bind(&device->dev, device->dev.zero_bo,
                           res->va->addr + device->dev.sparse_ro_offset,
                           sparse_size_B, 0, ro_flags);
      }
      if (ret) {
         agx_va_free(&device->dev, res->va, true);
         free(res);
         return E_OUTOFMEMORY;
      }

      res->reserved = true;
      res->sparse_size_B = sparse_size_B;
      if (!d12_resource_init_sparse_metadata(res, desc)) {
         agx_va_free(&device->dev, res->va, true);
         free(res);
         return E_OUTOFMEMORY;
      }
      if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
          res->sparse_tile_count != sparse_tile_count) {
         res->device = device;
         d12_resource_cleanup_sparse_storage(res);
         agx_va_free(&device->dev, res->va, true);
         free(res);
         return E_INVALIDARG;
      }

      uint64_t sparse_map_size_B = res->layout.sparse_table_size_B;
      if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
          !d12_sparse_buffer_map_size(res->sparse_tile_count, AIL_PAGESIZE,
                                      &sparse_map_size_B))
         sparse_map_size_B = 0;
      if (!sparse_map_size_B) {
         res->device = device;
         d12_resource_cleanup_sparse_storage(res);
         agx_va_free(&device->dev, res->va, true);
         free(res);
         return E_INVALIDARG;
      }
      res->sparse_map = agx_bo_create(
         &device->dev, sparse_map_size_B, AIL_PAGESIZE,
         AGX_BO_WRITEBACK, "d3d12 sparse residency map");
      res->sparse_map_size_B = sparse_map_size_B;
      {
         void *map = res->sparse_map ? agx_bo_map(res->sparse_map) : NULL;
         if (!map) {
            /* The normal release helper expects the device link, but object
             * publication happens later in this constructor. */
            res->device = device;
            d12_resource_cleanup_sparse_storage(res);
            agx_va_free(&device->dev, res->va, true);
            free(res);
            return E_OUTOFMEMORY;
         }
         memset(map, 0, sparse_map_size_B);
      }
   } else if (heap) {
      /*
       * PLACED: no allocation, a window into the heap's BO. The stencil
       * plane, when there is one, follows the depth plane inside the same
       * window -- D3D12 sizes the resource for both, which is what
       * GetResourceAllocationInfo reports.
       */
      uint64_t need = size_B;
      if (is_depth_stencil && stencil_layout.size_B) {
         stencil_offset_B = heap_offset_B + size_B;
         need += stencil_layout.size_B;
      }

      /*
       * The resource's REQUIRED alignment is part of whether it fits, not
       * just its raw size. desc->Alignment == 0 means "pick for me", which
       * for a multisampled resource is 4 MB -- so a heap sized from a
       * 64 KB-aligned query cannot hold it, and D3D12 refuses with
       * E_INVALIDARG.
       *
       * test_placed_msaa_alignment_workaround asserts exactly that, and
       * conditionally: it expects S_OK only from vkd3d-proton, which
       * deliberately allows what native rejects. We are not vkd3d-proton,
       * so the correct answer here is the refusal.
       */
      uint64_t required_align =
         desc->Alignment ? desc->Alignment
         : (desc->SampleDesc.Count > 1
               ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
               : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

      /*
       * The HEAP's alignment must be able to host the resource. A heap
       * created with Alignment == 0 is a 64 KB heap, and a resource that
       * requires 4 MB placement cannot go in one -- native D3D12 refuses
       * this even when the sizes happen to work out, which they do here
       * because the 64 KB- and 4 MB-aligned sizes of this texture
       * coincide. Checking only the size therefore accepted a placement
       * native rejects.
       */
      uint64_t heap_align = heap->desc.Alignment
                               ? heap->desc.Alignment
                               : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

      if (required_align > heap_align) {
         mesa_loge("CreatePlacedResource: resource needs %" PRIu64
                   "-byte placement, heap provides %" PRIu64,
                   required_align, heap_align);
         free(res);
         return E_INVALIDARG;
      }

      if (heap_offset_B % required_align) {
         mesa_loge("CreatePlacedResource: offset %" PRIu64 " is not %"
                   PRIu64 "-aligned", heap_offset_B, required_align);
         free(res);
         return E_INVALIDARG;
      }

      /*
       * Alignment constrains where the resource starts, not how much of the
       * heap its payload occupies.  In particular, D3D12 permits a 1 KiB
       * buffer at offset 0 in a 1 KiB heap even though the buffer's placement
       * alignment is 64 KiB.  Texture backing sizes already come from AIL;
       * rounding either kind here rejects an otherwise in-bounds resource.
       */

      if (heap_offset_B + need > heap->bo->size) {
         mesa_loge("CreatePlacedResource: %" PRIu64 " bytes at offset %"
                   PRIu64 " does not fit a %zu-byte heap",
                   need, heap_offset_B, (size_t)heap->bo->size);
         free(res);
         return E_INVALIDARG;
      }

      res->bo = heap->bo;
      res->stencil_bo = (is_depth_stencil && stencil_layout.size_B)
                           ? heap->bo : NULL;
   } else if (imported_bo) {
      uint64_t required_size = size_B + stencil_layout.size_B;
      if (imported_bo->size < required_size) {
         free(res);
         return E_INVALIDARG;
      }

      res->bo = imported_bo;
      agx_bo_reference(imported_bo);
      if (is_depth_stencil && stencil_layout.size_B) {
         stencil_offset_B = size_B;
         res->stencil_bo = imported_bo;
      }
   } else {
      /* UMA: every heap type is the same cached, coherent memory. */
      /* CREATE_NOT_ZEROED permits an implementation to omit the clear; it
       * does not require dirty contents.  Native Windows drivers clear in
       * practice, and applications rely on that behaviour even with the
       * hint set.  Fresh kernel BOs are already zeroed, so cover only the
       * userspace-cache reuse case here. */
      enum agx_bo_flags bo_flags =
         AGX_BO_WRITEBACK | AGX_BO_ZERO_ON_REUSE;

      HRESULT shared_hr = E_NOTIMPL;
      if (shared) {
         uint64_t allocation_size = size_B + stencil_layout.size_B;
         struct d12_d3dkmt_d3d12_desc metadata;
         d12_shared_metadata_init(&metadata, &resolved_desc,
                                  allocation_size);
         shared_hr = d12_platform_create_shared_bo(
            device, allocation_size, resolved_desc.Alignment, bo_flags,
            &metadata, sizeof(metadata), "d3d12 shared resource", &res->bo);
      }

      /* Linux retains its existing private-allocation behaviour. Its shared
       * handle methods remain E_NOTIMPL, while Windows must never silently
       * fall back to an unshareable BO after a real D3DKMT failure. */
      if (!shared || shared_hr == E_NOTIMPL)
         res->bo = agx_bo_create(&device->dev, size_B, 0, bo_flags,
                                 "d3d12 resource");
      if (!res->bo) {
         free(res);
         return shared && shared_hr != E_NOTIMPL ? shared_hr
                                                 : E_OUTOFMEMORY;
      }

      if (is_depth_stencil && stencil_layout.size_B) {
         if (shared_hr == S_OK) {
            stencil_offset_B = size_B;
            res->stencil_bo = res->bo;
         } else {
            res->stencil_bo =
               agx_bo_create(&device->dev, stencil_layout.size_B, 0,
                             bo_flags, "d3d12 stencil plane");
         }
         if (!res->stencil_bo) {
            agx_bo_unreference(&device->dev, res->bo);
            free(res);
            return E_OUTOFMEMORY;
         }
      }
   }

   /* A committed resource owns its whole BO, so it sits at offset 0. The
    * resolved addresses still go through these fields rather than being
    * read from the BO downstream, so a placed resource -- which does not --
    * needs no second code path. */
   if (reserved) {
      res->addr = res->va->addr;
      res->offset_B = 0;
      res->placed = false;
      res->stencil_addr = 0;
   } else if (heap) {
      res->addr = heap->bo->va->addr + heap_offset_B;
      res->offset_B = heap_offset_B;
      res->placed = true;
      res->placed_heap_desc = heap->desc;
      res->stencil_addr =
         res->stencil_bo ? heap->bo->va->addr + stencil_offset_B : 0;

      /*
       * Reference the STORAGE, not the heap object.
       *
       * The storage must outlive the heap -- D3D12 lets the application
       * release ID3D12Heap while placed resources are still alive -- but
       * the heap's PUBLIC refcount must not move: test_suballocate_small_
       * textures creates ten placed resources and then asserts
       * get_refcount(heap) == 1. Taking ID3D12Heap_AddRef here reported 11.
       *
       * agx_bo_reference is invisible to COM refcounting and does exactly
       * what is needed: the BO survives until the last placed resource
       * releases it.
       */
      agx_bo_reference(heap->bo);
   } else {
      res->addr = res->bo->va->addr;
      res->offset_B = 0;
      res->placed = false;
      res->stencil_addr =
         res->stencil_bo ? res->stencil_bo->va->addr + stencil_offset_B : 0;
   }

   res->iface.lpVtbl = (ID3D12ResourceVtbl *)&d12_resource_vtbl;
   res->refcount = 1;
   d12_private_data_store_init(&res->private_data, (IUnknown *)&res->iface);
   res->device = device;
   /* GetDesc reports the resolved description, including runtime-selected
    * mip count and placement alignment. The shared metadata carries this
    * same canonical spelling so another process reconstructs it exactly. */
   res->desc = resolved_desc;
   res->heap_props = *heap_props;
   res->heap_flags = heap_flags;
   if (reserved) {
      simple_mtx_lock(&device->sparse_resource_lock);
      list_addtail(&res->sparse_link, &device->sparse_resources);
      simple_mtx_unlock(&device->sparse_resource_lock);
   }
   d12_device_add_ref(device);

   HRESULT hr = d12_resource_QueryInterface(&res->iface, riid, out);
   d12_resource_Release(&res->iface);
   return hr;
}

uint64_t
d12_sparse_read_address(struct d12_device *device, uint64_t address)
{
   simple_mtx_lock(&device->sparse_resource_lock);
   list_for_each_entry(struct d12_resource, res, &device->sparse_resources,
                       sparse_link) {
      if (address >= res->addr && address - res->addr < res->sparse_size_B) {
         address += device->dev.sparse_ro_offset;
         break;
      }
   }
   simple_mtx_unlock(&device->sparse_resource_lock);
   return address;
}

struct d12_resource *
d12_sparse_buffer_for_address(struct d12_device *device, uint64_t address)
{
   if (!device)
      return NULL;

   if (agx_addr_is_ro(&device->dev, address))
      address -= device->dev.sparse_ro_offset;

   struct d12_resource *found = NULL;
   simple_mtx_lock(&device->sparse_resource_lock);
   list_for_each_entry(struct d12_resource, res, &device->sparse_resources,
                       sparse_link) {
      if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
          address >= res->addr && address - res->addr < res->sparse_size_B) {
         found = res;
         break;
      }
   }
   simple_mtx_unlock(&device->sparse_resource_lock);
   return found;
}

uint64_t
d12_sparse_buffer_residency_address(const struct d12_resource *res,
                                    uint64_t view_address)
{
   if (!res || !res->reserved || !res->sparse_map || !res->sparse_map->va ||
       res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
      return 0;

   if (agx_addr_is_ro(&res->device->dev, view_address))
      view_address -= res->device->dev.sparse_ro_offset;
   if (view_address < res->addr ||
       view_address - res->addr >= res->sparse_size_B)
      return 0;

   uint32_t index;
   if (!d12_sparse_buffer_residency_index(
          view_address - res->addr, res->sparse_size_B,
          res->sparse_tile_count, &index))
      return 0;
   const uint64_t byte_offset = (uint64_t)index * sizeof(uint32_t);
   if (byte_offset >= res->sparse_map_size_B ||
       res->sparse_map->va->addr > UINT64_MAX - byte_offset)
      return 0;
   return res->sparse_map->va->addr + byte_offset;
}

/*
 * D3D12 description validation, shared by the committed and placed paths.
 *
 * It lives in one function on purpose. These rules were originally written
 * only into CreateCommittedResource, and CreatePlacedResource discarded
 * initial_state and the clear value exactly as the committed path once did
 * -- the same silently-dropped-parameter shape, one call site along. A
 * second copy would drift the moment either is touched.
 */
static HRESULT
d12_validate_resource_desc(const D3D12_HEAP_PROPERTIES *heap_props,
                           const D3D12_RESOURCE_DESC *desc,
                           D3D12_RESOURCE_STATES initial_state,
                           const D3D12_CLEAR_VALUE *clear_value)
{
   if (!heap_props || !desc)
      return E_INVALIDARG;

   bool is_buffer = desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;

   /*
    * BUFFER-SPECIFIC RULES.
    *
    * A buffer has exactly one mip level -- 0 is not "resolve for me" here,
    * it is invalid -- and an optimized clear value is meaningless for one,
    * so passing a non-NULL clear value is an error rather than something to
    * ignore. Both are checked directly by the suite.
    */
   if (is_buffer) {
      if (desc->MipLevels != 1)
         return E_INVALIDARG;

      if (clear_value)
         return E_INVALIDARG;

      /* D3D12 oddly permits ALLOW_RENDER_TARGET on a buffer description,
       * but a buffer can never be a depth/stencil resource.
       */
      if (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
         return E_INVALIDARG;
   }

   /*
    * GPU_UPLOAD exists so a discrete GPU with a resizable BAR can be written
    * by the CPU without staging through system memory. On UMA that is what
    * every allocation already is, so it is accepted and validated exactly as
    * a DEFAULT heap: same resource shapes, same flags, same states. The only
    * differences are that it is CPU-mappable and that
    * GetCustomHeapProperties reports it write-combined in pool L0.
    */
   if (heap_props->Type == D3D12_HEAP_TYPE_GPU_UPLOAD &&
       !d12_core_get_caps()->gpu_upload_heap)
      return E_INVALIDARG;

   /*
    * FLAG / HEAP-TYPE COMBINATIONS. An UPLOAD or READBACK heap is CPU-side
    * staging: a resource there cannot also be a render target or a UAV.
    * ALLOW_SIMULTANEOUS_ACCESS is not meaningful for a buffer at all, on
    * any heap.
    */
   bool staging_heap = heap_props->Type == D3D12_HEAP_TYPE_UPLOAD ||
                       heap_props->Type == D3D12_HEAP_TYPE_READBACK;

   if (staging_heap &&
       (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)))
      return E_INVALIDARG;

   if (is_buffer && (desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS))
      return E_INVALIDARG;

   /*
    * A READBACK heap is a destination: the GPU writes, the CPU reads. Only
    * COPY_DEST (and COMMON, which newer runtimes accept) make sense, so a
    * read state on one is an error. UPLOAD is the mirror image and is
    * permissive about read states, which is why it is not checked here.
    */
   if (heap_props->Type == D3D12_HEAP_TYPE_READBACK &&
       initial_state != D3D12_RESOURCE_STATE_COMMON &&
       initial_state != D3D12_RESOURCE_STATE_COPY_DEST)
      return E_INVALIDARG;

   /*
    * A block-compressed format needs a 2D grid of blocks, so D3D12 rejects
    * it on a 1D texture. This is an INVALID description, not a dimension we
    * merely have not implemented -- answering E_NOTIMPL would tell the
    * caller to try again on better hardware when the request is illegal
    * everywhere.
    */
   /*
    * ROW_MAJOR TEXTURES. D3D12 permits a linear-laid-out texture only in a
    * narrow corner -- one mip, one layer, 2D only -- because it exists for
    * cross-adapter sharing rather than for general use. desc->Layout was
    * being ignored entirely, which is the same dropped-parameter shape as
    * initial_state and clear_value.
    */
   if (!is_buffer && desc->Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR) {
      if (desc->MipLevels != 1 || desc->DepthOrArraySize != 1 ||
          desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
          !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) ||
          (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) ||
          heap_props->Type != D3D12_HEAP_TYPE_CUSTOM ||
          heap_props->CPUPageProperty !=
             D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE)
         return E_INVALIDARG;
      enum pipe_format row_major_pipe = d12_format_to_pipe(desc->Format);
      if (row_major_pipe == PIPE_FORMAT_NONE ||
          util_format_get_blockwidth(row_major_pipe) != 1 ||
          util_format_get_blockheight(row_major_pipe) != 1 ||
          util_format_get_blocksize(row_major_pipe) > 16)
         return E_INVALIDARG;
   }

   if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
      enum pipe_format pf = d12_format_to_pipe(desc->Format);
      if (pf != PIPE_FORMAT_NONE && (util_format_get_blockwidth(pf) > 1 ||
                                     util_format_get_blockheight(pf) > 1))
         return E_INVALIDARG;
   }

   /*
    * D3D12 VALIDATION. These are refusals the runtime owes the caller, and
    * accepting them is not harmless leniency -- an application that gets
    * S_OK for an illegal description carries on and fails somewhere less
    * obvious. Each rule below is one the suite checks directly.
    */

   /* A sample count of zero is meaningless; 1 is the minimum. */
   if (desc->SampleDesc.Count == 0)
      return E_INVALIDARG;

   /* D3D12 has no sampled-only MSAA resource shape. A multisampled texture
    * must carry one of the output flags even when the application ultimately
    * creates an SRV for it. Keep this API rule separate from AGX's supported
    * sample-count test so unsupported counts retain their existing refusal. */
   if (!d12_resource_multisample_flags_valid(
          desc->SampleDesc.Count,
          desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
          desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
      return E_INVALIDARG;

   bool is_texture = !is_buffer;

   /*
    * Write states are exclusive: RENDER_TARGET, DEPTH_WRITE, UNORDERED_ACCESS,
    * COPY_DEST and STREAM_OUT cannot be combined with anything else.
    */
   const D3D12_RESOURCE_STATES write_states =
      D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE |
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_COPY_DEST |
      D3D12_RESOURCE_STATE_STREAM_OUT;

   D3D12_RESOURCE_STATES write_bits = initial_state & write_states;
   if (write_bits && (initial_state & ~write_bits))
      return E_INVALIDARG;
   if (write_bits & (write_bits - 1))
      return E_INVALIDARG;

   /* A state the resource was not created to support is illegal. */
   if ((initial_state & D3D12_RESOURCE_STATE_RENDER_TARGET) &&
       !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
      return E_INVALIDARG;

   if ((initial_state & (D3D12_RESOURCE_STATE_DEPTH_WRITE |
                         D3D12_RESOURCE_STATE_DEPTH_READ)) &&
       !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
      return E_INVALIDARG;

   if ((initial_state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) &&
       !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
      return E_INVALIDARG;

   /*
    * Heap type restricts what may live there. UPLOAD and READBACK are
    * buffer-only; GPU_UPLOAD is not, because it is a DEFAULT heap the CPU
    * can also write.
    */
   /*
    * An explicitly requested alignment must be one the resource can
    * actually use. 4 KB small placement is only legal when a single slice
    * of level 0 fits in 64 KB -- the same rule GetResourceAllocationInfo
    * applies, and it has to agree: a description the query refuses cannot
    * be one creation accepts.
    */
   if (is_texture &&
       desc->Alignment == D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) {
      enum pipe_format pf = d12_format_is_depth_stencil(desc->Format)
                               ? d12_format_depth_plane(desc->Format)
                               : d12_format_to_pipe(desc->Format);

      if (pf == PIPE_FORMAT_NONE || desc->SampleDesc.Count > 1 ||
          (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)))
         return E_INVALIDARG;

      uint64_t slice0 =
         (uint64_t)util_format_get_nblocksx(pf, MAX2((uint32_t)desc->Width, 1)) *
         util_format_get_nblocksy(pf, MAX2(desc->Height, 1)) *
         util_format_get_blocksize(pf);

      if (slice0 > D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT)
         return E_INVALIDARG;
   }

   if (is_texture && (heap_props->Type == D3D12_HEAP_TYPE_UPLOAD ||
                      heap_props->Type == D3D12_HEAP_TYPE_READBACK))
      return E_INVALIDARG;

   return S_OK;
}

HRESULT
d12_resource_create_committed(struct d12_device *device,
                              const D3D12_HEAP_PROPERTIES *heap_props,
                              D3D12_HEAP_FLAGS heap_flags,
                              const D3D12_RESOURCE_DESC *desc,
                              D3D12_RESOURCE_STATES initial_state,
                              const D3D12_CLEAR_VALUE *clear_value,
                              REFIID riid, void **out)
{
   HRESULT valid = d12_validate_resource_desc(heap_props, desc, initial_state,
                                              clear_value);
   if (FAILED(valid))
      return valid;

   return d12_resource_create(device, heap_props, heap_flags, desc, NULL, 0,
                              0, 0, false, NULL, riid, out);
}

HRESULT
d12_resource_create_placed(struct d12_device *device, struct d12_heap *heap,
                           uint64_t offset_B,
                           const D3D12_RESOURCE_DESC *desc,
                           uint32_t row_pitch_B, uint32_t slice_pitch_B,
                           D3D12_RESOURCE_STATES initial_state,
                           const D3D12_CLEAR_VALUE *clear_value, REFIID riid,
                           void **out)
{
   if (!heap || !desc)
      return E_INVALIDARG;

   /* The ALLOW_ONLY_* heap flags are aliases for combinations of these
    * three DENY flags. Validate the actual resource class before allocating
    * anything: placed resources do not get to cross a heap's declared type
    * boundary just because this UMA implementation uses the same BO kind
    * underneath.
    */
   bool is_buffer = desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
   bool is_rt_ds = !is_buffer &&
      (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));

   if ((is_buffer && (heap->desc.Flags & D3D12_HEAP_FLAG_DENY_BUFFERS)) ||
       (is_rt_ds &&
        (heap->desc.Flags & D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES)) ||
       (!is_buffer && !is_rt_ds &&
        (heap->desc.Flags & D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES)))
      return E_INVALIDARG;

   /* The same rules the committed path enforces. A placed resource is not
    * exempt from them just because its storage comes from elsewhere. */
   HRESULT valid = d12_validate_resource_desc(&heap->desc.Properties, desc,
                                              initial_state, clear_value);
   if (FAILED(valid))
      return valid;

   return d12_resource_create(device, &heap->desc.Properties,
                              heap->desc.Flags, desc, heap, offset_B,
                              row_pitch_B, slice_pitch_B, false, NULL, riid,
                              out);
}

HRESULT
d12_resource_create_reused(struct d12_device *device,
                           struct d12_resource *source,
                           uint64_t source_offset_B,
                           const D3D12_RESOURCE_DESC *desc,
                           uint32_t row_pitch_B, uint32_t slice_pitch_B,
                           D3D12_RESOURCE_STATES initial_state,
                           const D3D12_CLEAR_VALUE *clear_value, REFIID riid,
                           void **out)
{
   struct d12_heap captured_heap = {0};
   D3D12_RESOURCE_ALLOCATION_INFO allocation_info;
   uint64_t explicit_row_major_size_B = 0;
   uint64_t absolute_offset;

   if (!device || !source || source->device != device || !desc ||
       source->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       source->reserved || !source->bo || !source->bo->va ||
       source_offset_B > source->desc.Width ||
       source->offset_B > UINT64_MAX - source_offset_B)
      return E_INVALIDARG;

   d12_device_get_resource_allocation_info_runtime(device, 1, desc,
                                                    &allocation_info);
   if (!d12_resource_explicit_row_major_size(
          desc, row_pitch_B, slice_pitch_B, &explicit_row_major_size_B))
      return E_INVALIDARG;
   if (explicit_row_major_size_B) {
      if (!allocation_info.Alignment ||
          explicit_row_major_size_B >
             UINT64_MAX - (allocation_info.Alignment - 1))
         return E_INVALIDARG;
      allocation_info.SizeInBytes =
         (explicit_row_major_size_B + allocation_info.Alignment - 1) &
         ~(allocation_info.Alignment - 1);
   }
   if (allocation_info.SizeInBytes == UINT64_MAX ||
       allocation_info.SizeInBytes >
          source->desc.Width - source_offset_B)
      return E_INVALIDARG;

   absolute_offset = source->offset_B + source_offset_B;
   captured_heap.desc = source->placed
                           ? source->placed_heap_desc
                           : (D3D12_HEAP_DESC){
                                .SizeInBytes = source->bo->size,
                                .Properties = source->heap_props,
                                .Alignment = source->desc.Alignment,
                                .Flags = source->heap_flags,
                             };
   captured_heap.desc.SizeInBytes = source->bo->size;
   captured_heap.bo = source->bo;

   /* d12_resource_create_placed consumes the heap description and BO only
    * during this call, then retains the BO and copies the description. This
    * exactly models the DDI's capture-GPUVA lifetime without retaining the
    * public heap COM object. */
   return d12_resource_create_placed(device, &captured_heap, absolute_offset,
                                     desc, row_pitch_B, slice_pitch_B,
                                     initial_state, clear_value, riid, out);
}

HRESULT
d12_resource_validate_reserved_tier(const D3D12_RESOURCE_DESC *desc)
{
   if (!desc)
      return E_INVALIDARG;

   if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
         return E_INVALIDARG;
   } else {
      /* Tier 1 is limited to 2D textures with the undefined 64 KiB swizzle.
       * Standard swizzle and 1D/3D are separate capabilities. */
      if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
          desc->Layout != D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE ||
          desc->SampleDesc.Count != 1 || desc->SampleDesc.Quality != 0)
         return E_INVALIDARG;

      const bool depth = d12_format_is_depth_stencil(desc->Format);
      enum pipe_format pipe =
         depth ? d12_format_depth_plane(desc->Format)
               : d12_format_to_pipe(desc->Format);
      /* Multiplane depth/stencil needs independent sparse maps for each plane;
       * advertise the single-plane D16/D32 subset until that is implemented. */
      if (depth && d12_format_stencil_plane(desc->Format) != PIPE_FORMAT_NONE)
         return E_INVALIDARG;
      if (pipe == PIPE_FORMAT_NONE)
         return E_INVALIDARG;

      const unsigned major =
         MAX2((uint32_t)MAX2(desc->Width, 1), MAX2(desc->Height, 1));
      const unsigned levels =
         desc->MipLevels ? desc->MipLevels : util_logbase2(major) + 1;
      struct ail_layout layout = {
         .width_px = (uint32_t)desc->Width,
         .height_px = desc->Height,
         .depth_px = MAX2(desc->DepthOrArraySize, 1),
         .sample_count_sa = MAX2(desc->SampleDesc.Count, 1),
         .levels = levels,
         .tiling = AIL_TILING_GPU,
         .format = pipe,
         .writeable_image =
            desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      };
      ail_make_miptree(&layout);

      /* Tier 1 cannot represent a packed mip tail shared by array layers. */
      if (desc->DepthOrArraySize > 1 && layout.mip_tail_first_lod < levels)
         return E_INVALIDARG;
   }

   return S_OK;
}

HRESULT
d12_resource_create_reserved(struct d12_device *device,
                             const D3D12_RESOURCE_DESC *desc,
                             D3D12_RESOURCE_STATES initial_state,
                             const D3D12_CLEAR_VALUE *clear_value, REFIID riid,
                             void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!desc)
      return E_INVALIDARG;

   const D3D12_HEAP_PROPERTIES props = {
      .Type = D3D12_HEAP_TYPE_DEFAULT,
      .CreationNodeMask = 1,
      .VisibleNodeMask = 1,
   };

   /* Reserved buffers may begin in UAV state without ALLOW_UNORDERED_ACCESS;
    * that flag controls whether a UAV view can be created, while sparse root
    * descriptors still use the resource as an addressable buffer. */
   D3D12_RESOURCE_DESC validation_desc = *desc;
   if (initial_state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
      validation_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
   HRESULT valid = d12_validate_resource_desc(&props, &validation_desc,
                                              initial_state, clear_value);
   if (FAILED(valid))
      return valid;
   valid = d12_resource_validate_reserved_tier(desc);
   if (FAILED(valid))
      return valid;

   return d12_resource_create(device, &props, D3D12_HEAP_FLAG_NONE, desc,
                              NULL, 0, 0, 0, true, NULL, riid, out);
}

HRESULT
d12_resource_create_shared_handle(
   struct d12_device *device, ID3D12DeviceChild *object,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle)
{
   if (!handle)
      return E_POINTER;
   *handle = NULL;
   if (!object)
      return E_INVALIDARG;

   ID3D12Resource *resource_iface = NULL;
   HRESULT hr = ID3D12DeviceChild_QueryInterface(
      object, &IID_ID3D12Resource, (void **)&resource_iface);
   if (FAILED(hr)) {
      ID3D12Heap *heap_iface = NULL;
      hr = ID3D12DeviceChild_QueryInterface(
         object, &IID_ID3D12Heap, (void **)&heap_iface);
      if (FAILED(hr))
         return E_NOTIMPL;

      if (!d12_heap_iface_is_ours(heap_iface)) {
         ID3D12Heap_Release(heap_iface);
         return DXGI_ERROR_INVALID_CALL;
      }

      struct d12_heap *heap = d12_memheap_from_iface(heap_iface);
      if (heap->device != device ||
          !(heap->desc.Flags & D3D12_HEAP_FLAG_SHARED)) {
         ID3D12Heap_Release(heap_iface);
         return DXGI_ERROR_INVALID_CALL;
      }
      hr = d12_platform_export_shared_bo(device, heap->bo, attributes,
                                         access, name, handle);
      ID3D12Heap_Release(heap_iface);
      return hr;
   }

   if (!d12_resource_iface_is_ours(resource_iface)) {
      ID3D12Resource_Release(resource_iface);
      return DXGI_ERROR_INVALID_CALL;
   }

   struct d12_resource *resource =
      d12_resource_from_iface(resource_iface);
   if (resource->device != device || resource->placed ||
       !(resource->heap_flags & D3D12_HEAP_FLAG_SHARED)) {
      ID3D12Resource_Release(resource_iface);
      return DXGI_ERROR_INVALID_CALL;
   }

   hr = d12_platform_export_shared_bo(device, resource->bo, attributes,
                                      access, name, handle);
   ID3D12Resource_Release(resource_iface);
   return hr;
}

HRESULT
d12_resource_open_shared_handle(struct d12_device *device, HANDLE handle,
                                REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!riid || d12_handle_is_invalid(handle))
      return E_INVALIDARG;

   struct d12_d3dkmt_d3d12_desc metadata = {0};
   uint32_t metadata_size = sizeof(metadata);
   struct agx_bo *bo = NULL;
   HRESULT hr = d12_platform_import_shared_bo(
      device, handle, &metadata, &metadata_size,
      "d3d12 imported shared resource", &bo);
   if (FAILED(hr))
      return hr;

   D3D12_RESOURCE_DESC desc;
   if (!d12_shared_metadata_parse(&metadata, metadata_size, &desc)) {
      agx_bo_unreference(&device->dev, bo);
      return E_INVALIDARG;
   }

   const D3D12_HEAP_PROPERTIES heap_props = {
      .Type = D3D12_HEAP_TYPE_DEFAULT,
      .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
      .CreationNodeMask = 1,
      .VisibleNodeMask = 1,
   };
   hr = d12_validate_resource_desc(&heap_props, &desc,
                                   D3D12_RESOURCE_STATE_COMMON, NULL);
   if (SUCCEEDED(hr)) {
      hr = d12_resource_create(device, &heap_props,
                               D3D12_HEAP_FLAG_SHARED, &desc, NULL, 0, 0, 0,
                               false, bo, riid, out);
   }

   /* The constructor takes its own BO reference only after it has validated
    * and adopted the allocation. This drops the import helper's reference. */
   agx_bo_unreference(&device->dev, bo);
   return hr;
}


/*
 * ID3D12Heap. One agx_bo; on UMA every heap type is the same memory, so a
 * heap is an allocation and nothing more. Placed resources carve windows
 * out of it and hold a reference, because D3D12 lets the application
 * release the heap while they are still alive.
 */

static HRESULT STDMETHODCALLTYPE
d12_heap_QueryInterface(ID3D12Heap *iface, REFIID riid, void **out)
{
   struct d12_heap *heap = d12_memheap_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12Heap)) {
      p_atomic_inc(&heap->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&heap->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12Heap::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_heap_AddRef(ID3D12Heap *iface)
{
   return p_atomic_inc_return(&d12_memheap_from_iface(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_heap_Release(ID3D12Heap *iface)
{
   struct d12_heap *heap = d12_memheap_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&heap->refcount);

   if (refs == 0) {
      agx_bo_unreference(&heap->device->dev, heap->bo);
      d12_private_data_store_destroy(&heap->private_data);
      d12_device_release(heap->device);
      free(heap);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_heap, ID3D12Heap,
                                d12_memheap_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_heap_GetDevice(ID3D12Heap *iface, REFIID riid, void **out)
{
   struct d12_heap *heap = d12_memheap_from_iface(iface);
   return ID3D12Device_QueryInterface(&heap->device->iface, riid, out);
}

static D12_AGGREGATE_RETURN(D3D12_HEAP_DESC) STDMETHODCALLTYPE
d12_heap_GetDesc(ID3D12Heap *iface D12_AGGREGATE_OUT(D3D12_HEAP_DESC))
{
   D12_RETURN_AGGREGATE(d12_memheap_from_iface(iface)->desc);
}

static const ID3D12HeapVtbl d12_heap_vtbl = {
   .QueryInterface = d12_heap_QueryInterface,
   .AddRef = d12_heap_AddRef,
   .Release = d12_heap_Release,
   .GetPrivateData = d12_heap_GetPrivateData,
   .SetPrivateData = d12_heap_SetPrivateData,
   .SetPrivateDataInterface = d12_heap_SetPrivateDataInterface,
   .SetName = d12_heap_SetName,
   .GetDevice = d12_heap_GetDevice,
   .GetDesc = d12_heap_GetDesc,
};

bool
d12_heap_iface_is_ours(ID3D12Heap *iface)
{
   return iface && iface->lpVtbl == &d12_heap_vtbl;
}

static HRESULT
d12_heap_create_internal(struct d12_device *device,
                         const D3D12_HEAP_DESC *desc,
                         struct agx_bo *imported_bo, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (!d12_heap_desc_valid(desc, d12_core_get_caps()->gpu_upload_heap))
      return E_INVALIDARG;
   if (!out)
      return S_FALSE;

   /* Apple UMA has no separate VRAM pool. Match Honeykrisp's Asahi policy:
    * expose at most half of physical RAM to one GPU heap so allocation
    * cannot force the machine into swap or OOM before D3D12 can report a
    * normal resource-allocation failure. */
   uint64_t system_memory;
   if (os_get_total_physical_memory(&system_memory) &&
       desc->SizeInBytes > system_memory / 2)
      return E_OUTOFMEMORY;

   D3D12_HEAP_DESC resolved = *desc;

   /* A zero node mask selects node 0 on this single-node device. Store the
    * canonical spelling returned by the native runtime rather than echoing
    * the caller's shorthand from GetDesc().
    */
   if (!resolved.Properties.CreationNodeMask)
      resolved.Properties.CreationNodeMask = 1;
   if (!resolved.Properties.VisibleNodeMask)
      resolved.Properties.VisibleNodeMask = 1;

   if (!resolved.Alignment)
      resolved.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

   struct d12_heap *heap = calloc(1, sizeof(*heap));
   if (!heap)
      return E_OUTOFMEMORY;

   /* As with committed resources, conservatively clear a cached BO even
    * when CREATE_NOT_ZEROED allows us to skip it. */
   enum agx_bo_flags bo_flags = AGX_BO_WRITEBACK | AGX_BO_ZERO_ON_REUSE;

   if (imported_bo) {
      if (imported_bo->size < resolved.SizeInBytes) {
         free(heap);
         return E_INVALIDARG;
      }
      heap->bo = imported_bo;
      agx_bo_reference(imported_bo);
   } else {
      HRESULT shared_hr = E_NOTIMPL;
      if (resolved.Flags & D3D12_HEAP_FLAG_SHARED) {
         struct d12_d3dkmt_heap_desc metadata;
         d12_shared_heap_metadata_init(&metadata, &resolved);
         shared_hr = d12_platform_create_shared_bo(
            device, resolved.SizeInBytes, resolved.Alignment, bo_flags,
            &metadata, sizeof(metadata), "d3d12 shared heap", &heap->bo);
      }
      if (!(resolved.Flags & D3D12_HEAP_FLAG_SHARED) ||
          shared_hr == E_NOTIMPL) {
         heap->bo = agx_bo_create(&device->dev, resolved.SizeInBytes,
                                  resolved.Alignment, bo_flags,
                                  "d3d12 heap");
      }
      if (!heap->bo && shared_hr != E_NOTIMPL) {
         free(heap);
         return shared_hr;
      }
   }
   if (!heap->bo) {
      free(heap);
      return E_OUTOFMEMORY;
   }

   heap->iface.lpVtbl = (ID3D12HeapVtbl *)&d12_heap_vtbl;
   heap->refcount = 1;
   heap->device = device;
   heap->desc = resolved;
   d12_private_data_store_init(&heap->private_data, (IUnknown *)&heap->iface);
   d12_device_add_ref(device);

   HRESULT hr = d12_heap_QueryInterface(&heap->iface, riid, out);
   d12_heap_Release(&heap->iface);
   return hr;
}

HRESULT
d12_heap_create(struct d12_device *device, const D3D12_HEAP_DESC *desc,
                REFIID riid, void **out)
{
   return d12_heap_create_internal(device, desc, NULL, riid, out);
}

HRESULT
d12_heap_create_from_bo(struct d12_device *device,
                        const D3D12_HEAP_DESC *desc, struct agx_bo *bo,
                        REFIID riid, void **out)
{
   if (!bo)
      return E_INVALIDARG;
   return d12_heap_create_internal(device, desc, bo, riid, out);
}

HRESULT
d12_heap_open_shared_handle(struct d12_device *device, HANDLE handle,
                            REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!riid || d12_handle_is_invalid(handle))
      return E_INVALIDARG;

   uint8_t metadata[sizeof(struct d12_d3dkmt_d3d12_desc)] = {0};
   uint32_t metadata_size = sizeof(metadata);
   struct agx_bo *bo = NULL;
   HRESULT hr = d12_platform_import_shared_bo(
      device, handle, metadata, &metadata_size, "d3d12 imported shared heap",
      &bo);
   if (FAILED(hr))
      return hr;

   D3D12_HEAP_DESC desc;
   if (!d12_shared_heap_metadata_parse(metadata, metadata_size, &desc)) {
      agx_bo_unreference(&device->dev, bo);
      return E_INVALIDARG;
   }

   hr = d12_heap_create_internal(device, &desc, bo, riid, out);
   agx_bo_unreference(&device->dev, bo);
   return hr;
}
