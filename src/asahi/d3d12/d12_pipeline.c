/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12PipelineState and ID3D12RootSignature.
 *
 * This is where the DXIL front end meets the runtime: CreateComputePipeline
 * State takes the application's DXIL bytecode, compiles it to AGX machine
 * code through our own translator (no SPIR-V, no external compiler), and
 * uploads the result to GPU memory ready to dispatch.
 */

#include "d12_private.h"
#include "d12_blend_core.h"

#include "agx_linker.h"
#include "asahi/lib/agx3_code_container_core.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "asahi/lib/agx_ioto_program_core.h"

#include "agx_compile.h"
#include "agx_debug.h"
#include "d12_nir_g17p_txf.h"
#include "agx_helpers.h"
#include "agx_nir.h"
#include "disasm.h"
#include "dxil/d12_dxil.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_xfb_info.h"
#include "poly/nir/poly_nir.h"

/* The runtime mirrors Honeykrisp's helper-program scratch allocation. Keep a
 * targeted dump for verifying which compiled stages exercise that path. */
static void
d12_report_scratch(const char *what, const struct agx_shader_info *info)
{
   if (d12_debug()->dump_scratch &&
       (info->scratch_size || info->preamble_scratch_size))
      mesa_logi("%s uses scratch (%u dwords main, %u dwords preamble)", what,
                info->scratch_size, info->preamble_scratch_size);
}

bool
d12_upload_executable(struct agx_bo *bo, const void *binary, size_t size)
{
   void *mapping;

   if (!bo || !binary || !size || !(mapping = agx_bo_map(bo)))
      return false;
   memcpy(mapping, binary, size);

   /* Lock2 mappings may persist for the PSO lifetime. Establish explicit
    * CPU-write -> GPU-read ownership after every compiler upload rather than
    * relying on an eventual unmap or on cache coincidence. */
   return agx_bo_sync_for_device(bo, 0, size) == 0;
}

/*
 * Root signature. D3D12 requires the object; our binding model puts
 * resource addresses directly in uniform registers (see d12_dxil.h), so
 * there is no state to derive from the blob yet.
 */

/* agx_fast_link only reads its parts; cached parts are shared and const. */
static bool
d12_link(struct agx_linked_shader *linked, struct agx_device *dev,
         bool fragment, struct agx_shader_part *main,
         const struct agx_shader_part *prolog,
         const struct agx_shader_part *epilog, unsigned nr_samples_shaded)
{
   return agx_fast_link(linked, dev, fragment, main,
                        (struct agx_shader_part *)prolog,
                        (struct agx_shader_part *)epilog, nr_samples_shaded);
}

static struct d12_root_signature *
d12_root_sig_from_iface(ID3D12RootSignature *iface)
{
   return (struct d12_root_signature *)iface;
}

static struct d12_private_data_entry **
d12_private_data_find(struct d12_private_data_store *store, REFGUID guid)
{
   struct d12_private_data_entry **entry = &store->entries;

   while (*entry && !d12_guid_equal(&(*entry)->guid, guid))
      entry = &(*entry)->next;

   return entry;
}

static void
d12_private_data_entry_destroy(struct d12_private_data_entry *entry)
{
   if (entry->is_interface && entry->object)
      IUnknown_Release(entry->object);
   free(entry->bytes);
   free(entry);
}

/* DirectX-Guids in the current build does not export this d3dcommon IID. */
static const GUID d12_iid_destruction_notifier = {
   0xa06eb39a, 0x50da, 0x425b,
   {0x8c, 0x31, 0x4e, 0xec, 0xd6, 0xc2, 0x70, 0xf3},
};

static inline struct d12_destruction_notifier *
d12_notifier_from_iface(ID3DDestructionNotifier *iface)
{
   return (struct d12_destruction_notifier *)iface;
}

static HRESULT STDMETHODCALLTYPE
d12_notifier_QueryInterface(ID3DDestructionNotifier *iface, REFIID riid,
                            void **out)
{
   struct d12_destruction_notifier *notifier = d12_notifier_from_iface(iface);
   return IUnknown_QueryInterface(notifier->parent, riid, out);
}

static ULONG STDMETHODCALLTYPE
d12_notifier_AddRef(ID3DDestructionNotifier *iface)
{
   struct d12_destruction_notifier *notifier = d12_notifier_from_iface(iface);
   return IUnknown_AddRef(notifier->parent);
}

static ULONG STDMETHODCALLTYPE
d12_notifier_Release(ID3DDestructionNotifier *iface)
{
   struct d12_destruction_notifier *notifier = d12_notifier_from_iface(iface);
   return IUnknown_Release(notifier->parent);
}

static HRESULT STDMETHODCALLTYPE
d12_notifier_RegisterDestructionCallback(ID3DDestructionNotifier *iface,
                                         PFN_DESTRUCTION_CALLBACK callback,
                                         void *data, UINT *callback_id)
{
   struct d12_destruction_notifier *notifier = d12_notifier_from_iface(iface);

   if (!callback)
      return DXGI_ERROR_INVALID_CALL;

   simple_mtx_lock(&notifier->lock);

   if (notifier->notified) {
      simple_mtx_unlock(&notifier->lock);
      return DXGI_ERROR_INVALID_CALL;
   }

   if (notifier->callback_count == notifier->callback_capacity) {
      uint32_t capacity = notifier->callback_capacity
                              ? 2 * notifier->callback_capacity
                              : 4;
      struct d12_destruction_callback *callbacks =
         realloc(notifier->callbacks, capacity * sizeof(*callbacks));

      if (!callbacks) {
         simple_mtx_unlock(&notifier->lock);
         return E_OUTOFMEMORY;
      }

      notifier->callbacks = callbacks;
      notifier->callback_capacity = capacity;
   }

   struct d12_destruction_callback *entry =
      &notifier->callbacks[notifier->callback_count++];
   entry->callback = callback;
   entry->data = data;
   entry->id = 0;

   if (callback_id) {
      if (!(++notifier->next_callback_id))
         ++notifier->next_callback_id;

      entry->id = notifier->next_callback_id;
      *callback_id = entry->id;
   }

   simple_mtx_unlock(&notifier->lock);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_notifier_UnregisterDestructionCallback(ID3DDestructionNotifier *iface,
                                           UINT callback_id)
{
   struct d12_destruction_notifier *notifier = d12_notifier_from_iface(iface);

   if (!callback_id)
      return DXGI_ERROR_NOT_FOUND;

   simple_mtx_lock(&notifier->lock);

   for (uint32_t i = 0; i < notifier->callback_count; ++i) {
      if (notifier->callbacks[i].id != callback_id)
         continue;

      memmove(&notifier->callbacks[i], &notifier->callbacks[i + 1],
              (notifier->callback_count - i - 1) *
                 sizeof(notifier->callbacks[0]));
      --notifier->callback_count;
      simple_mtx_unlock(&notifier->lock);
      return S_OK;
   }

   simple_mtx_unlock(&notifier->lock);
   return DXGI_ERROR_NOT_FOUND;
}

static const ID3DDestructionNotifierVtbl d12_notifier_vtbl = {
   .QueryInterface = d12_notifier_QueryInterface,
   .AddRef = d12_notifier_AddRef,
   .Release = d12_notifier_Release,
   .RegisterDestructionCallback = d12_notifier_RegisterDestructionCallback,
   .UnregisterDestructionCallback = d12_notifier_UnregisterDestructionCallback,
};

static void
d12_notifier_notify_and_destroy(struct d12_destruction_notifier *notifier)
{
   simple_mtx_lock(&notifier->lock);
   struct d12_destruction_callback *callbacks = notifier->callbacks;
   uint32_t callback_count = notifier->callback_count;
   notifier->callbacks = NULL;
   notifier->callback_count = 0;
   notifier->callback_capacity = 0;
   notifier->notified = true;
   simple_mtx_unlock(&notifier->lock);

   for (uint32_t i = 0; i < callback_count; ++i)
      callbacks[i].callback(callbacks[i].data);

   free(callbacks);
   simple_mtx_destroy(&notifier->lock);
}

void
d12_private_data_store_init(struct d12_private_data_store *store,
                            IUnknown *parent)
{
   simple_mtx_init(&store->lock, mtx_plain);
   store->destruction_notifier.iface.lpVtbl =
      (ID3DDestructionNotifierVtbl *)&d12_notifier_vtbl;
   store->destruction_notifier.parent = parent;
   simple_mtx_init(&store->destruction_notifier.lock, mtx_plain);
}

bool
d12_private_data_query_notifier(struct d12_private_data_store *store,
                                REFIID riid, void **out)
{
   if (!d12_guid_equal(riid, &d12_iid_destruction_notifier))
      return false;

   d12_notifier_AddRef(&store->destruction_notifier.iface);
   *out = &store->destruction_notifier.iface;
   return true;
}

void
d12_private_data_store_destroy(struct d12_private_data_store *store)
{
   d12_notifier_notify_and_destroy(&store->destruction_notifier);

   simple_mtx_lock(&store->lock);
   struct d12_private_data_entry *entry = store->entries;
   store->entries = NULL;
   simple_mtx_unlock(&store->lock);

   while (entry) {
      struct d12_private_data_entry *next = entry->next;
      d12_private_data_entry_destroy(entry);
      entry = next;
   }

   simple_mtx_destroy(&store->lock);
}

static HRESULT
d12_private_data_replace(struct d12_private_data_store *store, REFGUID guid,
                         struct d12_private_data_entry *replacement)
{
   simple_mtx_lock(&store->lock);
   struct d12_private_data_entry **slot = d12_private_data_find(store, guid);
   struct d12_private_data_entry *old = *slot;

   if (replacement) {
      replacement->next = old ? old->next : NULL;
      *slot = replacement;
   } else if (old) {
      *slot = old->next;
   }

   simple_mtx_unlock(&store->lock);

   if (old)
      d12_private_data_entry_destroy(old);

   return old || replacement ? S_OK : S_FALSE;
}

HRESULT
d12_private_data_set(struct d12_private_data_store *store, REFGUID guid,
                     UINT size, const void *data)
{
   if (!size || !data)
      return d12_private_data_replace(store, guid, NULL);

   struct d12_private_data_entry *entry = calloc(1, sizeof(*entry));
   if (!entry)
      return E_OUTOFMEMORY;

   entry->bytes = malloc(size);
   if (!entry->bytes) {
      free(entry);
      return E_OUTOFMEMORY;
   }

   entry->guid = *guid;
   entry->size = size;
   memcpy(entry->bytes, data, size);
   return d12_private_data_replace(store, guid, entry);
}

HRESULT
d12_private_data_set_interface(struct d12_private_data_store *store,
                               REFGUID guid, const IUnknown *object)
{
   struct d12_private_data_entry *entry = calloc(1, sizeof(*entry));
   if (!entry)
      return E_OUTOFMEMORY;

   entry->guid = *guid;
   entry->size = sizeof(entry->object);
   entry->object = (IUnknown *)object;
   entry->is_interface = true;
   if (entry->object)
      IUnknown_AddRef(entry->object);
   return d12_private_data_replace(store, guid, entry);
}

HRESULT
d12_private_data_get(struct d12_private_data_store *store, REFGUID guid,
                     UINT *size, void *data)
{
   if (!size)
      return E_INVALIDARG;

   simple_mtx_lock(&store->lock);
   struct d12_private_data_entry **slot = d12_private_data_find(store, guid);
   struct d12_private_data_entry *entry = *slot;

   if (!entry) {
      *size = 0;
      simple_mtx_unlock(&store->lock);
      return DXGI_ERROR_NOT_FOUND;
   }

   UINT requested_size = *size;
   *size = entry->size;

   if (!data) {
      simple_mtx_unlock(&store->lock);
      return S_OK;
   }

   if (requested_size < entry->size) {
      simple_mtx_unlock(&store->lock);
      return DXGI_ERROR_MORE_DATA;
   }

   if (entry->is_interface) {
      if (entry->object)
         IUnknown_AddRef(entry->object);
      *(IUnknown **)data = entry->object;
   } else {
      memcpy(data, entry->bytes, entry->size);
   }

   simple_mtx_unlock(&store->lock);
   return S_OK;
}

HRESULT
d12_private_data_set_name(struct d12_private_data_store *store, LPCWSTR name)
{
   /* wsl's host stub typedefs WCHAR as native wchar_t (32-bit on Linux),
    * while the D3D12 ABI and callers use 16-bit UTF-16 code units.  Treat the
    * incoming pointer as wire-format UTF-16; sizeof(*name) would report four
    * and read pairs of code units at a time on this host. */
   const uint16_t *utf16 = (const uint16_t *)name;
   size_t length = 0;
   if (!utf16)
      return E_INVALIDARG;
   while (utf16[length])
      length++;

   return d12_private_data_set(store, &WKPDID_D3DDebugObjectNameW,
                               (length + 1) * sizeof(*utf16), utf16);
}

static HRESULT STDMETHODCALLTYPE
d12_root_sig_QueryInterface(ID3D12RootSignature *iface, REFIID riid,
                            void **out)
{
   struct d12_root_signature *rs = d12_root_sig_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12RootSignature)) {
      p_atomic_inc(&rs->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&rs->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12RootSignature::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_root_sig_AddRef(ID3D12RootSignature *iface)
{
   return p_atomic_inc_return(&d12_root_sig_from_iface(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_root_sig_Release(ID3D12RootSignature *iface)
{
   struct d12_root_signature *rs = d12_root_sig_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&rs->refcount);

   if (refs == 0) {
      d12_private_data_store_destroy(&rs->private_data);
      free(rs->static_samplers);
      d12_root_layout_cleanup(&rs->layout);
      d12_device_release(rs->device);
      free(rs);
   }

   return refs;
}

static void
d12_pack_static_sampler(struct d12_device *device, struct d12_descriptor *out,
                        const D3D12_STATIC_SAMPLER_DESC1 *in)
{
   D3D12_SAMPLER_DESC2 desc = {
      .Filter = in->Filter,
      .AddressU = in->AddressU,
      .AddressV = in->AddressV,
      .AddressW = in->AddressW,
      .MipLODBias = in->MipLODBias,
      .MaxAnisotropy = in->MaxAnisotropy,
      .ComparisonFunc = in->ComparisonFunc,
      .MinLOD = in->MinLOD,
      .MaxLOD = in->MaxLOD,
      .Flags = in->Flags,
   };

   bool uint_border =
      (desc.Flags & D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR) ||
      in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK_UINT ||
      in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT;

   if (uint_border) {
      desc.Flags |= D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
      if (in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK ||
          in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK_UINT)
         desc.UintBorderColor[3] = 1;
      else if (in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE ||
               in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT)
         for (unsigned c = 0; c < 4; c++)
            desc.UintBorderColor[c] = 1;
   } else {
      if (in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK)
         desc.FloatBorderColor[3] = 1.0f;
      else if (in->BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE)
         for (unsigned c = 0; c < 4; c++)
            desc.FloatBorderColor[c] = 1.0f;
   }

   d12_pack_sampler_descriptor(device, out, &desc);
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_root_sig, ID3D12RootSignature,
                                d12_root_sig_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_root_sig_GetDevice(ID3D12RootSignature *iface, REFIID riid, void **out)
{
   struct d12_root_signature *rs = d12_root_sig_from_iface(iface);
   return ID3D12Device_QueryInterface(&rs->device->iface, riid, out);
}

static const ID3D12RootSignatureVtbl d12_root_sig_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12RootSignature_DEFAULTS,
   .QueryInterface = d12_root_sig_QueryInterface,
   .AddRef = d12_root_sig_AddRef,
   .Release = d12_root_sig_Release,
   .GetPrivateData = d12_root_sig_GetPrivateData,
   .SetPrivateData = d12_root_sig_SetPrivateData,
   .SetPrivateDataInterface = d12_root_sig_SetPrivateDataInterface,
   .SetName = d12_root_sig_SetName,
   .GetDevice = d12_root_sig_GetDevice,
};

HRESULT
d12_root_signature_create(struct d12_device *device, const void *blob,
                          size_t blob_size, REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!blob || !blob_size)
      return E_INVALIDARG;

   struct d12_root_signature *rs = calloc(1, sizeof(*rs));
   if (!rs)
      return E_OUTOFMEMORY;

   /* The blob is parsed, not ignored. It is the only description of which
    * shader register each root parameter feeds, and accepting it without
    * reading it means every later binding is a guess.
    */
   if (!d12_root_signature_parse(blob, blob_size, &rs->layout)) {
      free(rs);
      /* A valid shader container may simply have no embedded root signature.
       * Native D3D12 distinguishes that case from a malformed argument with
       * E_FAIL (the conformance suite pins the otherwise surprising result).
       */
      return d12_dxbc_is_dxbc_shader(blob, blob_size) ? E_FAIL : E_INVALIDARG;
   }

   if (!d12_root_layout_build_map(&rs->layout, &rs->map)) {
      d12_root_layout_cleanup(&rs->layout);
      free(rs);
      return E_INVALIDARG;
   }
   rs->may_write_uavs = (rs->layout.flags &
                         D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) != 0;
   for (unsigned i = 0; i < rs->map.count; i++)
      rs->may_write_uavs |= rs->map.entries[i].klass == D12_RES_UAV;

   if (rs->layout.num_static_samplers) {
      rs->static_samplers = calloc(rs->layout.num_static_samplers,
                                   sizeof(*rs->static_samplers));
      if (!rs->static_samplers) {
         d12_root_layout_cleanup(&rs->layout);
         free(rs);
         return E_OUTOFMEMORY;
      }

      for (uint32_t i = 0; i < rs->layout.num_static_samplers; i++) {
         const D3D12_STATIC_SAMPLER_DESC1 *s =
            &rs->layout.static_samplers[i];
         if (d12_binding_map_find_root_range(
                &rs->map,
                d12_static_sampler_root_range(&rs->layout, i),
                s->ShaderRegister) == ~0u) {
            free(rs->static_samplers);
            d12_root_layout_cleanup(&rs->layout);
            free(rs);
            return E_INVALIDARG;
         }
         d12_pack_static_sampler(device, &rs->static_samplers[i], s);

         /* G17P has no USC sampler words and no shader-side sampler state:
          * its bounded native fragment sample publishes the packed HAL200
          * word itself. Record which static samplers that word describes
          * completely, so the compile can leave their side-table emulation
          * out. The HAL200 word carries the LOD bias and every static border
          * colour natively; only MINIMUM/MAXIMUM reduction still needs the
          * shader. */
         const unsigned slot = d12_binding_map_find_root_range(
            &rs->map, d12_static_sampler_root_range(&rs->layout, i),
            s->ShaderRegister);
         if (device->dev.chip == AGX_CHIP_G17P && slot < 16 &&
             agx_chip_sampler_generation(device->dev.chip) ==
                AGX_SAMPLER_GENERATION_GEN4_HAL200 &&
             d12_sampler_reduction_for_filter((uint32_t)s->Filter)
                   .reduction == D12_SAMPLER_REDUCE_NONE &&
             !(rs->static_samplers[i].sampler_reserved &
               (D12_CUSTOM_BORDER_ENABLE | D12_SAMPLER_REDUCTION_MIN |
                D12_SAMPLER_REDUCTION_MAX)))
            rs->map.plain_static_samplers |= BITFIELD_BIT(slot);
      }
   }

   rs->iface.lpVtbl = (ID3D12RootSignatureVtbl *)d12_api_trace_vtbl(
      D12_API_TRACE_ROOT_SIGNATURE, &d12_root_sig_vtbl);
   rs->refcount = 1;
   rs->device = device;
   d12_private_data_store_init(&rs->private_data, (IUnknown *)&rs->iface);
   d12_device_add_ref(device);

   HRESULT hr = d12_root_sig_QueryInterface(&rs->iface, riid, out);
   d12_root_sig_Release(&rs->iface);
   return hr;
}

/*
 * Pipeline state
 */

static struct d12_pipeline_state *
d12_pso_from_iface(ID3D12PipelineState *iface)
{
   return (struct d12_pipeline_state *)iface;
}

static ULONG
d12_pso_add_ref(struct d12_pipeline_state *pso)
{
   ULONG refs = p_atomic_inc_return(&pso->refcount);

   /* A library-owned PSO can have no public references while its compiled
    * object and private-data store remain cached. Reacquiring it starts a
    * new public epoch, which holds one lifetime reference. The library's
    * own reference is what keeps the object alive across this call. */
   if (refs == 1) {
      p_atomic_inc(&pso->lifetime);
      d12_device_add_ref(pso->device);
   }

   return refs;
}

static HRESULT STDMETHODCALLTYPE
d12_pso_QueryInterface(ID3D12PipelineState *iface, REFIID riid, void **out)
{
   struct d12_pipeline_state *pso = d12_pso_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12PipelineState)) {
      d12_pso_add_ref(pso);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&pso->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12PipelineState::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_pso_AddRef(ID3D12PipelineState *iface)
{
   return d12_pso_add_ref(d12_pso_from_iface(iface));
}

static void
d12_pso_release_bos(struct d12_device *device,
                    struct d12_pipeline_state *pso)
{
   if (pso->code_bo)
      agx_bo_unreference(&device->dev, pso->code_bo);
   if (pso->vs.linked.bo)
      agx_bo_unreference(&device->dev, pso->vs.linked.bo);
   if (pso->vs.code_bo)
      agx_bo_unreference(&device->dev, pso->vs.code_bo);
   if (pso->fs.linked.bo)
      agx_bo_unreference(&device->dev, pso->fs.linked.bo);
   if (pso->fs.code_bo)
      agx_bo_unreference(&device->dev, pso->fs.code_bo);
   if (pso->sw_vs_bo)
      agx_bo_unreference(&device->dev, pso->sw_vs_bo);
   if (pso->sw_tes_bo)
      agx_bo_unreference(&device->dev, pso->sw_tes_bo);
   if (pso->gs_count_bo)
      agx_bo_unreference(&device->dev, pso->gs_count_bo);
   if (pso->gs_pre_bo)
      agx_bo_unreference(&device->dev, pso->gs_pre_bo);
   if (pso->gs_main_bo)
      agx_bo_unreference(&device->dev, pso->gs_main_bo);
   if (pso->gs_rast.linked.bo)
      agx_bo_unreference(&device->dev, pso->gs_rast.linked.bo);
   if (pso->gs_rast.code_bo)
      agx_bo_unreference(&device->dev, pso->gs_rast.code_bo);
   if (pso->tcs_bo)
      agx_bo_unreference(&device->dev, pso->tcs_bo);
   if (pso->tes.linked.bo)
      agx_bo_unreference(&device->dev, pso->tes.linked.bo);
   if (pso->tes.code_bo)
      agx_bo_unreference(&device->dev, pso->tes.code_bo);
   if (pso->ms_bo)
      agx_bo_unreference(&device->dev, pso->ms_bo);
   if (pso->ms_setup_bo)
      agx_bo_unreference(&device->dev, pso->ms_setup_bo);
   if (pso->ms_rast.linked.bo)
      agx_bo_unreference(&device->dev, pso->ms_rast.linked.bo);
   if (pso->ms_rast.code_bo)
      agx_bo_unreference(&device->dev, pso->ms_rast.code_bo);
   if (pso->as_bo)
      agx_bo_unreference(&device->dev, pso->as_bo);
   if (pso->as_setup_bo)
      agx_bo_unreference(&device->dev, pso->as_setup_bo);
}

/* The public epoch or the owning library still holds the device here. */
static void
d12_pso_destroy(struct d12_pipeline_state *pso)
{
   if (pso->so_line)
      ID3D12PipelineState_Release(&pso->so_line->iface);
   if (pso->so_triangle)
      ID3D12PipelineState_Release(&pso->so_triangle->iface);
   d12_pso_release_bos(pso->device, pso);
   free(pso->cached_blob);
   d12_private_data_store_destroy(&pso->private_data);
   free(pso);
}

static void
d12_pso_lifetime_unref(struct d12_pipeline_state *pso)
{
   if (!p_atomic_dec_return(&pso->lifetime))
      d12_pso_destroy(pso);
}

static ULONG STDMETHODCALLTYPE
d12_pso_Release(ID3D12PipelineState *iface)
{
   struct d12_pipeline_state *pso = d12_pso_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&pso->refcount);

   /* The public epoch ends: drop its lifetime reference. A library entry
    * may still hold one, in which case nothing else happens here. */
   if (!refs) {
      struct d12_device *device = pso->device;
      d12_pso_lifetime_unref(pso);
      /* A cached PSO remains alive through its library, but its public
       * device reference follows the public epoch. Release after freeing
       * BOs when this was also the last lifetime reference. */
      d12_device_release(device);
   }

   return refs;
}

void
d12_pipeline_state_library_ref(struct d12_pipeline_state *pso)
{
   p_atomic_inc(&pso->lifetime);
}

void
d12_pipeline_state_library_unref(struct d12_pipeline_state *pso)
{
   d12_pso_lifetime_unref(pso);
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_pso, ID3D12PipelineState,
                                d12_pso_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_pso_GetDevice(ID3D12PipelineState *iface, REFIID riid, void **out)
{
   struct d12_pipeline_state *pso = d12_pso_from_iface(iface);
   return ID3D12Device_QueryInterface(&pso->device->iface, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_pso_GetCachedBlob(ID3D12PipelineState *iface, ID3DBlob **blob)
{
   struct d12_pipeline_state *pso = d12_pso_from_iface(iface);
   if (!blob)
      return E_POINTER;

   *blob = NULL;
   if (!pso->cached_blob || !pso->cached_blob_size)
      return E_NOTIMPL;

   *blob = d12_blob_create(pso->cached_blob, pso->cached_blob_size);
   return *blob ? S_OK : E_OUTOFMEMORY;
}

static const ID3D12PipelineStateVtbl d12_pso_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12PipelineState_DEFAULTS,
   .QueryInterface = d12_pso_QueryInterface,
   .AddRef = d12_pso_AddRef,
   .Release = d12_pso_Release,
   .GetPrivateData = d12_pso_GetPrivateData,
   .SetPrivateData = d12_pso_SetPrivateData,
   .SetPrivateDataInterface = d12_pso_SetPrivateDataInterface,
   .SetName = d12_pso_SetName,
   .GetDevice = d12_pso_GetDevice,
   .GetCachedBlob = d12_pso_GetCachedBlob,
};

static bool
d12_container_stage(const void *data, size_t size_B,
                    mesa_shader_stage *stage)
{
   return d12_dxil_container_stage(data, size_B, stage) ||
          d12_dxbc_container_stage(data, size_B, stage);
}

static D3D12_SHADER_VISIBILITY
d12_stage_visibility(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:    return D3D12_SHADER_VISIBILITY_VERTEX;
   case MESA_SHADER_TESS_CTRL: return D3D12_SHADER_VISIBILITY_HULL;
   case MESA_SHADER_TESS_EVAL: return D3D12_SHADER_VISIBILITY_DOMAIN;
   case MESA_SHADER_GEOMETRY:  return D3D12_SHADER_VISIBILITY_GEOMETRY;
   case MESA_SHADER_FRAGMENT:  return D3D12_SHADER_VISIBILITY_PIXEL;
   case MESA_SHADER_TASK:      return D3D12_SHADER_VISIBILITY_AMPLIFICATION;
   case MESA_SHADER_MESH:      return D3D12_SHADER_VISIBILITY_MESH;
   default:                    return D3D12_SHADER_VISIBILITY_ALL;
   }
}

/* Application samplers remain in the USC sampler-state file. Count them
 * before agx_nir_lower_texture reserves hardware state 0 for texture Load
 * and rewrites D3D sampler S to hardware state S + 1. The compiler only
 * reports states it itself promoted into the preamble, which is not enough
 * to size the runtime state file for an ordinary Sample instruction. */
static unsigned
d12_nir_sampler_state_count(nir_shader *nir,
                            const struct d12_binding_map *map)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (!nir_tex_instr_need_sampler(tex))
               continue;

            /* A sampler handle selects the command's global sampler heap;
             * it consumes no entry in the per-stage USC sampler file. */
            if (nir_tex_instr_src_index(
                   tex, nir_tex_src_sampler_handle) >= 0)
               continue;

            if (nir_tex_instr_src_index(tex,
                                        nir_tex_src_sampler_offset) >= 0) {
               if (!map || map->sampler_count >= 17)
                  return UINT_MAX;
               count = MAX2(count, map->sampler_count + 1);
            } else {
               if (tex->sampler_index >= 16)
                  return UINT_MAX;
               count = MAX2(count, tex->sampler_index + 2);
            }
         }
      }
   }

   return count;
}

/* AGX sampler state has only three fixed border colours. Preserve the native
 * path for those and for every non-point filter; for an arbitrary point
 * border, replace an out-of-range sample with the descriptor's exact value.
 * This is deliberately narrower than Honeykrisp's two-sample linear-filter
 * emulation, but it is fully correct for the covered point-filter case and
 * does not claim or perturb the still-unimplemented linear case. */
static nir_def *
d12_custom_border_apply_heap_swizzle(nir_builder *b, nir_tex_instr *tex,
                                     nir_def *border, nir_alu_type type,
                                     unsigned texture_stride_B)
{
   int handle_src =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle_src < 0)
      return border;

   nir_instr *producer = nir_def_instr(tex->src[handle_src].src.ssa);
   if (producer->type != nir_instr_type_intrinsic)
      return border;

   nir_intrinsic_instr *handle = nir_instr_as_intrinsic(producer);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_intrinsic_desc_set(handle) != D12_UNIFORM_HEAP_TEXTURES)
      return border;

   /* The native heap uses the descriptor generation's stride, while the
    * parallel D3D heap retains each SRV's 40-byte API descriptor. Recover
    * the common heap index from the bindless texture byte offset. */
   nir_def *slot =
      nir_udiv_imm(b, nir_u2u64(b, handle->src[0].ssa),
                   texture_stride_B);
   nir_def *descriptor = nir_iadd(
      b, nir_load_descriptor_heap_agx(b),
      nir_imul_imm(b, slot, D12_HEAP_DESC_SIZE));
   nir_def *mapping = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, descriptor, D12_DESC_TEXTURE_METADATA0_OFFSET),
      .align_mul = 4);

   nir_def *zero = nir_alu_type_get_base_type(type) == nir_type_float
                      ? nir_imm_floatN_t(b, 0.0, border->bit_size)
                      : nir_imm_intN_t(b, 0, border->bit_size);
   nir_def *one = nir_alu_type_get_base_type(type) == nir_type_float
                     ? nir_imm_floatN_t(b, 1.0, border->bit_size)
                     : nir_imm_intN_t(b, 1, border->bit_size);
   nir_def *result[4];

   for (unsigned c = 0; c < 4; c++) {
      nir_def *select =
         nir_iand_imm(b, nir_ushr_imm(b, mapping, c * 3), 0x7);
      nir_def *value = nir_channel(b, border, 0);
      for (unsigned source = 1; source < 4; source++) {
         value = nir_bcsel(b, nir_ieq_imm(b, select, source),
                           nir_channel(b, border, source), value);
      }
      value = nir_bcsel(b, nir_ieq_imm(b, select, 4), zero, value);
      result[c] = nir_bcsel(b, nir_ieq_imm(b, select, 5), one, value);
   }

   return nir_vec(b, result, 4);
}

static bool
d12_sampler_sideband_entry(nir_builder *b, nir_tex_instr *tex, bool *bindless,
                           nir_def **entry)
{
   int handle_src = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   nir_def *logical;

   *bindless = handle_src >= 0;
   if (*bindless) {
      nir_intrinsic_instr *handle =
         nir_src_as_intrinsic(tex->src[handle_src].src);
      if (!handle || handle->intrinsic != nir_intrinsic_bindless_sampler_agx)
         return false;

      /* The translator's pre-compaction handle carries logical+28 in its
       * hardware-index source.  A later pass replaces both handle sources
       * with the deduplicated native index. */
      logical = nir_iadd_imm(b, nir_u2u32(b, handle->src[1].ssa), -28);
   } else {
      logical = nir_imm_int(b, tex->sampler_index);
      int sampler_offset =
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset);
      if (sampler_offset >= 0) {
         logical = nir_iadd(b, logical,
                            nir_u2u32(b, tex->src[sampler_offset].src.ssa));
      }
   }

   nir_def *table = nir_load_custom_border_table_agx(b);
   unsigned stride = *bindless ? sizeof(struct d12_descriptor)
                               : sizeof(struct d12_custom_border_entry);
   *entry = nir_iadd(b, table, nir_u2u64(b, nir_imul_imm(b, logical, stride)));
   return true;
}

/* Shared context of the sampler side-table lowerings below. */
struct d12_sampler_lowering {
   const struct d12_native_descriptor_layout *native_layout;
   /* Logical API sampler slots whose native word is complete (see
    * d12_binding_map::plain_static_samplers); a tex/txb/txl/txd through one
    * of them keeps its hardware sample and reads no side table. HAL200 packs
    * the LOD bias into the word for every LOD form, so the gradient sample
    * the DXBC front end builds for a per-view minimum LOD is just as plain. */
   uint16_t plain_static_samplers;
};

static bool
d12_tex_uses_plain_static_sampler(const nir_tex_instr *tex,
                                  const struct d12_sampler_lowering *ctx)
{
   if (!ctx || !ctx->plain_static_samplers ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd) ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
       tex->sampler_index >= 16)
      return false;
   return ctx->plain_static_samplers & BITFIELD_BIT(tex->sampler_index);
}

static bool
d12_lower_custom_point_border(nir_builder *b, nir_tex_instr *tex,
                              void *data)
{
   const struct d12_sampler_lowering *ctx = data;
   const struct d12_native_descriptor_layout *native_layout =
      ctx->native_layout;

   if (!nir_tex_instr_need_sampler(tex) || tex->is_shadow ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0 ||
       tex->op == nir_texop_tg4 ||
       d12_tex_uses_plain_static_sampler(tex, ctx))
      return false;

   unsigned spatial;
   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_1D: spatial = 1; break;
   case GLSL_SAMPLER_DIM_2D: spatial = 2; break;
   case GLSL_SAMPLER_DIM_3D: spatial = 3; break;
   default: return false;
   }

   int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_src < 0 || tex->src[coord_src].src.ssa->num_components < spatial)
      return false;

   b->cursor = nir_after_instr(&tex->instr);

   /* Keep an independent copy of the hardware sample. Replacing tex->def
    * below then dead-code-eliminates the original without making the bcsel
    * self-referential. */
   nir_instr *clone = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, clone);
   nir_def *sample = &nir_instr_as_tex(clone)->def;

   bool bindless;
   nir_def *entry;
   if (!d12_sampler_sideband_entry(b, tex, &bindless, &entry))
      return false;

   nir_def *flags = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, entry,
                   bindless ? offsetof(struct d12_descriptor, sampler_reserved)
                            : offsetof(struct d12_custom_border_entry, flags)),
      .align_mul = 4);
   nir_def *border = nir_load_global_constant(
      b, 4, 32,
      nir_iadd_imm(b, entry,
                   bindless ? offsetof(struct d12_descriptor, sampler_border)
                            : offsetof(struct d12_custom_border_entry, color)),
      .align_mul = 8, .align_offset = 0);

   nir_def *oob = nir_imm_false(b);
   nir_def *coord = tex->src[coord_src].src.ssa;
   const uint32_t axis_flags[] = {
      D12_CUSTOM_BORDER_U, D12_CUSTOM_BORDER_V, D12_CUSTOM_BORDER_W,
   };
   for (unsigned i = 0; i < spatial; i++) {
      nir_def *c = nir_channel(b, coord, i);
      nir_def *axis_oob = nir_ior(b, nir_flt_imm(b, c, 0.0f),
                                 nir_fge_imm(b, c, 1.0f));
      nir_def *axis_border =
         nir_ine_imm(b, nir_iand_imm(b, flags, axis_flags[i]), 0);
      oob = nir_ior(b, oob, nir_iand(b, axis_border, axis_oob));
   }

   const uint32_t enabled =
      D12_CUSTOM_BORDER_ENABLE | D12_CUSTOM_BORDER_POINT;
   nir_def *custom_point =
      nir_ieq_imm(b, nir_iand_imm(b, flags, enabled), enabled);
   nir_def *replace = nir_iand(b, custom_point, oob);

   nir_alu_type type = nir_alu_type_get_base_type(tex->dest_type);
   border = nir_convert_to_bit_size(b, border, type, tex->def.bit_size);
   border = d12_custom_border_apply_heap_swizzle(
      b, tex, border, type, native_layout->texture_stride_B);
   if (tex->def.num_components < 4)
      border = nir_trim_vector(b, border, tex->def.num_components);

   nir_def *result = nir_bcsel(b, replace, border, sample);
   nir_def_replace(&tex->def, result);
   return true;
}

static bool
d12_nir_lower_custom_point_borders(nir_shader *nir,
                                   const struct d12_sampler_lowering *ctx)
{
   return nir_shader_tex_pass(nir, d12_lower_custom_point_border,
                              nir_metadata_none, (void *)ctx);
}

/* AGX2 and G15's Gen4 sampler keep LOD bias outside the packed word. D3D12
 * applies it to implicit, biased, explicit-LOD, and gradient sampling alike,
 * so recover it from the same per-stage side table used for arbitrary border
 * colours before lowering texture ops. HAL200/Gen5 pack the bias natively and
 * publish zero in this side-table field, making the shared lowering a no-op. */
static bool
d12_lower_sampler_lod_bias(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (!nir_tex_instr_need_sampler(tex) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd) ||
       d12_tex_uses_plain_static_sampler(tex, data))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   bool bindless;
   nir_def *entry;
   if (!d12_sampler_sideband_entry(b, tex, &bindless, &entry))
      return false;

   nir_def *bias = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, entry,
                   bindless
                      ? offsetof(struct d12_descriptor, sampler_lod_bias)
                      : offsetof(struct d12_custom_border_entry, lod_bias)),
      .align_mul = 4);
   nir_def *flags = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, entry,
                   bindless ? offsetof(struct d12_descriptor, sampler_reserved)
                            : offsetof(struct d12_custom_border_entry, flags)),
      .align_mul = 4);

   if (tex->op == nir_texop_txd) {
      nir_def *ddx = nir_steal_tex_src(tex, nir_tex_src_ddx);
      nir_def *ddy = nir_steal_tex_src(tex, nir_tex_src_ddy);
      if (!ddx || !ddy)
         return false;

      nir_def *scale = nir_fexp2(b, bias);
      nir_tex_instr_add_src(tex, nir_tex_src_ddx,
                            nir_fmul(b, ddx, scale));
      nir_tex_instr_add_src(tex, nir_tex_src_ddy,
                            nir_fmul(b, ddy, scale));
   } else if (tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, bias);
   } else if (tex->op == nir_texop_txl || tex->op == nir_texop_txb) {
      nir_tex_src_type type = tex->op == nir_texop_txl
                                 ? nir_tex_src_lod
                                 : nir_tex_src_bias;
      nir_def *original = nir_steal_tex_src(tex, type);
      nir_def *combined = original ? nir_fadd(b, original, bias) : bias;

      if (tex->op == nir_texop_txl) {
         /* AGX and D3D both select the nearest point-filtered mip, but AGX
          * breaks exact half-level ties downward while D3D breaks them up.
          * Its LOD input quantizes away an FP16 ULP, so add a visible 1/16
          * only when the unquantized value is exactly at the tie. */
         nir_def *point = nir_ine_imm(
            b, nir_iand_imm(b, flags, D12_SAMPLER_MIP_POINT), 0);
         nir_def *tie = nir_feq_imm(b, nir_ffract(b, combined), 0.5f);
         combined = nir_fadd(
            b, combined,
            nir_bcsel(b, nir_iand(b, point, tie),
                      nir_imm_float(b, 0.0625f),
                      nir_imm_float(b, 0.0f)));

         /* ResourceMinLODClamp applies after the shader and sampler biases.
          * Consume the temporary NIR source here because AGX txl has no
          * separate minimum-LOD operand. */
         nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
         if (min_lod)
            combined = nir_fmax(b, combined, min_lod);
      }

      nir_tex_instr_add_src(tex, type, combined);
   } else {
      return false;
   }

   return true;
}

static bool
d12_nir_lower_sampler_lod_bias(nir_shader *nir,
                               const struct d12_sampler_lowering *ctx)
{
   return nir_shader_tex_pass(nir, d12_lower_sampler_lod_bias,
                              nir_metadata_none, (void *)ctx);
}

/*
 * D3D12 MINIMUM/MAXIMUM filters replace the filter's weighted average with a
 * reduction over the same texel set. The runnable AGX2 packed sampler has no
 * reduction mode, so the reduction is performed in the shader, exactly as
 * Honeykrisp already does for custom border colours. SamplerGen5 executes the
 * all-linear/anisotropic subset natively; mixed nearest/linear filters still
 * require this fallback.
 *
 * The texel set of a linear 2D filter is the gather footprint, so one tg4 per
 * requested component yields it directly and a componentwise fmin/fmax over
 * the four lanes is the reduction.  d12_pack_sampler_descriptor forces such a
 * sampler to nearest mip filtering, so the level the gather lands on is the
 * level the hardware selected.
 *
 * The reduction bits are per descriptor, so the branch below is dynamically
 * uniform for every invocation that shares a sampler, and it is absent
 * entirely from shaders whose samplers never reduce.
 */
static bool
d12_reduction_gatherable(const nir_tex_instr *tex)
{
   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_CUBE:
      return true;
   default:
      /* 1D and 3D reduce over 2 and 8 texels respectively, neither of which
       * tg4 expresses.  Reproducing them from texel fetches would have to
       * reimplement wrap and border selection, where a mistake is worse than
       * the standard filter this leaves in place. */
      return false;
   }
}

static nir_def *
d12_reduction_gather(nir_builder *b, const nir_tex_instr *tex,
                     unsigned component)
{
   unsigned num_srcs = 0;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
      case nir_tex_src_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      case nir_tex_src_texture_deref:
      case nir_tex_src_sampler_deref:
         num_srcs++;
         break;
      default:
         break;
      }
   }

   nir_tex_instr *gather = nir_tex_instr_create(b->shader, num_srcs);
   gather->op = nir_texop_tg4;
   gather->component = component;
   gather->sampler_dim = tex->sampler_dim;
   gather->coord_components = tex->coord_components;
   gather->is_array = tex->is_array;
   gather->dest_type = tex->dest_type;
   gather->texture_index = tex->texture_index;
   gather->sampler_index = tex->sampler_index;
   gather->can_speculate = tex->can_speculate;
   gather->backend_flags = tex->backend_flags;

   unsigned slot = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
      case nir_tex_src_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      case nir_tex_src_texture_deref:
      case nir_tex_src_sampler_deref:
         gather->src[slot].src_type = tex->src[i].src_type;
         gather->src[slot].src = nir_src_for_ssa(tex->src[i].src.ssa);
         slot++;
         break;
      default:
         break;
      }
   }

   nir_def_init(&gather->instr, &gather->def, 4,
                nir_alu_type_get_type_size(tex->dest_type));
   nir_builder_instr_insert(b, &gather->instr);
   return &gather->def;
}

static bool
d12_lower_sampler_reduction(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex) ||
       tex->is_shadow || tex->op == nir_texop_tg4 ||
       tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
       !d12_reduction_gatherable(tex) ||
       d12_tex_uses_plain_static_sampler(tex, data))
      return false;

   /* Reduction is defined over filterable, i.e. floating-point, texels. */
   if (nir_alu_type_get_base_type(tex->dest_type) != nir_type_float)
      return false;

   /* A sparse tex publishes its residency code in the last channel. Reduce
    * only the colour channels and keep the hardware's own status word. */
   const unsigned components =
      tex->def.num_components - (tex->is_sparse ? 1 : 0);
   if (!components || components > 4)
      return false;

   b->cursor = nir_after_instr(&tex->instr);

   /* Resolve the per-sampler metadata first: a sampler this pass cannot
    * address leaves the instruction untouched rather than a dead clone. */
   bool bindless;
   nir_def *entry;
   if (!d12_sampler_sideband_entry(b, tex, &bindless, &entry))
      return false;

   /* Keep an independent copy of the hardware sample so replacing tex->def
    * does not make the phi self-referential. */
   nir_instr *clone = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, clone);
   nir_def *sampled = &nir_instr_as_tex(clone)->def;

   nir_def *flags = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, entry,
                   bindless ? offsetof(struct d12_descriptor, sampler_reserved)
                            : offsetof(struct d12_custom_border_entry, flags)),
      .align_mul = 4);

   nir_def *reduces =
      nir_ine_imm(b, nir_iand_imm(b, flags, D12_SAMPLER_REDUCTION_MASK), 0);
   nir_def *is_min =
      nir_ine_imm(b, nir_iand_imm(b, flags, D12_SAMPLER_REDUCTION_MIN), 0);

   nir_push_if(b, reduces);
   nir_def *reduced;
   {
      nir_def *channels[5];

      for (unsigned c = 0; c < components; c++) {
         nir_def *texels = d12_reduction_gather(b, tex, c);
         nir_def *lo = nir_fmin(b, nir_channel(b, texels, 0),
                                nir_channel(b, texels, 1));
         nir_def *hi = nir_fmin(b, nir_channel(b, texels, 2),
                                nir_channel(b, texels, 3));
         nir_def *minimum = nir_fmin(b, lo, hi);

         nir_def *lo_max = nir_fmax(b, nir_channel(b, texels, 0),
                                    nir_channel(b, texels, 1));
         nir_def *hi_max = nir_fmax(b, nir_channel(b, texels, 2),
                                    nir_channel(b, texels, 3));
         nir_def *maximum = nir_fmax(b, lo_max, hi_max);

         channels[c] = nir_bcsel(b, is_min, minimum, maximum);
      }

      /* The gathers cover the same footprint as the sample they replace, so
       * the sample's own residency code still describes this access. */
      if (tex->is_sparse)
         channels[components] = nir_channel(b, sampled, components);

      reduced = nir_vec(b, channels, tex->def.num_components);
   }
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, reduced, sampled);
   nir_def_replace(&tex->def, result);
   return true;
}

static bool
d12_nir_lower_sampler_reduction(nir_shader *nir,
                                const struct d12_sampler_lowering *ctx)
{
   return nir_shader_tex_pass(nir, d12_lower_sampler_reduction,
                              nir_metadata_none, (void *)ctx);
}

static bool
d12_lower_bindless_sampler_map(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_bindless_sampler_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *logical = nir_iadd_imm(b, nir_u2u32(b, intr->src[1].ssa), -28);
   nir_def *table = nir_load_custom_border_table_agx(b);
   nir_def *entry = nir_iadd(
      b, table,
      nir_u2u64(b, nir_imul_imm(b, logical, sizeof(struct d12_descriptor))));
   nir_def *native = nir_load_global_constant(
      b, 1, 16,
      nir_iadd_imm(b, entry,
                   offsetof(struct d12_descriptor, sampler_native_index)),
      .align_mul = 2);

   nir_src_rewrite(&intr->src[0], nir_imul_imm(b, nir_u2u32(b, native),
                                               D12_SAMPLER_DESC_SIZE));
   nir_src_rewrite(&intr->src[1], nir_u2u16(b, nir_iadd_imm(b, native, 28)));
   return true;
}

static bool
d12_nir_lower_bindless_sampler_map(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, d12_lower_bindless_sampler_map,
                                     nir_metadata_none, NULL);
}

static bool
d12_binding_map_uses_sampler_heap(const struct d12_binding_map *map)
{
   if (!map)
      return false;

   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].klass == D12_RES_SAMPLER && map->entries[i].unbounded)
         return true;
   }
   return false;
}


static void
d12_disassemble_shader(bool native, uint8_t *code, size_t size, FILE *fp)
{
   if (native)
      agx3b_disassemble(code, size, fp);
   else
      agx2_disassemble(code, size, fp);
}

/*
 * DXIL/DXBC -> NIR -> AGX for one compute shader. `o` is zeroed by the
 * caller; everything in it and in `part` is a function of the inputs, so
 * the device stage cache replays it for an identical compile.
 */
static HRESULT
compile_compute_stage(struct d12_device *device, const void *dxil,
                      size_t dxil_size, const struct d12_binding_map *map,
                      uint32_t descriptor_generation,
                      const struct d12_native_descriptor_layout *native_layout,
                      struct d12_stage_outputs *o,
                      struct agx_shader_part *part)
{
   const bool native_compute = device->dev.chip == AGX_CHIP_G17P;

   const uint8_t *bc;
   size_t bc_size;
   const char *why = NULL;
   nir_shader *dxbc_nir = NULL;
   if (!d12_dxil_find_bitcode(dxil, dxil_size, &bc, &bc_size, &why)) {
      /* Shader Model 5.x compute has no DXIL either -- same fallback as
       * the graphics path. */
      if (d12_dxbc_is_dxbc_shader(dxil, dxil_size)) {
         const char *dxbc_error = "unknown";
         dxbc_nir = d12_dxbc_to_nir(
            dxil, dxil_size, MESA_SHADER_COMPUTE, map,
            descriptor_generation, o->texture_shapes, &dxbc_error);
         if (!dxbc_nir) {
            mesa_loge("CreateComputePipelineState: DXBC->NIR failed: %s",
                      dxbc_error);
            return E_INVALIDARG;
         }
      } else {
         mesa_loge("CreateComputePipelineState: %s", why);
         return E_INVALIDARG;
      }
   }

   nir_shader *nir = dxbc_nir;
   const char *error = NULL;

   if (!nir) {
      struct d12_module *mod = d12_dxil_module_alloc();
      if (!mod)
         return E_OUTOFMEMORY;
      if (!d12_dxil_module_parse(mod, bc, bc_size)) {
         mesa_loge("CreateComputePipelineState: DXIL parse failed: %s",
                   mod->error ? mod->error : "unknown");
         d12_dxil_module_free(mod);
         return E_INVALIDARG;
      }

      if (!d12_dxil_collect_texture_shapes(mod, map, o->texture_shapes)) {
         mesa_loge("CreateComputePipelineState: one root slot has "
                   "conflicting texture declarations");
         d12_dxil_module_free(mod);
         return E_INVALIDARG;
      }

      nir = d12_dxil_to_nir(mod, MESA_SHADER_COMPUTE, map,
                            descriptor_generation, &error);
      d12_dxil_module_free(mod);
   }

   if (!nir) {
      mesa_loge("CreateComputePipelineState: DXIL->NIR failed: %s", error);
      return E_INVALIDARG;
   }

   const struct d12_sampler_lowering sampler_lowering = {
      .native_layout = native_layout,
   };
   NIR_PASS(_, nir, d12_nir_lower_sampler_lod_bias, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_custom_point_borders, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_sampler_reduction, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_bindless_sampler_map);

   unsigned runtime_samplers = d12_nir_sampler_state_count(nir, map);
   if (runtime_samplers == UINT_MAX) {
      mesa_loge("CreateComputePipelineState: shader sampler state range "
                "exceeds the 16 API sampler limit");
      ralloc_free(nir);
      return E_INVALIDARG;
   }

   bool native_txf = false;
   if (native_compute &&
       !d12_nir_g17p_txf_supported(nir, &native_txf)) {
      mesa_loge("CreateComputePipelineState: native texture compilation subset refused");
      ralloc_free(nir);
      return E_NOTIMPL;
   }
   if (native_txf)
      NIR_PASS(_, nir, agx_nir_lower_texture_for_generation,
               descriptor_generation);
   else
      NIR_PASS(_, nir, agx_nir_lower_texture);
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);
   if (native_txf &&
       !d12_nir_lower_g17p_txf(nir, descriptor_generation)) {
      mesa_loge("CreateComputePipelineState: native table fetch lowering refused");
      ralloc_free(nir);
      return E_NOTIMPL;
   }
   if (native_txf)
      agx_preprocess_nir_g17p(nir);
   else
      agx_preprocess_nir(nir);

   /* Uniform indices become stable after preprocessing. */
   NIR_PASS(_, nir, d12_dxil_lower_root);
   if (native_txf)
      NIR_PASS(_, nir, d12_nir_shift_g17p_txf_root);

   struct agx_shader_key key = {
      .dev = agx_gather_device_key(&device->dev),
      /* Allocas live in per-thread scratch, and large shaders spill. */
      .has_scratch = true,
      /* Match the existing native API-compute contract: main is monolithic. */
      .no_preamble = native_compute,
      /*
       * Constant promotion parks constants in uniform registers that only
       * the shader's preamble fills. Both USC builders push a PRESHADER
       * word, so this looks safe -- and for compute it is. Turning it on
       * was measured on 2026-08-04 and it fails every graphics test
       * (g4_triangle, g5_depth) while changing nothing about compute.
       *
       * The graphics path takes the preamble's address from
       * st->info.preamble_offset but the code from st->linked.bo, and
       * those are the PRE-link info and the POST-fast-link binary. The
       * offset does not survive agx_fast_link, so the GPU runs whatever
       * sits at that offset in the linked binary. That is the "high half
       * of a 64-bit address came back 0x86 instead of 0x2f" this comment
       * used to describe -- not a missing preamble, a misaddressed one.
       *
       * Graphics stages now retain their original preamble code, but
       * constant promotion remains disabled until its separate register and
       * rodata interactions have a broad regression pass. Texture promotion
       * below is independently required and validated.
       */
      .promote_constants = false,
      /* Uniform registers below this are the driver's: the root pointer. */
      .reserved_preamble = D12_UNIFORM_RESERVED,
   };

      memset(part, 0, sizeof(*part));
   const bool compiled = agx_compile_shader_nir(nir, &key, part);
   ralloc_free(nir);
   /* agx_nir_lower_texture consumes a dynamic sampler handle before the
    * backend can conservatively set this bit.  The root map remains the
    * authoritative declaration that the hardware sampler heap is needed. */
   part->info.uses_sampler_heap |= d12_binding_map_uses_sampler_heap(map);
   part->info.sampler_state_count =
      MAX2(part->info.sampler_state_count, runtime_samplers);
   d12_report_scratch("shader", &part->info);

   if (d12_debug()->dump_tex && part->binary) {
      fprintf(stderr,
              "driver cs: has_preamble=%d preamble_off=0x%x main_off=0x%x "
              "push_count=%u nr_preamble_gprs=%u rodata.size_16=%u\n",
              part->info.has_preamble, part->info.preamble_offset,
              part->info.main_offset, part->info.push_count,
              part->info.nr_preamble_gprs, part->info.rodata.size_16);
      if (part->info.has_preamble) {
         fprintf(stderr, "; driver preamble\n");
         d12_disassemble_shader(
            native_compute,
            (uint8_t *)part->binary + part->info.preamble_offset,
                          part->info.main_offset > part->info.preamble_offset
                             ? part->info.main_offset - part->info.preamble_offset
                             : part->info.binary_size - part->info.preamble_offset,
                          stderr);
      }
      fprintf(stderr, "; driver main\n");
      d12_disassemble_shader(
         native_compute, (uint8_t *)part->binary + part->info.main_offset,
         part->info.binary_size - part->info.main_offset, stderr);
   }

   if (!compiled || !part->binary || !part->info.binary_size) {
      const char *refusal = agx_last_compile_refusal();
      mesa_loge("CreateComputePipelineState: AGX compile produced no code: %s",
                refusal && refusal[0] ? refusal : "no compiler reason recorded");
      free(part->binary);
      part->binary = NULL;
      return E_FAIL;
   }

   o->native_txf = native_txf;
   return S_OK;
}

/*
 * The real work: application DXIL -> our translator -> NIR -> AGX, then
 * upload. This is the campaign's shader path running inside the driver.
 */
HRESULT
d12_pipeline_state_create_compute(struct d12_device *device,
                                  const void *dxil, size_t dxil_size,
                                  const struct d12_binding_map *map,
                                  REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   if (!dxil || !dxil_size)
      return E_INVALIDARG;

   const bool native_compute = device->dev.chip == AGX_CHIP_G17P;
   const uint32_t descriptor_generation =
      (uint32_t)agx_chip_descriptor_generation(device->dev.chip);
   struct d12_native_descriptor_layout native_layout;
   if (!d12_native_descriptor_layout(descriptor_generation, &native_layout)) {
      mesa_loge("CreateComputePipelineState: unsupported native descriptor "
                "generation %u", descriptor_generation);
      return E_NOTIMPL;
   }

   struct d12_binding_map filtered_map;
   if (map) {
      filtered_map = *map;
      filtered_map.visibility_filter = D3D12_SHADER_VISIBILITY_ALL;
      map = &filtered_map;
   }

   /* The container declares its stage; honour it or refuse. Forcing a
    * non-compute shader through this path compiles its store_output
    * against the compute backend, which asserts "Unhandled intrinsic"
    * and takes the process with it -- a crash where an E_INVALIDARG
    * belongs. Same rule the standalone tool already follows
    * (docs/findings.md).
    */
   mesa_shader_stage declared;
   if (d12_container_stage(dxil, dxil_size, &declared) &&
       declared != MESA_SHADER_COMPUTE) {
      mesa_loge("CreateComputePipelineState: bytecode declares a %s shader",
                _mesa_shader_stage_to_string(declared));
      return E_INVALIDARG;
   }

   /* Shader dumps are a per-compile side effect; keep them per PSO. */
   const bool bypass_cache =
      d12_debug()->dump_shaders || d12_debug()->dump_tex;

   struct d12_stage_cache_key key;
   d12_stage_cache_key_init(&key, MESA_SHADER_COMPUTE, dxil, dxil_size, map);
   struct d12_cached_stage *entry = NULL;
   if (!bypass_cache)
      entry = d12_shader_cache_find_stage(device, &key, dxil);
   if (!entry) {
      struct d12_stage_outputs o = {0};
      struct agx_shader_part compiled;
      HRESULT chr = compile_compute_stage(device, dxil, dxil_size, map,
                                          descriptor_generation,
                                          &native_layout, &o, &compiled);
      if (FAILED(chr))
         return chr;
      entry = d12_shader_cache_add_stage(device, &key, dxil, &compiled, &o,
                                         !bypass_cache);
      if (!entry) {
         free(compiled.binary);
         return E_OUTOFMEMORY;
      }
   }
   const struct agx_shader_part *part = &entry->part;
   HRESULT hr = S_OK;

   struct d12_pipeline_state *pso = calloc(1, sizeof(*pso));
   if (!pso) {
      hr = E_OUTOFMEMORY;
      goto fail;
   }

   if (native_compute) {
      hr = d12_native_compute_code_create(device, part, &pso->code_bo,
                                           &pso->code_va, &pso->g17p_rodata_va);
      pso->g17p_compute_container = SUCCEEDED(hr);
      if (FAILED(hr)) {
         mesa_loge("CreateComputePipelineState: native container/setup refused (%#x)",
                   (unsigned)hr);
         goto fail;
      }
   } else {
      /* One executable upload per compiled shader, shared by every PSO
       * created from it. */
      pso->code_bo = d12_cached_stage_code_bo(device, entry, "d3d12 shader");
      if (!pso->code_bo) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
   }

   pso->iface.lpVtbl = (ID3D12PipelineStateVtbl *)d12_api_trace_vtbl(
      D12_API_TRACE_PIPELINE_STATE, &d12_pso_vtbl);
   pso->refcount = 1;
   pso->lifetime = 1;
   pso->device = device;
   pso->is_compute = true;
   pso->g17p_native_txf = entry->out.native_txf;
   memcpy(pso->texture_shapes, entry->out.texture_shapes,
          sizeof(pso->texture_shapes));
   pso->info = part->info;
   if (!native_compute)
      pso->code_va = pso->code_bo->va->addr;
   d12_private_data_store_init(&pso->private_data, (IUnknown *)&pso->iface);
   d12_device_add_ref(device);

   mesa_logi("compiled compute shader: %u bytes AGX, %u GPRs, main +0x%x, "
             "numthreads %ux%ux%u, push %u, preamble %d",
             part->info.binary_size, part->info.nr_gprs,
             part->info.main_offset, part->info.workgroup_size[0],
             part->info.workgroup_size[1], part->info.workgroup_size[2],
             part->info.push_count, part->info.has_preamble);

   if (d12_debug()->dump_shaders) {
      if (part->info.has_preamble) {
         fprintf(stderr, "; preamble +0x%x\n", part->info.preamble_offset);
         d12_disassemble_shader(
            native_compute,
            (uint8_t *)part->binary + part->info.preamble_offset,
                          part->info.main_offset - part->info.preamble_offset,
                          stderr);
      }
      fprintf(stderr, "; main +0x%x\n", part->info.main_offset);
      d12_disassemble_shader(
         native_compute, (uint8_t *)part->binary + part->info.main_offset,
         part->info.binary_size - part->info.main_offset, stderr);
   }

   d12_cached_stage_unref(device, entry);
   hr = d12_pso_QueryInterface(&pso->iface, riid, out);
   d12_pso_Release(&pso->iface);
   return hr;

fail:
   d12_cached_stage_unref(device, entry);
   free(pso);
   return hr;
}

/*
 * Graphics: compile one stage of DXIL to an AGX shader part, running the
 * stage's mandatory lowering (tilebuffer for fragment, UVS for vertex).
 */
/*
 * DXIL container -> NIR, with nothing lowered yet.
 *
 * Split out because the geometry and tessellation lowerings have to run
 * on raw NIR, before agx_preprocess_nir: poly_nir_lower_gs REWRITES
 * info.stage to COMPUTE, so anything that reads the stage back has to
 * come after it.
 */
static nir_shader *dxil_to_nir_stage_vis(const struct d12_binding_map *map,
                                         uint32_t descriptor_generation,
                                         const void *dxil, size_t dxil_size,
                                         mesa_shader_stage stage,
                                         uint8_t *texture_shapes,
                                         D3D12_SHADER_VISIBILITY visibility);

/* Every ordinary stage takes its binding visibility straight from its NIR
 * stage. Only mesh and amplification need the two to differ. */
static nir_shader *
dxil_to_nir_stage(const struct d12_binding_map *map,
                  uint32_t descriptor_generation, const void *dxil,
                  size_t dxil_size, mesa_shader_stage stage,
                  uint8_t *texture_shapes)
{
   return dxil_to_nir_stage_vis(map, descriptor_generation, dxil, dxil_size,
                                stage, texture_shapes,
                                d12_stage_visibility(stage));
}

/*
 * visibility is split from stage because a MESH or AMPLIFICATION shader is
 * translated as COMPUTE -- that is what its container truthfully declares
 * and what it compiles to -- while the root signature still describes its
 * bindings under D3D12_SHADER_VISIBILITY_MESH / _AMPLIFICATION. Deriving
 * the filter from the NIR stage would resolve those to _ALL and hide every
 * mesh-visible range, which reads as "shader uses a resource the root
 * signature does not bind".
 */
static nir_shader *
dxil_to_nir_stage_vis(const struct d12_binding_map *map,
                      uint32_t descriptor_generation, const void *dxil,
                      size_t dxil_size, mesa_shader_stage stage,
                      uint8_t *texture_shapes,
                      D3D12_SHADER_VISIBILITY visibility)
{
   struct d12_binding_map filtered_map;
   if (map) {
      filtered_map = *map;
      filtered_map.visibility_filter = visibility;
      map = &filtered_map;
   }

   mesa_shader_stage declared;
   if (d12_container_stage(dxil, dxil_size, &declared) &&
       declared != stage) {
      mesa_loge("graphics PSO: %s bytecode declares a %s shader",
                _mesa_shader_stage_to_string(stage),
                _mesa_shader_stage_to_string(declared));
      return NULL;
   }

   const uint8_t *bc;
   size_t bc_size;
   const char *why = NULL;
   if (!d12_dxil_find_bitcode(dxil, dxil_size, &bc, &bc_size, &why)) {
      /*
       * No DXIL. Shader Model 5.x has none -- fxc emits a SHEX/SHDR
       * token stream instead -- and SM 5.1 is legal D3D12, so fall
       * through to the DXBC front end rather than failing. This is the
       * single largest conformance gap measured against vkd3d-proton:
       * 202 of 557 tests stop right here.
       */
      if (d12_dxbc_is_dxbc_shader(dxil, dxil_size)) {
         const char *dxbc_error = "unknown";
         nir_shader *nir =
            d12_dxbc_to_nir(dxil, dxil_size, stage, map,
                            descriptor_generation, NULL, &dxbc_error);
         if (!nir)
            mesa_loge("graphics PSO: %s DXBC->NIR failed: %s",
                      _mesa_shader_stage_to_string(stage), dxbc_error);
         return nir;
      }

      mesa_loge("graphics PSO: %s bytecode rejected: %s",
                _mesa_shader_stage_to_string(stage), why);
      return NULL;
   }

   struct d12_module *mod = d12_dxil_module_alloc();
   if (!mod)
      return NULL;
   if (!d12_dxil_module_parse(mod, bc, bc_size)) {
      mesa_loge("graphics PSO: DXIL parse failed: %s",
                mod->error ? mod->error : "unknown");
      d12_dxil_module_free(mod);
      return NULL;
   }

   /* The draw path packs texture descriptors with the dimension the shader
    * declared for that slot (2D vs array vs cube). Collect it for every
    * stage into the PSO's table; the stages must agree on a shared slot. */
   if (texture_shapes &&
       !d12_dxil_collect_texture_shapes(mod, map, texture_shapes)) {
      mesa_loge("graphics PSO: %s declares a conflicting texture shape for "
                "a root slot", _mesa_shader_stage_to_string(stage));
      d12_dxil_module_free(mod);
      return NULL;
   }

   const char *error = NULL;
   nir_shader *nir = d12_dxil_to_nir(mod, stage, map,
                                     descriptor_generation, &error);
   d12_dxil_module_free(mod);

   if (!nir)
      mesa_loge("graphics PSO: %s DXIL->NIR failed: %s",
                _mesa_shader_stage_to_string(stage), error);

   return nir;
}

/*
 * The vertex-fetch keys a PSO compiles with, indexed by DXIL input
 * signature element id. Empty means "no input layout", which keeps the
 * original static_vi prolog path byte for byte.
 */
struct d12_vs_fetch {
   struct agx_velem_key keys[D12_MAX_VB_ATTRIBS];
   unsigned nr;
};

enum d12_sig_section {
   D12_SIG_INPUT,
   D12_SIG_OUTPUT,
   D12_SIG_PATCH,
};

static unsigned
d12_read_signature(const void *bytecode, size_t size,
                   enum d12_sig_section section,
                   struct d12_input_sig_elem *out, unsigned max)
{
   bool dxbc = d12_dxbc_is_dxbc_shader(bytecode, size);

   switch (section) {
   case D12_SIG_INPUT:
      return dxbc ? d12_dxbc_input_signature(bytecode, size, out, max)
                  : d12_dxil_input_signature(bytecode, size, out, max);
   case D12_SIG_OUTPUT:
      return dxbc ? d12_dxbc_output_signature(bytecode, size, out, max)
                  : d12_dxil_output_signature(bytecode, size, out, max);
   case D12_SIG_PATCH:
      return dxbc ? d12_dxbc_patch_signature(bytecode, size, out, max)
                  : d12_dxil_patch_signature(bytecode, size, out, max);
   }

   return 0;
}

static bool
d12_sig_is_system(const struct d12_input_sig_elem *e)
{
   return !strncasecmp(e->name, "SV_", 3);
}

/* System values in this list are supplied by fixed function or by the stage
 * invocation itself; they need not occupy an output register in the preceding
 * shader. SV_PrimitiveID is only generated for the pixel shader when no
 * geometry shader intervenes -- a GS must explicitly forward it. */
static bool
d12_sig_is_generated(const struct d12_input_sig_elem *e,
                     mesa_shader_stage consumer_stage,
                     bool generate_primitive_id)
{
   if (!strcasecmp(e->name, "SV_VertexID") ||
       !strcasecmp(e->name, "SV_InstanceID"))
      return consumer_stage == MESA_SHADER_VERTEX;
   if (!strcasecmp(e->name, "SV_OutputControlPointID"))
      return consumer_stage == MESA_SHADER_TESS_CTRL;
   if (!strcasecmp(e->name, "SV_DomainLocation"))
      return consumer_stage == MESA_SHADER_TESS_EVAL;
   if (!strcasecmp(e->name, "SV_GSInstanceID"))
      return consumer_stage == MESA_SHADER_GEOMETRY;
   if (!strcasecmp(e->name, "SV_PrimitiveID"))
      return consumer_stage == MESA_SHADER_GEOMETRY ||
             (consumer_stage == MESA_SHADER_FRAGMENT &&
              generate_primitive_id);
   if (consumer_stage != MESA_SHADER_FRAGMENT)
      return false;

   return !strcasecmp(e->name, "SV_Position") ||
          !strcasecmp(e->name, "SV_IsFrontFace") ||
          !strcasecmp(e->name, "SV_SampleIndex") ||
          !strcasecmp(e->name, "SV_Coverage") ||
          !strcasecmp(e->name, "SV_InnerCoverage") ||
          !strcasecmp(e->name, "SV_Barycentrics");
}

static const struct d12_input_sig_elem *
d12_find_signature_semantic(const struct d12_input_sig_elem *sig, unsigned nr,
                            const struct d12_input_sig_elem *needle)
{
   for (unsigned i = 0; i < nr; i++) {
      if (sig[i].semantic_index == needle->semantic_index &&
          !strcasecmp(sig[i].name, needle->name))
         return &sig[i];
   }

   return NULL;
}

static bool
d12_signature_element_compatible(const struct d12_input_sig_elem *producer,
                                 const struct d12_input_sig_elem *consumer,
                                 bool exact)
{
   if (producer->register_index != consumer->register_index ||
       producer->start_component != consumer->start_component ||
       (producer->component_type && consumer->component_type &&
        producer->component_type != consumer->component_type) ||
       producer->min_precision != consumer->min_precision)
      return false;

   if (exact)
      return producer->rows == consumer->rows &&
             producer->component_count == consumer->component_count;

   return producer->rows >= consumer->rows &&
          producer->component_count >= consumer->component_count;
}

static bool
d12_validate_stage_link(const void *producer_bc, size_t producer_size,
                        enum d12_sig_section producer_section,
                        const void *consumer_bc, size_t consumer_size,
                        enum d12_sig_section consumer_section,
                        mesa_shader_stage consumer_stage,
                        bool generate_primitive_id, bool exact)
{
   struct d12_input_sig_elem producer[D12_MAX_SIG_ELEMENTS] = {0};
   struct d12_input_sig_elem consumer[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr_producer = d12_read_signature(
      producer_bc, producer_size, producer_section, producer,
      ARRAY_SIZE(producer));
   unsigned nr_consumer = d12_read_signature(
      consumer_bc, consumer_size, consumer_section, consumer,
      ARRAY_SIZE(consumer));

   for (unsigned i = 0; i < nr_consumer; i++) {
      const struct d12_input_sig_elem *p =
         d12_find_signature_semantic(producer, nr_producer, &consumer[i]);

      if (!p && d12_sig_is_generated(&consumer[i], consumer_stage,
                                     generate_primitive_id))
         continue;

      if (!p || !d12_signature_element_compatible(p, &consumer[i], exact))
         return false;
   }

   if (exact) {
      for (unsigned i = 0; i < nr_producer; i++) {
         if (!d12_find_signature_semantic(consumer, nr_consumer,
                                          &producer[i]))
            return false;
      }
   }

   return true;
}

/*
 * DXIL assigns mesh per-vertex outputs, mesh per-primitive outputs and pixel
 * inputs independent physical registers.  The software raster linker already
 * joins those streams by semantic, so applying the ordinary stage comparator's
 * register/start-component requirement rejects valid repacking.  Mesh links do
 * require an exact semantic width: unlike the ordinary vertex path, consuming
 * only part of a mesh output is rejected by native D3D12.
 */
static bool
d12_mesh_signature_element_compatible(const struct d12_input_sig_elem *producer,
                                      const struct d12_input_sig_elem *consumer)
{
   return producer->rows == consumer->rows &&
          producer->component_count == consumer->component_count &&
          (!producer->component_type || !consumer->component_type ||
           producer->component_type == consumer->component_type) &&
          producer->min_precision == consumer->min_precision;
}

static bool
d12_validate_mesh_ps_link(const void *ms, size_t ms_size, const void *ps,
                          size_t ps_size)
{
   if (!ps || !ps_size)
      return true;

   struct d12_input_sig_elem vertex[D12_MAX_SIG_ELEMENTS] = {0};
   struct d12_input_sig_elem primitive[D12_MAX_SIG_ELEMENTS] = {0};
   struct d12_input_sig_elem input[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr_vertex = d12_read_signature(ms, ms_size, D12_SIG_OUTPUT, vertex,
                                           ARRAY_SIZE(vertex));
   unsigned nr_primitive = d12_read_signature(ms, ms_size, D12_SIG_PATCH,
                                              primitive, ARRAY_SIZE(primitive));
   unsigned nr_input =
      d12_read_signature(ps, ps_size, D12_SIG_INPUT, input, ARRAY_SIZE(input));

   bool has_position = false;
   for (unsigned i = 0; i < nr_vertex; i++)
      has_position |= !strcasecmp(vertex[i].name, "SV_Position");
   if (!has_position)
      return false;

   for (unsigned i = 0; i < nr_input; i++) {
      const struct d12_input_sig_elem *v =
         d12_find_signature_semantic(vertex, nr_vertex, &input[i]);
      const struct d12_input_sig_elem *p =
         d12_find_signature_semantic(primitive, nr_primitive, &input[i]);

      if ((v && d12_mesh_signature_element_compatible(v, &input[i])) ||
          (p && d12_mesh_signature_element_compatible(p, &input[i])))
         continue;

      /* PrimitiveID is not generated across a mesh link: the mesh shader must
       * publish it as a primitive output, hence generate_primitive_id=false. */
      if (!v && !p &&
          d12_sig_is_generated(&input[i], MESA_SHADER_FRAGMENT, false))
         continue;

      return false;
   }

   return true;
}

static bool
d12_validate_vertex_input_layout(const void *vs, size_t vs_size,
                                 const D3D12_INPUT_LAYOUT_DESC *layout)
{
   struct d12_input_sig_elem sig[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr = d12_read_signature(vs, vs_size, D12_SIG_INPUT, sig,
                                    ARRAY_SIZE(sig));

   for (unsigned i = 0; i < nr; i++) {
      if (d12_sig_is_system(&sig[i]))
         continue;

      bool found = false;
      if (layout && layout->pInputElementDescs) {
         for (unsigned e = 0; e < layout->NumElements; e++) {
            const D3D12_INPUT_ELEMENT_DESC *el =
               &layout->pInputElementDescs[e];
            if (el->SemanticName &&
                el->SemanticIndex == sig[i].semantic_index &&
                !strcasecmp(el->SemanticName, sig[i].name)) {
               found = true;
               break;
            }
         }
      }

      if (!found)
         return false;
   }

   return true;
}

static bool
d12_validate_graphics_signatures(
   const void *vs, size_t vs_size, const void *ps, size_t ps_size,
   const void *gs, size_t gs_size, const void *hs, size_t hs_size,
   const void *ds, size_t ds_size, const D3D12_INPUT_LAYOUT_DESC *layout)
{
   if (!d12_validate_vertex_input_layout(vs, vs_size, layout))
      return false;

   const void *last = vs;
   size_t last_size = vs_size;

   if (hs && hs_size && ds && ds_size) {
      if (!d12_validate_stage_link(vs, vs_size, D12_SIG_OUTPUT,
                                   hs, hs_size, D12_SIG_INPUT,
                                   MESA_SHADER_TESS_CTRL, false, false) ||
          !d12_validate_stage_link(hs, hs_size, D12_SIG_OUTPUT,
                                   ds, ds_size, D12_SIG_INPUT,
                                   MESA_SHADER_TESS_EVAL, false, true) ||
          !d12_validate_stage_link(hs, hs_size, D12_SIG_PATCH,
                                   ds, ds_size, D12_SIG_PATCH,
                                   MESA_SHADER_TESS_EVAL, false, true))
         return false;
      last = ds;
      last_size = ds_size;
   }

   if (gs && gs_size) {
      if (!d12_validate_stage_link(last, last_size, D12_SIG_OUTPUT,
                                   gs, gs_size, D12_SIG_INPUT,
                                   MESA_SHADER_GEOMETRY, false, false))
         return false;
      last = gs;
      last_size = gs_size;
   }

   if (ps && ps_size &&
       !d12_validate_stage_link(last, last_size, D12_SIG_OUTPUT,
                                ps, ps_size, D12_SIG_INPUT,
                                MESA_SHADER_FRAGMENT, !gs_size, false))
      return false;

   return true;
}

/*
 * Resolve a D3D12_INPUT_LAYOUT_DESC against the vertex shader's own DXIL
 * input signature.
 *
 * D3D12 binds attributes by (SemanticName, SemanticIndex) -- NOT by
 * position in either list. An input layout may describe attributes the
 * shader never reads, list them in any order, and the shader may read
 * inputs the layout does not supply. Matching positionally happens to
 * work for a layout written to match its shader and silently corrupts
 * every other one, so the match is done by name.
 *
 * Semantic names are ASCII and compared case-insensitively, as D3D does.
 */
static void
resolve_input_layout(struct d12_pipeline_state *pso,
                     const D3D12_INPUT_LAYOUT_DESC *layout, const void *vs_dxil,
                     size_t vs_size, struct d12_vs_fetch *fetch)
{
   memset(fetch, 0, sizeof(*fetch));
   pso->nr_vb_attribs = 0;

   if (!layout || !layout->NumElements || !layout->pInputElementDescs)
      return;

   /*
    * DXBC and DXIL keep their input signatures in different places: DXIL
    * in the LLVM bitcode module, DXBC in the container's ISGN chunk. Using
    * the DXIL reader on a DXBC shader silently returns zero elements, so
    * nothing matches the input layout, no attributes are fetched, and
    * load_input reaches the AGX backend as "Unhandled intrinsic" -- which
    * was the sole blocker on all eight tests in that cluster, every one of
    * them a _dxbc variant.
    */
   struct d12_input_sig_elem sig[D12_MAX_SIG_ELEMENTS];
   unsigned nr_sig =
      d12_dxbc_is_dxbc_shader(vs_dxil, vs_size)
         ? d12_dxbc_input_signature(vs_dxil, vs_size, sig, ARRAY_SIZE(sig))
         : d12_dxil_input_signature(vs_dxil, vs_size, sig, ARRAY_SIZE(sig));

   for (unsigned e = 0; e < layout->NumElements; e++) {
      const D3D12_INPUT_ELEMENT_DESC *el = &layout->pInputElementDescs[e];
      if (!el->SemanticName)
         continue;

      /* Find the shader input this element feeds. An element with no
       * match is legal and simply unused. */
      unsigned id = UINT32_MAX;
      for (unsigned i = 0; i < nr_sig; i++) {
         if (sig[i].semantic_index == el->SemanticIndex &&
             !strcasecmp(sig[i].name, el->SemanticName)) {
            id = sig[i].id;
            break;
         }
      }

      if (id == UINT32_MAX)
         continue;

      if (id >= D12_MAX_VB_ATTRIBS || el->InputSlot >= D12_MAX_VB_SLOTS) {
         mesa_logw("input layout: %s%u out of range (id %u, slot %u)",
                   el->SemanticName, el->SemanticIndex, id, el->InputSlot);
         continue;
      }

      enum pipe_format fmt = d12_format_to_pipe(el->Format);
      if (fmt == PIPE_FORMAT_NONE) {
         mesa_logw("input layout: %s%u has unsupported DXGI format %u",
                   el->SemanticName, el->SemanticIndex,
                   (unsigned)el->Format);
         continue;
      }

      bool instanced =
         el->InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;

      /* D3D12_APPEND_ALIGNED_ELEMENT means "pack after the previous
       * element in the same slot". */
      uint32_t offset = el->AlignedByteOffset;
      if (offset == D3D12_APPEND_ALIGNED_ELEMENT) {
         offset = 0;
         for (unsigned k = 0; k < e; k++) {
            const D3D12_INPUT_ELEMENT_DESC *pe =
               &layout->pInputElementDescs[k];
            if (pe->InputSlot != el->InputSlot)
               continue;
            enum pipe_format pf = d12_format_to_pipe(pe->Format);
            if (pf == PIPE_FORMAT_NONE)
               continue;
            uint32_t end = (pe->AlignedByteOffset ==
                            D3D12_APPEND_ALIGNED_ELEMENT)
                              ? offset
                              : pe->AlignedByteOffset;
            offset = MAX2(offset, end + util_format_get_blocksize(pf));
         }
      }

      enum pipe_format interchange = agx_vbo_internal_format(fmt);

      pso->vb_attribs[id] = (struct d12_vb_attrib){
         .live = true,
         .slot = (uint8_t)el->InputSlot,
         .offset_B = offset,
         .format = fmt,
         .interchange_align = util_format_get_blocksize(interchange),
      };

      /* stride stays 0: it is published per draw, see the dynamic-stride
       * note in compile_gfx_stage. */
      fetch->keys[id] = (struct agx_velem_key){
         .format = fmt,
         .stride = 0,
         /* StepRate 0 means element zero for every instance in the draw.
          * agx_nir_lower_vbo already implements that distinct mode when
          * divisor is zero, so it must not be collapsed to divisor 1.
          */
         .divisor = instanced ? el->InstanceDataStepRate : 0,
         .instanced = instanced,
      };

      fetch->nr = MAX2(fetch->nr, id + 1);
      pso->nr_vb_attribs = fetch->nr;
   }

   /*
    * An input the SHADER reads but the layout does not supply is legal:
    * D3D12 defines the fetch as reading zeroes. Give it a well-formed key
    * anyway, because agx_nir_lower_vbo has to fetch SOMETHING for every
    * load_input it sees -- a zeroed key means PIPE_FORMAT_NONE, which has
    * no channels to describe. Left un-live, so the draw path points its
    * base at the zero page.
    */
   for (unsigned i = 0; i < nr_sig; i++) {
      uint32_t id = sig[i].id;
      if (id >= D12_MAX_VB_ATTRIBS || pso->vb_attribs[id].live)
         continue;

      fetch->keys[id] = (struct agx_velem_key){
         .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      };
      fetch->nr = MAX2(fetch->nr, id + 1);
      pso->vb_attribs[id].interchange_align = 4;
      pso->nr_vb_attribs = fetch->nr;
   }
}

static bool
d12_lower_coverage_to_epilog_intr(nir_builder *b,
                                  nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != FRAG_RESULT_SAMPLE_MASK)
      return false;

   /* The split fragment path writes colour in a linked epilog.  A hardware
    * sample-mask instruction in the API part does not constrain that later
    * tilebuffer store, so pass SV_Coverage through the established epilog ABI
    * as its explicit write-sample mask instead. */
   b->cursor = nir_instr_remove(&intr->instr);
   nir_export_agx(b, nir_u2u16(b, intr->src[0].ssa),
                  .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   *(bool *)data = true;
   return true;
}

static bool
d12_nir_lower_coverage_to_epilog(nir_shader *nir, bool *lowered)
{
   return nir_shader_intrinsics_pass(
      nir, d12_lower_coverage_to_epilog_intr, nir_metadata_control_flow,
      lowered);
}

static bool
d12_lower_a8_output_instr(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned rt = sem.location - FRAG_RESULT_DATA0;
   uint8_t mask = *(const uint8_t *)data;
   if (sem.location < FRAG_RESULT_DATA0 || rt >= 8 ||
       !(mask & BITFIELD_BIT(rt)))
      return false;

   nir_def *value = intr->src[0].ssa;
   if (value->num_components < 4)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *alpha = nir_channel(b, value, 3);
   nir_src_rewrite(&intr->src[0], nir_replicate(b, alpha,
                                                value->num_components));
   return true;
}

static bool
d12_nir_lower_a8_outputs(nir_shader *nir, uint8_t mask)
{
   if (!mask)
      return false;

   return nir_shader_intrinsics_pass(
      nir, d12_lower_a8_output_instr, nir_metadata_control_flow, &mask);
}

static unsigned
d12_flat_transport_type(const struct d12_flat_transport *transport,
                        unsigned location, unsigned component)
{
   if (!transport || !transport->enabled || location >= 64 || component >= 4)
      return 0;

   uint64_t bit = BITFIELD64_BIT(location);
   if (transport->uint_components[component] & bit)
      return 1;
   if (transport->sint_components[component] & bit)
      return 2;
   return 0;
}

static bool
d12_lower_flat_input_transport_instr(nir_builder *b,
                                     nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned first = nir_intrinsic_component(intr);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   bool progress = false;

   b->cursor = nir_after_instr(&intr->instr);
   for (unsigned i = 0; i < intr->def.num_components; i++) {
      nir_def *value = nir_channel(b, &intr->def, i);
      unsigned type = d12_flat_transport_type(data, sem.location, first + i);
      components[i] = type && ((struct d12_flat_transport *)data)->bitcast
                         ? value
                      : type == 1 ? nir_f2u32(b, value)
                      : type == 2 ? nir_f2i32(b, value)
                                  : value;
      progress |= type != 0;
   }

   if (!progress)
      return false;

   if (((struct d12_flat_transport *)data)->bitcast)
      nir_intrinsic_set_dest_type(intr, nir_type_float32);

   nir_def *replacement =
      nir_vec(b, components, intr->def.num_components);
   nir_def_rewrite_uses_after(&intr->def, replacement);
   return true;
}

static bool
d12_nir_lower_flat_input_transport(
   nir_shader *nir, const struct d12_flat_transport *transport)
{
   if (!transport || !transport->enabled)
      return false;

   return nir_shader_intrinsics_pass(
      nir, d12_lower_flat_input_transport_instr, nir_metadata_control_flow,
      (void *)transport);
}

static bool
d12_lower_flat_output_transport_instr(nir_builder *b,
                                      nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned first = nir_intrinsic_component(intr);
   nir_def *value = intr->src[0].ssa;
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   bool progress = false;

   b->cursor = nir_before_instr(&intr->instr);
   for (unsigned i = 0; i < value->num_components; i++) {
      nir_def *component = nir_channel(b, value, i);
      unsigned type = d12_flat_transport_type(data, sem.location, first + i);
      components[i] = type == 1 ? nir_u2f32(b, component)
                                : type == 2 ? nir_i2f32(b, component)
                                            : component;
      progress |= type != 0;
   }

   if (!progress)
      return false;

   nir_src_rewrite(&intr->src[0],
                   nir_vec(b, components, value->num_components));
   nir_intrinsic_set_src_type(intr, nir_type_float32);
   return true;
}

static bool
d12_nir_lower_flat_output_transport(
   nir_shader *nir, const struct d12_flat_transport *transport)
{
   if (!transport || !transport->enabled)
      return false;

   return nir_shader_intrinsics_pass(
      nir, d12_lower_flat_output_transport_instr, nir_metadata_control_flow,
      (void *)transport);
}

static void
d12_gather_flat_transport(const void *bytecode, size_t size,
                          const nir_shader *nir,
                          struct d12_flat_transport *transport)
{
   memset(transport, 0, sizeof(*transport));

   const uint64_t generic_inputs =
      nir->info.inputs_read & ~BITFIELD64_MASK(VARYING_SLOT_VAR0);
   transport->flat_outputs =
      generic_inputs &
      ~(nir->info.linear_varyings | nir->info.perspective_varyings);

   struct d12_input_sig_elem sig[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr = d12_read_signature(bytecode, size, D12_SIG_INPUT, sig,
                                    ARRAY_SIZE(sig));
   bool has_integer = false, has_float = false;

   for (unsigned i = 0; i < nr; i++) {
      /* DXIL signature IDs name semantic elements, while start_row is the
       * packed interstage register used by loadInput. DXBC exposes the same
       * value in both fields. Multiple mixed-type elements may therefore
       * share one varying slot at different start components. */
      unsigned location = VARYING_SLOT_VAR0 + sig[i].register_index;
      if (location >= 64 ||
          !(transport->flat_outputs & BITFIELD64_BIT(location)))
         continue;

      if (sig[i].component_type == 3)
         has_float = true;
      else if (sig[i].component_type == 1 || sig[i].component_type == 2)
         has_integer = true;
      else
         continue;

      for (unsigned c = 0; c < sig[i].component_count; c++) {
         unsigned component = sig[i].start_component + c;
         if (component >= 4)
            continue;
         if (sig[i].component_type == 1)
            transport->uint_components[component] |=
               BITFIELD64_BIT(location);
         else if (sig[i].component_type == 2)
            transport->sint_components[component] |=
               BITFIELD64_BIT(location);
      }
   }

   /* Preserve the native path unless flat integer and float inputs coexist. */
   transport->enabled = has_integer && has_float;
}

/*
 * Merge one stage's declared texture shapes into the PSO's table. The
 * stages must agree on a shared slot: the draw path packs each descriptor
 * with one dimension (same rule d12_dxil_collect_texture_shapes applies
 * within a stage).
 */
static bool
d12_merge_texture_shapes(uint8_t *pso_shapes, const uint8_t *stage_shapes)
{
   for (unsigned i = 0; i < D12_MAX_RESOURCE_BINDINGS; i++) {
      if (stage_shapes[i] == D12_TEXTURE_SHAPE_NONE)
         continue;
      if (pso_shapes[i] != D12_TEXTURE_SHAPE_NONE &&
          pso_shapes[i] != stage_shapes[i])
         return false;
      pso_shapes[i] = stage_shapes[i];
   }
   return true;
}

/*
 * `o` is zeroed by the caller except o->use_fs_epilog, which carries the
 * fixed-function request in and the resolved split decision out. Everything
 * in it is a function of the inputs here, so a cached stage replays it.
 */
static bool
compile_gfx_stage(struct d12_device *device, const struct d12_binding_map *map,
                  const void *dxil, size_t dxil_size, mesa_shader_stage stage,
                  struct agx_tilebuffer_layout *tib, uint16_t api_sample_mask,
                  uint8_t a8_mask, const struct d12_vs_fetch *fetch,
                  bool allow_flat_transport, bool lossless_flat_transport,
                  struct d12_stage_outputs *o, struct agx_shader_part *out)
{
   const uint32_t descriptor_generation =
      (uint32_t)agx_chip_descriptor_generation(device->dev.chip);
   struct d12_native_descriptor_layout native_layout;
   if (!d12_native_descriptor_layout(descriptor_generation, &native_layout)) {
      mesa_loge("graphics shader: unsupported native descriptor generation "
                "%u", descriptor_generation);
      return false;
   }

   nir_shader *nir;
   if (!dxil && stage == MESA_SHADER_FRAGMENT) {
      /* D3D12 permits a null pixel shader, notably for depth-only draws.
       * AGX still needs a fragment program bound, so compile the same empty
       * NIR stage Honeykrisp uses for API-level null fragment shaders.
       */
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "d12 empty fragment shader");
      nir = b.shader;
   } else if (!dxil && stage == MESA_SHADER_VERTEX) {
      /* A MESH pipeline has no vertex shader. Its rasterization vertex
       * stage is built separately into pso->ms_rast, and the draw path
       * binds ms_rast/ms_uvs/ms_varyings rather than vs/uvs/varyings, so
       * nothing compiled here is ever used by a mesh draw. The ordinary
       * graphics path still wants a vertex stage to link and to lay the
       * tilebuffer out against, so compile an empty one.
       *
       * The mesh shader must NOT be handed to this slot instead: its
       * container declares DXIL kind 13, which d12_dxil_container_stage
       * truthfully reports as COMPUTE, and the stage-equality check in
       * dxil_to_nir_stage then refuses the whole pipeline.
       */
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "d12 empty vertex shader");
      /* The native compiler requires a hardware VS to export position,
       * even for this unused graphics-layout placeholder. */
      if (device->dev.chip == AGX_CHIP_G17P) {
         nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1),
                          nir_imm_int(&b, 0), .write_mask = 0xf,
                          .io_semantics = {.location = VARYING_SLOT_POS});
      }
      nir = b.shader;
   } else {
      nir = dxil_to_nir_stage(map, descriptor_generation, dxil, dxil_size,
                              stage, o->texture_shapes);
   }
   if (!nir)
      return false;

   /* The bytecode front ends lower I/O directly to intrinsics and therefore
    * set compact clip/cull sizes themselves. nir_shader_gather_info cannot
    * reconstruct those array lengths from variables (there are none), so
    * retain them across the generic metadata refresh below. */
   const uint8_t translated_clip_size = nir->info.clip_distance_array_size;
   const uint8_t translated_cull_size = nir->info.cull_distance_array_size;
   if (stage != MESA_SHADER_FRAGMENT)
      o->cull_distance_size = nir->info.cull_distance_array_size;

   /* AGX exposes integer pixel coordinates. D3D12, like Vulkan, defines
    * SV_Position.xy at the sample position (pixel centre without sample
    * shading), so add the standard half-pixel/sample offset before the
    * backend splits frag_coord into pixel-coordinate and Z/W loads. */
   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_lower_wpos_center);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, d12_nir_lower_a8_outputs, a8_mask);

   /*
    * The DXIL and DXBC front ends do not gather shader-wide metadata.  In
    * particular, fragment UAV atomics otherwise leave writes_memory false,
    * causing AGX preprocessing to omit the side-effect lowering and compile
    * the shader with tag writes disabled.  Honeykrisp gathers this at the
    * same boundary, before agx_preprocess_nir.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   o->writes_memory = nir->info.writes_memory;
   o->root_only = nir->info.num_textures == 0 && nir->info.num_images == 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex ||
                (instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(instr)->intrinsic ==
                    nir_intrinsic_bindless_image_agx))
               o->root_only = false;
         }
      }
   }

   nir->info.clip_distance_array_size =
      MAX2(nir->info.clip_distance_array_size, translated_clip_size);
   nir->info.cull_distance_array_size =
      MAX2(nir->info.cull_distance_array_size, translated_cull_size);

   if (stage == MESA_SHADER_FRAGMENT && allow_flat_transport) {
      d12_gather_flat_transport(dxil, dxil_size, nir, &o->flat_transport);
      o->flat_transport.bitcast = lossless_flat_transport;
      NIR_PASS(_, nir, d12_nir_lower_flat_input_transport,
               &o->flat_transport);
   }

   /* Only the native G17P fragment sample consumes the packed sampler word
    * directly; every other stage keeps the side-table emulation. */
   const struct d12_sampler_lowering sampler_lowering = {
      .native_layout = &native_layout,
      .plain_static_samplers =
         (device->dev.chip == AGX_CHIP_G17P && map &&
          stage == MESA_SHADER_FRAGMENT) ? map->plain_static_samplers : 0,
   };
   NIR_PASS(_, nir, d12_nir_lower_sampler_lod_bias, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_custom_point_borders, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_sampler_reduction, &sampler_lowering);
   NIR_PASS(_, nir, d12_nir_lower_bindless_sampler_map);

   /* The DXBC front end reads every pixel-shader input register as a vec4
    * whatever its declared mask (HelloTexture's TEXCOORD is v1.xy), so the
    * fragment would bind coefficients for components the vertex shader
    * never wrote. G17P's native coefficient table refuses such a binding;
    * load only what the shader reads, as the DXIL front end already does. */
   if (device->dev.chip == AGX_CHIP_G17P && stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_shrink_vectors, false);

   unsigned runtime_samplers = d12_nir_sampler_state_count(nir, map);
   if (runtime_samplers == UINT_MAX) {
      mesa_loge("graphics shader sampler state range exceeds the 16 API "
                "sampler limit");
      ralloc_free(nir);
      return false;
   }

   o->outputs_written = nir->info.outputs_written;

   if (stage == MESA_SHADER_FRAGMENT && api_sample_mask != 0xff) {
      /* Match Honeykrisp's FS prolog: the D3D12 PSO sample mask kills the
       * samples it does not cover before the application shader runs. */
      nir_builder b =
         nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
      nir_demote_samples(
         &b, nir_imm_intN_t(&b, api_sample_mask ^ 0xff, 16));
      nir->info.fs.uses_discard = true;
   }

   bool split_fs = false;
   if (stage == MESA_SHADER_FRAGMENT) {
      /* A shader-written coverage mask has to execute before tilebuffer
       * stores.  A monolithic NIR shader may legally write SV_Target before
       * SV_Coverage; lowering that literally would place sample_mask after
       * store_local_pixel, which AGX explicitly forbids.  Honeykrisp's split
       * path gives the required ordering: the API shader applies coverage and
       * exports colour, then the epilog writes the tilebuffer. */
      const bool writes_coverage =
         nir->info.outputs_written &
         BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      split_fs = (o->use_fs_epilog || writes_coverage) &&
                 (nir->info.outputs_written &
                  BITFIELD64_RANGE(FRAG_RESULT_DATA0, 8));
      o->use_fs_epilog = split_fs;
   }

   NIR_PASS(_, nir, agx_nir_lower_texture);
   if (device->dev.chip == AGX_CHIP_G17P)
      agx_preprocess_nir_g17p(nir);
   else
      agx_preprocess_nir(nir);
   NIR_PASS(_, nir, d12_dxil_lower_root_mask, api_sample_mask);

   if (stage == MESA_SHADER_FRAGMENT && split_fs) {
      /*
       * Blending reads the old tile value, so it lives in the standard AGX
       * fragment epilog rather than in the API shader. Export the API
       * shader's colours/depth/stencil for that epilog and leave both parts
       * unterminated; agx_fast_link appends the final stop.
       *
       * This is the same split Honeykrisp uses. Keeping it conditional means
       * the established monolithic path remains byte-for-byte unchanged for
       * the overwhelmingly common no-blend case.
       */
      bool coverage_to_epilog = false;
      NIR_PASS(_, nir, d12_nir_lower_coverage_to_epilog,
               &coverage_to_epilog);
      o->fs_epilog_link.sample_mask_after_force_early |= coverage_to_epilog;
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog,
               &o->fs_epilog_link);
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
      NIR_PASS(_, nir, agx_nir_lower_sample_mask);
      NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);
      NIR_PASS(_, nir, agx_nir_lower_interpolation, false);

      /* The passes above may introduce driver-owned sysvals. */
      NIR_PASS(_, nir, d12_dxil_lower_root_mask, api_sample_mask);
      NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, nir, agx_nir_lower_texture);
   } else if (stage == MESA_SHADER_FRAGMENT) {
      unsigned bindless_base = 0;
      NIR_PASS(_, nir, agx_nir_lower_tilebuffer, tib, NULL, &bindless_base,
               NULL, NULL, false, false);
      /* Depth/stencil-ref writes become zs_emit, a coverage write
       * becomes demote_samples, and discards become sample_mask writes.
       * hk orders these the same way (hk_shader.c): zs_emit before the
       * sample-mask lowering, everything before MSAA lowering.
       */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
      NIR_PASS(_, nir, agx_nir_lower_sample_mask);
      NIR_PASS(_, nir, agx_nir_lower_monolithic_msaa,
               MAX2(tib->nr_samples, 1));
      NIR_PASS(_, nir, agx_nir_lower_interpolation, false);
      /* The passes above can introduce driver-owned sysvals
       * (api_sample_mask, shader_part_tests_zs); rerun the lowering that
       * resolves them. No-op when none appeared.
       */
      NIR_PASS(_, nir, d12_dxil_lower_root_mask, api_sample_mask);
      NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, nir, agx_nir_lower_texture);

      /* G17P has no USC texture/sampler words: a fragment sample goes
       * through the native two-pointer ABI or not at all. Rewrite the one
       * admitted shape here, after agx_nir_lower_texture assigned the
       * sampler handle; the draw publishes the pointers. Anything else
       * keeps the ordinary bindless form and is refused at the draw. */
      if (device->dev.chip == AGX_CHIP_G17P) {
         /* The handle offset (slot * stride) and the +1 sampler index are
          * still ALU chains here; the pass needs them as constants. */
         NIR_PASS(_, nir, nir_opt_constant_folding);
         NIR_PASS(_, nir, nir_opt_dce);
         struct d12_nir_g17p_fragment_sample binding;
         if (d12_nir_lower_g17p_fragment_sample(nir, descriptor_generation,
                                                &binding) ||
             d12_nir_lower_g17p_graphics_resources(nir, descriptor_generation,
                                                  &binding)) {
            o->g17p_fragment_sample = true;
            o->g17p_fragment_fetch = binding.indexed_fetch;
            o->g17p_indexed_samplers = binding.indexed_samplers;
            o->g17p_sample_slot = binding.sampler_slot;
            o->g17p_texture_offset_B = binding.texture_offset_B;
         }
      }
   } else {
      /* Real vertex attributes arrive through the prolog ABI: the main
       * shader reads registers the prolog exported, so load_input must
       * become load_exported_agx here (hk_shader.c does the same). A
       * DRAW with a nonempty input layout also needs an attribute-
       * fetching prolog keyed from that layout, which does not exist
       * yet: the static_vi prolog only fixes up vertex/instance IDs.
       */
      /* D3D12 has no point size: a point is always one pixel. AGX reads
       * the varying regardless, so a shader that never writes it
       * rasterises points at whatever was left there -- measured as
       * either nothing at all or the entire render target. hk lowers the
       * same default for its hardware vertex stage (hk_shader.c:1135).
       */
      NIR_PASS(_, nir, nir_lower_default_point_size);

      /*
       * With a real input layout the shader FETCHES its attributes, and
       * the prolog goes back to being nothing but a vertex-ID fixup.
       *
       * The alternative -- agx_nir_lower_vs_input_to_prolog plus an
       * attribute-fetching prolog -- is what Honeykrisp does when its
       * pipeline is built in pieces, and it is not available to us: that
       * prolog answers load_vbo_base_agx from AGX_ABI_VUNI_VBO_BASE,
       * uniform register 4*n counting from ZERO, which runs straight
       * through D12_UNIFORM_ROOT at 4. hk takes this same monolithic path
       * whenever it can (hk_shader.c, !use_prolog).
       *
       * Strides are DYNAMIC because D3D12 puts StrideInBytes in the
       * VERTEX BUFFER VIEW, which is draw state, not in the input layout,
       * which is pipeline state. Baking it into the key would mean
       * recompiling whenever an application rebinds the same geometry
       * with a different stride.
       */
      if (fetch && fetch->nr) {
         struct agx_robustness rs = {
            .soft_fault = agx_has_soft_fault(&device->dev),
            .level = AGX_ROBUSTNESS_D3D,
         };

         NIR_PASS(_, nir, agx_nir_lower_vbo, fetch->keys, rs, true);
         NIR_PASS(_, nir, d12_dxil_lower_vbo_params);
      } else {
         NIR_PASS(_, nir, agx_nir_lower_vs_input_to_prolog);
      }

      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
               NULL);
      NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);
      o->cull_distance_size = nir->info.cull_distance_array_size;
      NIR_PASS(_, nir, agx_nir_lower_uvs, &o->uvs, false);

      /* Indexed output stores read their UVS index through
       * load_uvs_index_agx, which the driver must lower. Honeykrisp
       * defers it to draw time through its root descriptor because
       * linking happens there; our assignment is deterministic from the
       * unlinked layout (agx_assign_uvs with the same zero masks the PSO
       * uses), so the index is a compile-time constant. The PSO-level
       * agx_assign_uvs later recomputes the identical layout.
       */
      struct agx_varyings_vs linked = {0};
      agx_assign_uvs(&linked, &o->uvs, 0, 0);
      NIR_PASS(_, nir, d12_dxil_lower_uvs_index, linked.slots);
      struct d12_nir_g17p_fragment_sample binding = {0};
      const bool native_resources = descriptor_generation == 9 &&
         (d12_nir_lower_g17p_txf(nir, 9) ||
          d12_nir_lower_g17p_graphics_resources(nir, 9, &binding));
      if (native_resources) {
         NIR_PASS(_, nir, d12_nir_shift_g17p_txf_root);
         o->g17p_fragment_sample = true;
         o->g17p_fragment_fetch = true;
         o->g17p_indexed_samplers = binding.indexed_samplers;
      }

   }

   if (stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, d12_dxil_lower_sample_positions,
               d12_default_sample_positions(MAX2(tib->nr_samples, 1)));
   }

   struct agx_shader_key key = {
      .dev = agx_gather_device_key(&device->dev),
      /* Allocas live in per-thread scratch, and large shaders spill. */
      .has_scratch = true,
      /* Native stage publication below admits only the no-CProg ABI. */
      .no_preamble = device->dev.chip == AGX_CHIP_G17P,
      /* Constant promotion is still outside this texture-binding change;
       * see the validation note on the compute key above. */
      .promote_constants = false,
      /* Honeykrisp's proven path promotes texture descriptors into a
       * preamble. This avoids leaving a bindless texture-handle load in the
       * fragment main. The compiled part's preamble must be retained below;
       * agx_fast_link intentionally copies only main/prolog/epilog. */
      .promote_textures = true,
      .reserved_preamble = D12_UNIFORM_RESERVED,
      /* A split fragment main falls through into its epilog. All other
       * programs are self-contained and terminate normally. */
      .no_stop = stage == MESA_SHADER_FRAGMENT && split_fs,
   };

   /* Root uploads only invalidate a stage that consumes the root pointer.
    * Inspect lowered NIR, including driver-generated shader code, rather
    * than assuming the root signature's visibility describes actual use. */
   const unsigned root_base = stage == MESA_SHADER_FRAGMENT
                                 ? D12_UNIFORM_FRAGMENT_ROOT
                                 : o->g17p_fragment_fetch
                                      ? AGX_G17P_TXF_ROOT_HALFS
                                      : D12_UNIFORM_ROOT;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_preamble) {
               unsigned base = nir_intrinsic_base(intr);
               unsigned end =
                  base + intr->def.num_components * intr->def.bit_size / 16;
               o->reads_root |=
                  base < root_base + D12_UNIFORM_ROOT_REGS && end > root_base;
            }
         }
      }
   }

   memset(out, 0, sizeof(*out));
   const bool compiled = agx_compile_shader_nir(nir, &key, out);
   if (!compiled)
      mesa_loge("graphics shader compile refused: %s", agx_last_compile_refusal());
   out->info.uses_sampler_heap |= d12_binding_map_uses_sampler_heap(map);
   /* The native two-pointer sample consumes no USC sampler state: the
    * sampler is a pointer the Ioto program loads, not a state-file entry. */
   out->info.sampler_state_count =
      o->g17p_indexed_samplers
         ? MAX2(runtime_samplers, map ? map->sampler_count + 1 : 1)
         : o->g17p_fragment_sample
            ? (o->g17p_fragment_fetch ? 1 : 0)
            : MAX2(out->info.sampler_state_count, runtime_samplers);
   d12_report_scratch("shader", &out->info);

   if (d12_debug()->dump_shaders && out->binary) {
      fprintf(stderr, "; --- %s stage, push_count %u, preamble %d ---\n",
              _mesa_shader_stage_to_string(stage), out->info.push_count,
              out->info.has_preamble);
      if (out->info.has_preamble) {
         fprintf(stderr, "; preamble +0x%x\n", out->info.preamble_offset);
         d12_disassemble_shader(device->dev.chip == AGX_CHIP_G17P, (uint8_t *)out->binary +
                             out->info.preamble_offset,
                          out->info.main_offset - out->info.preamble_offset,
                          stderr);
      }
      fprintf(stderr, "; main +0x%x\n", out->info.main_offset);
      d12_disassemble_shader(device->dev.chip == AGX_CHIP_G17P, (uint8_t *)out->binary + out->info.main_offset,
                       out->info.binary_size - out->info.main_offset, stderr);
   }

   ralloc_free(nir);
   return compiled && out->binary && out->info.binary_size;
}

/* Optional compiler artifacts. A failed compile still owns its final NIR and
 * records the backend's exact refusal; it must never look like certification. */
static void
d12_dump_native_stage(nir_shader *nir, const struct agx_shader_part *part,
                      bool compiled)
{
   const char *dir = getenv("D3D12AGX_NATIVE_STAGE_ARTIFACTS");
   if (!dir || !*dir)
      return;
   static int serial;
   unsigned id = p_atomic_inc_return(&serial);
   char path[4096];
   snprintf(path, sizeof(path), "%s/stage-%03u.nir", dir, id);
   FILE *f = fopen(path, "w");
   if (!f)
      return;
   nir_print_shader(nir, f);
   fclose(f);
   snprintf(path, sizeof(path), "%s/stage-%03u.name", dir, id);
   f = fopen(path, "w");
   if (f) {
      fprintf(f, "%s\n", nir->info.name ? nir->info.name : "unnamed");
      fclose(f);
   }
   snprintf(path, sizeof(path), "%s/stage-%03u.refusal", dir, id);
   f = fopen(path, "w");
   if (f) {
      const char *reason = compiled ? "none" : agx_last_compile_refusal();
      fprintf(f, "%s\n", reason && *reason ? reason : "no compiler reason recorded");
      fclose(f);
   }
   const struct agx_shader_info *i = &part->info;
   snprintf(path, sizeof(path), "%s/stage-%03u.json", dir, id);
   f = fopen(path, "w");
   if (f) {
      fprintf(f, "{\"compiled\":%s,\"stage\":%u,\"binary_size\":%u,"
                 "\"main_offset\":%u,\"main_size\":%u,\"gprs\":%u,"
                 "\"workgroup\":[%u,%u,%u],\"push_halfs\":%u,"
                 "\"staging_known\":%s,\"staging_halfs\":%u,"
                 "\"preamble\":%s,\"preamble_gprs\":%u,"
                 "\"preamble_size\":%u,\"scratch\":%u,"
                 "\"preamble_scratch\":%u,\"spectpr_scratch\":%u,"
                 "\"shared_bytes\":%u,\"textures\":%u,\"samplers\":%u,"
                 "\"uses_txf\":%s,\"uses_sampler_heap\":%s,"
                 "\"uniform_dma\":%s,\"rodata\":[%u,%u,%u],"
                 "\"uniform_halfs_valid\":%s,\"uniform_halfs\":[",
              compiled ? "true" : "false", nir->info.stage, i->binary_size,
              i->main_offset, i->main_size, i->nr_gprs,
              i->workgroup_size[0], i->workgroup_size[1], i->workgroup_size[2],
              i->push_count, i->agx3_staging_known ? "true" : "false",
              i->agx3_staging_halfs, i->has_preamble ? "true" : "false",
              i->nr_preamble_gprs, i->preamble_size, i->scratch_size,
              i->preamble_scratch_size, i->agx3_spectpr_scratch_size,
              i->local_size, i->texture_state_count, i->sampler_state_count,
              i->uses_txf ? "true" : "false",
              i->uses_sampler_heap ? "true" : "false",
              i->agx3_uniform_dma_valid ? "true" : "false",
              i->rodata.offset, i->rodata.base_uniform, i->rodata.size_16,
              i->agx3_uniform_halfs_valid ? "true" : "false");
      bool first = true;
      for (unsigned h = 0; h < AGX3_CPROG_INPUT_HALVES; ++h) {
         if (!(i->agx3_uniform_halfs[h / 64] & BITFIELD64_BIT(h % 64)))
            continue;
         fprintf(f, "%s%u", first ? "" : ",", h);
         first = false;
      }
      fprintf(f, "]}\n");
      fclose(f);
   }
   if (compiled && part->binary && i->binary_size) {
      snprintf(path, sizeof(path), "%s/stage-%03u.bin", dir, id);
      f = fopen(path, "wb");
      if (f) {
         fwrite(part->binary, 1, i->binary_size, f);
         fclose(f);
      }
   }
}

/* Same CODE producer as API compute. Internal stage launches remain closed;
 * the trial Ioto uses synthetic cell addresses and is never submitted. P10
 * supplies recording-owned cells, ordering and retained parameter buffers. */
static HRESULT
d12_upload_internal_compute(struct d12_device *device,
                             const struct agx_shader_part *part,
                             struct agx_bo **bo, uint64_t *va,
                             struct agx_shader_info *info,
                             struct d12_native_compute_abi *abi,
                             const char *name)
{
   if (device->dev.chip != AGX_CHIP_G17P) {
      *bo = agx_bo_create(&device->dev, part->info.binary_size, 0,
                          AGX_BO_EXEC | AGX_BO_LOW_VA, name);
      if (!*bo)
         return E_OUTOFMEMORY;
      if (!d12_upload_executable(*bo, part->binary, part->info.binary_size))
         return E_FAIL;
      *va = (*bo)->va->addr;
      *info = part->info;
      return S_OK;
   }

   struct d12_native_compute_abi staged = {0};
   if (!d12_native_compute_uniforms(&part->info, &staged.required_cells)) {
      mesa_loge("%s: native driver uniform declaration refused", name);
      return E_NOTIMPL;
   }
   HRESULT hr = d12_native_compute_code_create(device, part, bo, va,
                                               &staged.rodata_va);
   if (FAILED(hr)) {
      mesa_loge("%s: native CODE metadata refused (%#x)", name, (unsigned)hr);
      return hr;
   }
   uint64_t cells[D12_NATIVE_COMPUTE_CELLS];
   for (unsigned slot = 0; slot < ARRAY_SIZE(cells); ++slot)
      cells[slot] = UINT64_C(0x10000000) + slot * 8;
   staged.certified = true;
   struct agx3_g17p_compute_program trial;
   if (!d12_native_compute_generate(&part->info, &staged, *va,
                                     (*bo)->va->addr, cells, &trial)) {
      mesa_loge("%s: native Ioto/resource metadata refused", name);
      agx_bo_unreference(&device->dev, *bo);
      *bo = NULL;
      *va = 0;
      return E_NOTIMPL;
   }
   *info = part->info;
   *abi = staged;
   mesa_logi("%s: native CODE/Ioto certified, main=%u gprs=%u cells=%#x; "
             "native internal-stage ABI", name, info->main_size,
             info->nr_gprs, abi->required_cells);
   return S_OK;
}

/*
 * Finish one of the programs poly_nir_lower_gs produced. `uvs` non-NULL
 * marks the RASTERIZATION program, which is a hardware vertex shader and
 * therefore needs the unified-vertex-store lowering; the others are
 * compute. Takes ownership of `nir` and frees it.
 */
static bool
compile_poly_program(struct d12_device *device, nir_shader *nir,
                     struct agx_unlinked_uvs_layout *uvs,
                     struct agx_shader_part *out)
{
   if (device->dev.chip == AGX_CHIP_G17P)
      NIR_PASS(_, nir, d12_dxil_lower_compute_quad_derivatives);
   /* Keep small DXIL function-local payload temporaries in SSA. Native
    * preraster submission has no scratch producer, and scalar allocas do
    * not require one once their memory accesses have been promoted. */
   if (device->dev.chip == AGX_CHIP_G17P && nir->scratch_size <= 256) {
      NIR_PASS(_, nir, nir_lower_scratch_to_var);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_function_temp, 256);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   }
   /* DXIL global stores/atomics need the same side-effect metadata as API
    * compute and graphics stages, including execution-mask lowering. */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   NIR_PASS(_, nir, agx_nir_lower_texture);
   const bool native = device->dev.chip == AGX_CHIP_G17P;
   if (native)
      agx_preprocess_nir_g17p(nir);
   else
      agx_preprocess_nir(nir);
   NIR_PASS(_, nir, d12_dxil_lower_root);

   if (uvs) {
      /* This program is the HARDWARE vertex stage, so it needs the whole
       * hardware-VS lowering sequence, not just the unified vertex
       * store. hk_lower_hw_vs (hk_shader.c:1045) is the reference and the
       * order is load-bearing: agx_nir_lower_uvs runs last, over scalar
       * outputs. Handing it vec4 stores -- which is what the geometry
       * rasterization program emits -- produced a render command that
       * died outright, leaving the target not even cleared.
       */
      NIR_PASS(_, nir, nir_lower_default_point_size);
      NIR_PASS(_, nir, nir_lower_point_size, 1.0f, 511.95f);
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
               NULL);
      NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);

      NIR_PASS(_, nir, agx_nir_lower_uvs, uvs, false);

      struct agx_varyings_vs linked = {0};
      agx_assign_uvs(&linked, uvs, 0, 0);
      NIR_PASS(_, nir, d12_dxil_lower_uvs_index, linked.slots);
   }

   struct agx_shader_key key = {
      .dev = agx_gather_device_key(&device->dev),
      .has_scratch = true,
      .no_preamble = native,
      .promote_constants = false,
      .reserved_preamble = D12_UNIFORM_RESERVED,
      .no_stop = false,
   };

   memset(out, 0, sizeof(*out));
   const bool compiled = agx_compile_shader_nir(nir, &key, out);
   if (native)
      d12_dump_native_stage(nir, out, compiled);
   if (!compiled) {
      mesa_loge("native/Poly program compile refused: %s",
                agx_last_compile_refusal());
   }
   d12_report_scratch("shader", &out->info);

   if (d12_debug()->dump_shaders && out->binary) {
      fprintf(stderr, "; --- poly %s%s, push_count %u ---\n",
              nir->info.name ? nir->info.name : "program",
              uvs ? " (raster)" : "", out->info.push_count);
      fprintf(stderr, "; main +0x%x\n", out->info.main_offset);
      d12_disassemble_shader(device->dev.chip == AGX_CHIP_G17P, (uint8_t *)out->binary + out->info.main_offset,
                       out->info.binary_size - out->info.main_offset,
                       stderr);
   }

   ralloc_free(nir);
   return compiled && out->binary && out->info.binary_size;
}

/* Build NIR's byte-addressed transform-feedback description from D3D12's
 * semantic declaration. D3D12 packs declaration entries per OutputSlot in
 * declaration order; a NULL semantic is an explicit gap. */
static nir_xfb_info *
d12_build_xfb_info(void *mem_ctx,
                   const D3D12_STREAM_OUTPUT_DESC *so,
                   const void *bytecode, size_t bytecode_size,
                   bool *valid)
{
   *valid = false;
   if (!so || !so->NumEntries) {
      *valid = true;
      return NULL;
   }

   if (!so->pSODeclaration || so->NumEntries > 64 ||
       so->NumStrides > NIR_MAX_XFB_BUFFERS ||
       (so->NumStrides && !so->pBufferStrides) ||
       (so->RasterizedStream != D3D12_SO_NO_RASTERIZED_STREAM &&
        so->RasterizedStream >= NIR_MAX_XFB_STREAMS))
      return NULL;

   struct d12_input_sig_elem sig[D12_MAX_SIG_ELEMENTS];
   bool dxbc = d12_dxbc_is_dxbc_shader(bytecode, bytecode_size);
   unsigned nr_sig =
      dxbc ? d12_dxbc_output_signature(bytecode, bytecode_size, sig,
                                       ARRAY_SIZE(sig))
           : d12_dxil_output_signature(bytecode, bytecode_size, sig,
                                       ARRAY_SIZE(sig));
   if (!nr_sig)
      return NULL;

   unsigned output_count = 0;
   for (unsigned i = 0; i < so->NumEntries; i++)
      output_count += so->pSODeclaration[i].SemanticName != NULL;
   if (!output_count)
      return NULL;

   nir_xfb_info *xfb =
      rzalloc_size(mem_ctx, nir_xfb_info_size(output_count));
   if (!xfb)
      return NULL;

   uint16_t packed_offset[NIR_MAX_XFB_BUFFERS] = {0};
   bool buffer_seen[NIR_MAX_XFB_BUFFERS] = {0};

   for (unsigned i = 0; i < so->NumEntries; i++) {
      const D3D12_SO_DECLARATION_ENTRY *entry = &so->pSODeclaration[i];
      if (entry->Stream >= NIR_MAX_XFB_STREAMS ||
          entry->OutputSlot >= NIR_MAX_XFB_BUFFERS ||
          entry->StartComponent > 3 || !entry->ComponentCount ||
          entry->StartComponent + entry->ComponentCount > 4 ||
          entry->OutputSlot >= so->NumStrides)
         goto fail;

      unsigned buffer = entry->OutputSlot;
      unsigned bytes = entry->ComponentCount * 4;
      if (packed_offset[buffer] > UINT16_MAX - bytes)
         goto fail;

      if (!entry->SemanticName) {
         packed_offset[buffer] += bytes;
         continue;
      }

      const struct d12_input_sig_elem *match = NULL;
      for (unsigned s = 0; s < nr_sig; s++) {
         if (sig[s].semantic_index == entry->SemanticIndex &&
             !strcasecmp(sig[s].name, entry->SemanticName)) {
            match = &sig[s];
            break;
         }
      }
      if (!match || match->id >= 32 ||
          match->start_component + entry->StartComponent +
                entry->ComponentCount >
             4)
         goto fail;

      if (buffer_seen[buffer] &&
          xfb->buffer_to_stream[buffer] != entry->Stream)
         goto fail;

      unsigned component = match->start_component + entry->StartComponent;
      unsigned location;
      if (!strcasecmp(match->name, "SV_Position")) {
         location = VARYING_SLOT_POS;
      } else if (dxbc && !strcasecmp(match->name, "SV_ClipDistance")) {
         location = VARYING_SLOT_CLIP_DIST0 + match->semantic_index;
      } else {
         location = VARYING_SLOT_VAR0 + match->id;
      }

      nir_xfb_output_info *out = &xfb->outputs[xfb->output_count++];
      *out = (nir_xfb_output_info){
         .buffer = buffer,
         .offset = packed_offset[buffer],
         .location = location,
         .component_mask = BITFIELD_RANGE(component, entry->ComponentCount),
         .component_offset = component,
      };

      xfb->buffers[buffer].varying_count++;
      xfb->buffers_written |= BITFIELD_BIT(buffer);
      xfb->streams_written |= BITFIELD_BIT(entry->Stream);
      xfb->buffer_to_stream[buffer] = entry->Stream;
      buffer_seen[buffer] = true;
      packed_offset[buffer] += bytes;
   }

   for (unsigned b = 0; b < so->NumStrides; b++) {
      UINT stride = so->pBufferStrides[b];
      if ((stride & 3) || stride > UINT16_MAX || packed_offset[b] > stride)
         goto fail;
      xfb->buffers[b].stride = stride;
   }

   *valid = true;
   return xfb;

fail:
   ralloc_free(xfb);
   return NULL;
}

static enum mesa_prim
d12_so_primitive(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
   switch (type) {
   case D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT:    return MESA_PRIM_POINTS;
   case D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE:     return MESA_PRIM_LINES;
   case D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE: return MESA_PRIM_TRIANGLES;
   default:                                     return MESA_PRIM_UNKNOWN;
   }
}

/* MIT-derived from Honeykrisp's passthrough-GS builder. The DXBC/DXIL
 * front ends use intrinsic I/O rather than variables, so Mesa's generic
 * variable-based nir_create_passthrough_gs cannot preserve their slots. */
static nir_shader *
d12_build_so_passthrough_gs(uint64_t outputs, uint8_t clip_size,
                            uint8_t cull_size, enum mesa_prim prim,
                            const nir_xfb_info *xfb, bool handle_flat)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "d12 stream-output passthrough");
   nir_shader *s = b.shader;

   enum mesa_prim out_prim = prim == MESA_PRIM_POINTS
                                ? MESA_PRIM_POINTS
                                : prim == MESA_PRIM_LINES
                                     ? MESA_PRIM_LINE_STRIP
                                     : MESA_PRIM_TRIANGLE_STRIP;
   unsigned vertices = mesa_vertices_per_prim(prim);

   s->info.inputs_read = outputs;
   s->info.outputs_written = outputs;
   s->info.clip_distance_array_size = clip_size;
   s->info.cull_distance_array_size = cull_size;
   s->info.gs.input_primitive = prim;
   s->info.gs.output_primitive = out_prim;
   s->info.gs.vertices_in = vertices;
   s->info.gs.vertices_out = vertices;
   s->info.gs.invocations = 1;
   s->info.gs.active_stream_mask = 1;

   if (xfb) {
      s->xfb_info = ralloc_memdup(s, xfb,
                                  nir_xfb_info_size(xfb->output_count));
      s->info.has_transform_feedback_varyings = true;
      for (unsigned i = 0; i < NIR_MAX_XFB_BUFFERS; i++)
         s->info.xfb_stride[i] = xfb->buffers[i].stride / 4;
   }

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *one = nir_imm_int(&b, 1);
   nir_def *flat_mask = handle_flat ? nir_load_flat_mask(&b) : NULL;
   nir_def *provoking = NULL;
   if (handle_flat) {
      nir_def *last = nir_imm_int(&b, vertices - 1);
      provoking = nir_bcsel(&b,
                            nir_ine_imm(&b, nir_load_provoking_last(&b), 0),
                            last, zero);
   }

   for (unsigned v = 0; v < vertices; v++) {
      nir_def *vertex = nir_imm_int(&b, v);

      u_foreach_bit64(loc, outputs) {
         unsigned adjusted_loc = loc;
         nir_def *offset = zero;
         unsigned num_slots = 1;
         bool scalar = loc == VARYING_SLOT_LAYER ||
                       loc == VARYING_SLOT_VIEW_INDEX ||
                       loc == VARYING_SLOT_VIEWPORT ||
                       loc == VARYING_SLOT_PSIZ;
         unsigned comps = scalar ? 1 : 4;

         if (loc == VARYING_SLOT_CLIP_DIST1 ||
             loc == VARYING_SLOT_CULL_DIST1) {
            adjusted_loc--;
            offset = one;
         }
         if (adjusted_loc == VARYING_SLOT_CLIP_DIST0 ||
             adjusted_loc == VARYING_SLOT_CULL_DIST0) {
            num_slots = clip_size + cull_size;
            comps = loc > adjusted_loc ? num_slots - 4 : MIN2(num_slots, 4);
         }

         nir_io_semantics sem = {
            .location = adjusted_loc,
            .num_slots = num_slots,
         };
         nir_def *input_vertex = vertex;
         if (handle_flat && loc != VARYING_SLOT_POS) {
            nir_def *is_flat = nir_ine_imm(
               &b, nir_iand_imm(&b, flat_mask, BITFIELD64_BIT(loc)), 0);
            input_vertex = nir_bcsel(&b, is_flat, provoking, vertex);
         }
         nir_def *value = nir_load_per_vertex_input(
            &b, comps, 32, input_vertex, offset, .io_semantics = sem);
         for (unsigned c = 0; c < comps; c++) {
            nir_store_output(&b, nir_channel(&b, value, c), offset,
                             .io_semantics = sem,
                             .src_type = nir_type_uint32, .component = c);
         }
      }
      nir_emit_vertex(&b, 0);
   }

   return s;
}

/*
 * A geometry shader expands into the main body and rasterization shader,
 * plus count/pre-GS programs when transform feedback needs them. The draw
 * path dispatches the latter only for XFB, matching hk_cmd_draw.c.
 */
static bool
compile_gs_nir(struct d12_device *device, nir_shader *nir,
               struct poly_gs_info *gs_info,
               struct agx_unlinked_uvs_layout *rast_uvs,
               const struct d12_flat_transport *flat_transport,
               struct agx_shader_part *count_out,
               struct agx_shader_part *pre_out,
               struct agx_shader_part *main_out,
               struct agx_shader_part *rast_out)
{
   nir_shader *gs_count = NULL, *gs_copy = NULL, *pre_gs = NULL;
   NIR_PASS(_, nir, d12_nir_lower_flat_output_transport, flat_transport);
   NIR_PASS(_, nir, poly_nir_lower_gs, &gs_count, &gs_copy, &pre_gs, gs_info);

   /* poly clones the three derived programs with no ralloc parent, so each
    * one is released separately: compile_poly_program frees what it is
    * handed, the rest are freed here. */
   bool ok = false;
   memset(count_out, 0, sizeof(*count_out));
   memset(pre_out, 0, sizeof(*pre_out));
   memset(main_out, 0, sizeof(*main_out));
   memset(rast_out, 0, sizeof(*rast_out));

   if (gs_info->shape < POLY_GS_SHAPE_DYNAMIC_INDEXED ||
       gs_info->shape > POLY_GS_SHAPE_STATIC_PER_INSTANCE) {
      mesa_loge("graphics PSO: geometry lowering produced invalid "
                "rasterization shape %d", gs_info->shape);
      goto out;
   }

   if (!gs_copy) {
      mesa_loge("graphics PSO: geometry lowering produced no rasterization "
                "program");
      goto out;
   }

   /* Poly keeps GS metadata while lowering its main/count variants. After
    * that lowering, native CDM programs need the actual compute stage and
    * local sizes used by d12_list_gs_prerast. The pre-GS patch-up is one
    * invocation; main/count use the existing 64-thread preraster ABI. */
   if (device->dev.chip == AGX_CHIP_G17P) {
      nir_shader *compute[] = {nir, gs_count, pre_gs};
      for (unsigned i = 0; i < ARRAY_SIZE(compute); ++i) {
         if (!compute[i])
            continue;
         compute[i]->info.stage = MESA_SHADER_COMPUTE;
         memset(&compute[i]->info.cs, 0, sizeof(compute[i]->info.cs));
         compute[i]->xfb_info = NULL;
         compute[i]->info.has_transform_feedback_varyings = false;
         compute[i]->info.workgroup_size[0] = i == 2 ? 1 : 64;
         compute[i]->info.workgroup_size[1] = 1;
         compute[i]->info.workgroup_size[2] = 1;
      }
   }

   if (gs_count) {
      const bool compiled =
         compile_poly_program(device, gs_count, NULL, count_out);
      gs_count = NULL;
      if (!compiled)
         goto out;
   }
   if (pre_gs) {
      const bool compiled = compile_poly_program(device, pre_gs, NULL, pre_out);
      pre_gs = NULL;
      if (!compiled)
         goto out;
   }
   if (gs_info->xfb && !pre_out->binary) {
      mesa_loge("graphics PSO: transform feedback produced no pre-GS program");
      goto out;
   }

   {
      const bool compiled = compile_poly_program(device, nir, NULL, main_out);
      nir = NULL;
      if (!compiled)
         goto out;
   }

   ok = compile_poly_program(device, gs_copy, rast_uvs, rast_out);
   gs_copy = NULL;

out:
   ralloc_free(gs_count);
   ralloc_free(pre_gs);
   ralloc_free(gs_copy);
   ralloc_free(nir);
   return ok;
}

/*
 * A geometry shader following tessellation inherits the ID of the input patch,
 * not the index of the primitive generated by the tessellator.  Poly normally
 * lowers a GS primitive ID to its compute invocation X coordinate, which is
 * the latter.  Carry the patch ID through the software TES output buffer so
 * each generated primitive can recover the former from any of its vertices.
 */
static void
d12_nir_export_tess_patch_id(nir_shader *nir)
{
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_io_semantics sem = {
      .location = VARYING_SLOT_PRIMITIVE_ID,
      .num_slots = 1,
   };

   nir_store_output(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0),
                    .write_mask = 1, .src_type = nir_type_uint32,
                    .io_semantics = sem);
   nir->info.outputs_written |= VARYING_BIT_PRIMITIVE_ID;
}

static bool
d12_nir_lower_tess_gs_patch_id(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_primitive_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = {
      .location = VARYING_SLOT_PRIMITIVE_ID,
      .num_slots = 1,
   };
   nir_def *patch_id =
      nir_load_per_vertex_input(b, 1, intr->def.bit_size, nir_imm_int(b, 0),
                                nir_imm_int(b, 0), .io_semantics = sem);

   nir_def_replace(&intr->def, patch_id);
   return true;
}

static bool
compile_gs_stage(struct d12_device *device, const struct d12_binding_map *map,
                 const void *dxil, size_t dxil_size,
                 const nir_xfb_info *xfb,
                 D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology_type,
                 const struct d12_pipeline_state *pso,
                 struct poly_gs_info *gs_info,
                 struct agx_unlinked_uvs_layout *rast_uvs,
                 struct agx_shader_part *count_out,
                 struct agx_shader_part *pre_out,
                 struct agx_shader_part *main_out,
                 struct agx_shader_part *rast_out)
{
   nir_shader *nir =
      dxil_to_nir_stage(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         dxil, dxil_size, MESA_SHADER_GEOMETRY, NULL);
   if (!nir)
      return false;

   enum mesa_prim input = nir->info.gs.input_primitive;
   bool compatible = false;
   if (pso->has_tess) {
      /* Tessellation produces points, lines (isolines), or triangles.
       * Unlike an IA topology, its output never carries adjacency. */
      enum mesa_prim tess_output =
         pso->tess_points ? MESA_PRIM_POINTS
         : pso->tess_mode == TESS_PRIMITIVE_ISOLINES ? MESA_PRIM_LINES
                                                     : MESA_PRIM_TRIANGLES;
      compatible = input == tess_output;
   } else {
      switch (primitive_topology_type) {
      case D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT:
         compatible = input == MESA_PRIM_POINTS;
         break;
      case D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE:
         compatible = input == MESA_PRIM_LINES ||
                      input == MESA_PRIM_LINES_ADJACENCY;
         break;
      case D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE:
         compatible = input == MESA_PRIM_TRIANGLES ||
                      input == MESA_PRIM_TRIANGLES_ADJACENCY;
         break;
      default:
         compatible = false;
         break;
      }
   }

   if (!compatible) {
      mesa_loge("graphics PSO: geometry input primitive %u is incompatible "
                "with topology type %u%s", input, primitive_topology_type,
                pso->has_tess ? " after tessellation" : "");
      ralloc_free(nir);
      return false;
   }

   if (xfb) {
      nir->xfb_info = ralloc_memdup(nir, xfb,
                                    nir_xfb_info_size(xfb->output_count));
      nir->info.has_transform_feedback_varyings = true;
      for (unsigned i = 0; i < NIR_MAX_XFB_BUFFERS; i++)
         nir->info.xfb_stride[i] = xfb->buffers[i].stride / 4;
   }

   if (pso->has_tess) {
      nir->info.inputs_read |= VARYING_BIT_PRIMITIVE_ID;
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               d12_nir_lower_tess_gs_patch_id, nir_metadata_control_flow,
               NULL);
   }

   return compile_gs_nir(device, nir, gs_info, rast_uvs, NULL, count_out,
                         pre_out, main_out, rast_out);
}

/*
 * The hull shader. It runs as compute, one workgroup per patch, writing
 * per-vertex and per-patch outputs into the TCS buffer. Everything the
 * runtime needs about the patch layout has to be gathered BEFORE the
 * lowering, which rewrites the output stores into global stores and
 * takes the shader's tess info with it.
 */
static bool
compile_tcs_stage(struct d12_device *device, const struct d12_binding_map *map,
                  const void *dxil, size_t dxil_size,
                  struct d12_pipeline_state *pso,
                  struct agx_shader_part *out)
{
   nir_shader *nir =
      dxil_to_nir_stage(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         dxil, dxil_size, MESA_SHADER_TESS_CTRL, NULL);
   if (!nir)
      return false;

   pso->tess_ccw |= nir->info.tess.ccw;
   pso->tess_points |= nir->info.tess.point_mode;
   if (nir->info.tess.spacing)
      pso->tess_spacing = nir->info.tess.spacing;
   if (nir->info.tess._primitive_mode)
      pso->tess_mode = nir->info.tess._primitive_mode;

   pso->tcs_output_patch_size = nir->info.tess.tcs_vertices_out;
   pso->tcs_per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
   pso->tcs_nr_patch_outputs = util_last_bit(nir->info.patch_outputs_written);
   pso->tcs_output_stride = poly_tcs_output_stride(nir);

   /* DXBC fork/join phases can require more invocations than there are output
    * control points. The translator records the largest phase width; DXIL has
    * no phase-width metadata here and therefore falls back to one invocation
    * per output control point. */
   nir->info.workgroup_size[0] =
      MAX2(nir->info.workgroup_size[0], pso->tcs_output_patch_size);
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   NIR_PASS(_, nir, poly_nir_lower_tcs);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));

   return compile_poly_program(device, nir, NULL, out);
}

/*
 * The domain shader, as the HARDWARE vertex shader: the tessellator
 * produces an index buffer and the hardware invokes this once per index,
 * which is what poly_nir_lower_tes(to_hw_vs = true) sets up.
 */
static bool
compile_tes_stage(struct d12_device *device, const struct d12_binding_map *map,
                  const void *dxil, size_t dxil_size,
                  struct d12_pipeline_state *pso,
                  struct agx_unlinked_uvs_layout *uvs,
                  struct agx_shader_part *out)
{
   nir_shader *nir =
      dxil_to_nir_stage(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         dxil, dxil_size, MESA_SHADER_TESS_EVAL, NULL);
   if (!nir)
      return false;

   /* Gather before lowering: it destroys info. */
   pso->tess_ccw |= nir->info.tess.ccw;
   pso->tess_points |= nir->info.tess.point_mode;
   if (nir->info.tess.spacing)
      pso->tess_spacing = nir->info.tess.spacing;
   if (nir->info.tess._primitive_mode)
      pso->tess_mode = nir->info.tess._primitive_mode;

   NIR_PASS(_, nir, poly_nir_lower_tes, true);

   return compile_poly_program(device, nir, uvs, out);
}

/*
 * The domain shader immediately ahead of a geometry/XFB stage.  The
 * tessellator still produces an indexed indirect draw, but instead of feeding
 * the hardware vertex stage directly, the domain shader runs as compute and
 * writes the ordinary poly vertex-output buffer consumed by the geometry
 * stage.  This is the same tessellation -> software TES -> GS composition
 * used by Honeykrisp.
 */
static bool
compile_sw_tes_stage(struct d12_device *device,
                     const struct d12_binding_map *map, const void *dxil,
                     size_t dxil_size, struct agx_shader_part *out,
                     uint64_t *out_outputs, uint8_t *out_clip_size,
                     uint8_t *out_cull_size, bool export_patch_id)
{
   nir_shader *nir =
      dxil_to_nir_stage(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         dxil, dxil_size, MESA_SHADER_TESS_EVAL, NULL);
   if (!nir)
      return false;

   if (export_patch_id)
      d12_nir_export_tess_patch_id(nir);

   NIR_PASS(_, nir, poly_nir_lower_tes, false);

   *out_outputs = nir->info.outputs_written;
   if (out_clip_size)
      *out_clip_size = nir->info.clip_distance_array_size;
   if (out_cull_size)
      *out_cull_size = nir->info.cull_distance_array_size;

   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;
   nir->info.workgroup_size[0] = 64;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   return compile_poly_program(device, nir, NULL, out);
}

/*
 * The vertex shader ahead of a geometry shader does not feed the
 * rasterizer -- it writes its outputs to a buffer the GS reads. That
 * makes it a compute shader, and poly_nir_lower_vs_before_gs is what
 * rewrites its stores into that layout. It must run before
 * agx_preprocess_nir, like every other poly lowering.
 */
static bool
compile_sw_vs_stage(struct d12_device *device,
                    const struct d12_binding_map *map, const void *dxil,
                    size_t dxil_size, const struct d12_vs_fetch *fetch,
                    struct agx_shader_part *out, uint64_t *out_outputs,
                    uint8_t *out_clip_size, uint8_t *out_cull_size)
{
   nir_shader *nir =
      dxil_to_nir_stage(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         dxil, dxil_size, MESA_SHADER_VERTEX, NULL);
   if (!nir)
      return false;

   /* Preserve D3D12's raw SV_VertexID before the VBO lowering inserts its
    * separate effective vertex-id load.  The marker is resolved after that
    * insertion and before Poly creates its own zero-based output index. */
   NIR_PASS(_, nir, d12_dxil_mark_sw_vs_vertex_id);

   /* CAPTURE THE OUTPUT MASK FIRST. The lowering rewrites every
    * store_output into a global store, so after it runs the shader has no
    * NIR outputs at all and agx_shader_info::outputs reports 0. That mask
    * is not decoration: the geometry program derives EVERY vertex address
    * in the output buffer from it, so a zero mask sends every vertex to
    * the same address and the rasterization program reads a position of
    * (0,0,0,0) -- w=0, which rasterises as the entire render target.
    * hk captures it in the same place, before the branch
    * (hk_shader.c:1138).
    */
   *out_outputs = nir->info.outputs_written;
   if (out_clip_size)
      *out_clip_size = nir->info.clip_distance_array_size;
   if (out_cull_size)
      *out_cull_size = nir->info.cull_distance_array_size;

   /*
    * VERTEX ATTRIBUTES, and this has to happen while the shader is still
    * a VERTEX shader -- agx_nir_lower_vbo asserts on the stage, and the
    * switch to compute is a few lines below.
    *
    * Skipping it is why every geometry and tessellation test that read a
    * real vertex attribute aborted the process with "Unhandled intrinsic
    * load_input": the hardware VS path grew attribute fetch in G10, and
    * the SOFTWARE VS that a GS or tessellation pipeline substitutes for
    * it did not. Our own g6/g7 probes never caught it because they
    * derive their geometry from SV_VertexID.
    *
    * The emitted load_vertex_id is resolved further down by
    * d12_dxil_lower_sw_vs, which turns it into the dispatch's invocation
    * index pulled through the index buffer.
    */
   if (fetch && fetch->nr) {
      struct agx_robustness rs = {
         .soft_fault = agx_has_soft_fault(&device->dev),
         .level = AGX_ROBUSTNESS_D3D,
      };

      NIR_PASS(_, nir, agx_nir_lower_vbo, fetch->keys, rs, true);
      NIR_PASS(_, nir, d12_dxil_lower_vbo_params);
   }

   NIR_PASS(_, nir, d12_dxil_lower_sw_vs_vertex_id);

   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);

   /* The pass does NOT rewrite the stage -- the caller must, and the AGX
    * backend asserts on it ("only for SW VS"). hk does exactly this
    * immediately after the same pass (hk_shader.c:1143-1147).
    */
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;

   /* The launch programs a local size of 64; say so, so the backend's
    * thread-index maths agrees with what the hardware is told. */
   nir->info.workgroup_size[0] = 64;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   /* A compute shader has no vertex ID. poly_nir_lower_sw_vs is what
    * turns load_vertex_id into global_invocation_id.x (pulling through
    * the index buffer and adding the base vertex) and load_instance_id
    * into .y. Without it the shader keeps asking for a sysval that does
    * not exist in a dispatch, every thread computes the same address,
    * and the vertex output buffer stays zero -- which then rasterises as
    * a position of (0,0,0,0).
    *
    * It emits load_index_size_poly / load_vertex_param_buffer_poly /
    * load_first_vertex, so it has to run before d12_dxil_lower_root
    * resolves those.
    */
   NIR_PASS(_, nir, d12_dxil_lower_sw_vs);

   return compile_poly_program(device, nir, NULL, out);
}

/* Compiler diagnostic only: an API signature refusal must not hide a stage
 * compiler refusal. This entry is restricted to an actual registered noop fd;
 * it uses the same stage lowerings/uploads as the ordinary graphics PSO and
 * never grants command recording or physical admission. */
HRESULT
d12_pipeline_state_certify_native_gs(struct d12_device *device,
                                     const struct d12_binding_map *map,
                                     const void *vs, size_t vs_size,
                                     const void *gs, size_t gs_size,
                                     ID3D12PipelineState **out)
{
#ifndef _WIN32
   extern bool asahi_noop_d3d12_graphics_qualification_allowed(int fd)
      __attribute__((weak, visibility("default")));
   if (!device || !out || device->dev.chip != AGX_CHIP_G17P ||
       !asahi_noop_d3d12_graphics_qualification_allowed ||
       !asahi_noop_d3d12_graphics_qualification_allowed(device->dev.fd))
      return E_NOTIMPL;
   *out = NULL;
   struct d12_pipeline_state *pso = calloc(1, sizeof(*pso));
   if (!pso)
      return E_OUTOFMEMORY;
   struct agx_shader_part sw = {0}, count = {0}, pre = {0}, main = {0}, rast = {0};
   struct agx_shader_part prolog = {0};
   HRESULT hr = E_FAIL;
   if (!compile_sw_vs_stage(device, map, vs, vs_size, NULL, &sw,
                             &pso->sw_vs_outputs, NULL, NULL) ||
       !compile_gs_stage(device, map, gs, gs_size, NULL,
                         D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT, pso,
                         &pso->gs_info, &pso->gs_uvs,
                         &count, &pre, &main, &rast))
      goto done;
   /* P09's initial certificate is the original static point GS, no XFB. */
   if (count.binary || !pre.binary ||
       pso->gs_info.shape != POLY_GS_SHAPE_STATIC_PER_INSTANCE)
      goto done;
   hr = d12_upload_internal_compute(device, &sw, &pso->sw_vs_bo,
      &pso->sw_vs_va, &pso->sw_vs_info, &pso->native_stages[D12_NATIVE_SW_VS],
      "d3d12 sw vs");
   if (FAILED(hr))
      goto done;
   hr = d12_upload_internal_compute(device, &pre, &pso->gs_pre_bo,
      &pso->gs_pre_va, &pso->gs_pre_info,
      &pso->native_stages[D12_NATIVE_GS_PRE], "d3d12 gs pre");
   if (FAILED(hr))
      goto done;
   hr = d12_upload_internal_compute(device, &main, &pso->gs_main_bo,
      &pso->gs_main_va, &pso->gs_main_info,
      &pso->native_stages[D12_NATIVE_GS_MAIN], "d3d12 gs main");
   if (FAILED(hr))
      goto done;

   struct agx_vs_prolog_key pk = {.hw = true, .static_vi = true};
   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   agx_nir_vs_prolog(&b, &pk);
   struct agx_shader_key key = {
      .dev = agx_gather_device_key(&device->dev), .has_scratch = true,
      .secondary = true, .no_stop = true,
   };
   bool compiled = agx_compile_shader_nir(b.shader, &key, &prolog);
   ralloc_free(b.shader);
   if (!compiled || !prolog.binary ||
       !agx_fast_link(&pso->gs_rast.linked, &device->dev, false,
                      &rast, &prolog, NULL, 0)) {
      hr = E_FAIL;
      goto done;
   }
   pso->gs_rast.info = rast.info;
   pso->has_gs = true;
   pso->iface.lpVtbl = (ID3D12PipelineStateVtbl *)&d12_pso_vtbl;
   pso->refcount = 1;
   pso->device = device;
   pso->lifetime = 1;
   d12_device_add_ref(device);
   d12_private_data_store_init(&pso->private_data, (IUnknown *)&pso->iface);
   *out = &pso->iface;
   hr = S_OK;
done:
   free(sw.binary);
   free(count.binary);
   free(pre.binary);
   free(main.binary);
   free(rast.binary);
   free(prolog.binary);
   if (FAILED(hr)) {
      d12_pso_release_bos(device, pso);
      free(pso);
   }
   return hr;
#else
   return E_NOTIMPL;
#endif
}

/*
 * D3D12 comparison functions map one-for-one onto the AGX ZS functions, but
 * write the mapping out rather than relying on the constant offset: a silent
 * off-by-one here means NEVER where ALWAYS was asked for, which rejects every
 * fragment and looks like a missing draw rather than a wrong enum.
 */
static enum agx_zs_func
d12_translate_compare(D3D12_COMPARISON_FUNC func)
{
   switch (func) {
   case D3D12_COMPARISON_FUNC_NEVER:         return AGX_ZS_FUNC_NEVER;
   case D3D12_COMPARISON_FUNC_LESS:          return AGX_ZS_FUNC_LESS;
   case D3D12_COMPARISON_FUNC_EQUAL:         return AGX_ZS_FUNC_EQUAL;
   case D3D12_COMPARISON_FUNC_LESS_EQUAL:    return AGX_ZS_FUNC_LEQUAL;
   case D3D12_COMPARISON_FUNC_GREATER:       return AGX_ZS_FUNC_GREATER;
   case D3D12_COMPARISON_FUNC_NOT_EQUAL:     return AGX_ZS_FUNC_NOT_EQUAL;
   case D3D12_COMPARISON_FUNC_GREATER_EQUAL: return AGX_ZS_FUNC_GEQUAL;
   case D3D12_COMPARISON_FUNC_ALWAYS:        return AGX_ZS_FUNC_ALWAYS;
   default:
      /* Includes _NONE (0), which D3D12 uses to mean "no test". */
      return AGX_ZS_FUNC_ALWAYS;
   }
}

static enum agx_stencil_op
d12_translate_stencil_op(D3D12_STENCIL_OP op)
{
   switch (op) {
   case D3D12_STENCIL_OP_KEEP:     return AGX_STENCIL_OP_KEEP;
   case D3D12_STENCIL_OP_ZERO:     return AGX_STENCIL_OP_ZERO;
   case D3D12_STENCIL_OP_REPLACE:  return AGX_STENCIL_OP_REPLACE;
   case D3D12_STENCIL_OP_INCR_SAT: return AGX_STENCIL_OP_INCR_SAT;
   case D3D12_STENCIL_OP_DECR_SAT: return AGX_STENCIL_OP_DECR_SAT;
   case D3D12_STENCIL_OP_INVERT:   return AGX_STENCIL_OP_INVERT;
   case D3D12_STENCIL_OP_INCR:     return AGX_STENCIL_OP_INCR_WRAP;
   case D3D12_STENCIL_OP_DECR:     return AGX_STENCIL_OP_DECR_WRAP;
   default:                        return AGX_STENCIL_OP_KEEP;
   }
}

static void
d12_translate_stencil_face(struct d12_stencil_face *out,
                           const D3D12_DEPTH_STENCILOP_DESC *in)
{
   out->compare = d12_translate_compare(in->StencilFunc);
   out->pass = d12_translate_stencil_op(in->StencilPassOp);
   out->depth_fail = d12_translate_stencil_op(in->StencilDepthFailOp);
   out->stencil_fail = d12_translate_stencil_op(in->StencilFailOp);
}

static bool
d12_translate_blend_factor(D3D12_BLEND in, enum pipe_blendfactor *out)
{
   static_assert(D3D12_BLEND_ALPHA_FACTOR == D12_CORE_BLEND_ALPHA_FACTOR &&
                    D3D12_BLEND_INV_ALPHA_FACTOR ==
                       D12_CORE_BLEND_INV_ALPHA_FACTOR,
                 "public and portable alpha blend factors must match");

   switch (in) {
   case D3D12_BLEND_ZERO:             *out = PIPE_BLENDFACTOR_ZERO; break;
   case D3D12_BLEND_ONE:              *out = PIPE_BLENDFACTOR_ONE; break;
   case D3D12_BLEND_SRC_COLOR:        *out = PIPE_BLENDFACTOR_SRC_COLOR; break;
   case D3D12_BLEND_INV_SRC_COLOR:    *out = PIPE_BLENDFACTOR_INV_SRC_COLOR; break;
   case D3D12_BLEND_SRC_ALPHA:        *out = PIPE_BLENDFACTOR_SRC_ALPHA; break;
   case D3D12_BLEND_INV_SRC_ALPHA:    *out = PIPE_BLENDFACTOR_INV_SRC_ALPHA; break;
   case D3D12_BLEND_DEST_ALPHA:       *out = PIPE_BLENDFACTOR_DST_ALPHA; break;
   case D3D12_BLEND_INV_DEST_ALPHA:   *out = PIPE_BLENDFACTOR_INV_DST_ALPHA; break;
   case D3D12_BLEND_DEST_COLOR:       *out = PIPE_BLENDFACTOR_DST_COLOR; break;
   case D3D12_BLEND_INV_DEST_COLOR:   *out = PIPE_BLENDFACTOR_INV_DST_COLOR; break;
   case D3D12_BLEND_SRC_ALPHA_SAT:    *out = PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE; break;
   case D3D12_BLEND_BLEND_FACTOR:     *out = PIPE_BLENDFACTOR_CONST_COLOR; break;
   case D3D12_BLEND_INV_BLEND_FACTOR: *out = PIPE_BLENDFACTOR_INV_CONST_COLOR; break;
   case D3D12_BLEND_SRC1_COLOR:       *out = PIPE_BLENDFACTOR_SRC1_COLOR; break;
   case D3D12_BLEND_INV_SRC1_COLOR:   *out = PIPE_BLENDFACTOR_INV_SRC1_COLOR; break;
   case D3D12_BLEND_SRC1_ALPHA:       *out = PIPE_BLENDFACTOR_SRC1_ALPHA; break;
   case D3D12_BLEND_INV_SRC1_ALPHA:   *out = PIPE_BLENDFACTOR_INV_SRC1_ALPHA; break;
   case D3D12_BLEND_ALPHA_FACTOR:
   case D3D12_BLEND_INV_ALPHA_FACTOR: {
      const enum d12_alpha_blend_factor_kind alpha =
         d12_alpha_blend_factor_classify((uint32_t)in);
      *out = alpha == D12_ALPHA_BLEND_FACTOR_CONSTANT
                ? PIPE_BLENDFACTOR_CONST_ALPHA
                : PIPE_BLENDFACTOR_INV_CONST_ALPHA;
      break;
   }
   default:
      return false;
   }

   return true;
}

static bool
d12_blend_factor_uses_src1(D3D12_BLEND factor)
{
   switch (factor) {
   case D3D12_BLEND_SRC1_COLOR:
   case D3D12_BLEND_INV_SRC1_COLOR:
   case D3D12_BLEND_SRC1_ALPHA:
   case D3D12_BLEND_INV_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

static bool
d12_translate_blend_op(D3D12_BLEND_OP in, enum pipe_blend_func *out)
{
   switch (in) {
   case D3D12_BLEND_OP_ADD:          *out = PIPE_BLEND_ADD; break;
   case D3D12_BLEND_OP_SUBTRACT:     *out = PIPE_BLEND_SUBTRACT; break;
   case D3D12_BLEND_OP_REV_SUBTRACT: *out = PIPE_BLEND_REVERSE_SUBTRACT; break;
   case D3D12_BLEND_OP_MIN:          *out = PIPE_BLEND_MIN; break;
   case D3D12_BLEND_OP_MAX:          *out = PIPE_BLEND_MAX; break;
   default:
      return false;
   }

   return true;
}

static bool
d12_translate_logic_op(D3D12_LOGIC_OP in, enum pipe_logicop *out)
{
   switch (in) {
   case D3D12_LOGIC_OP_CLEAR:         *out = PIPE_LOGICOP_CLEAR; break;
   case D3D12_LOGIC_OP_SET:           *out = PIPE_LOGICOP_SET; break;
   case D3D12_LOGIC_OP_COPY:          *out = PIPE_LOGICOP_COPY; break;
   case D3D12_LOGIC_OP_COPY_INVERTED: *out = PIPE_LOGICOP_COPY_INVERTED; break;
   case D3D12_LOGIC_OP_NOOP:          *out = PIPE_LOGICOP_NOOP; break;
   case D3D12_LOGIC_OP_INVERT:        *out = PIPE_LOGICOP_INVERT; break;
   case D3D12_LOGIC_OP_AND:           *out = PIPE_LOGICOP_AND; break;
   case D3D12_LOGIC_OP_NAND:          *out = PIPE_LOGICOP_NAND; break;
   case D3D12_LOGIC_OP_OR:            *out = PIPE_LOGICOP_OR; break;
   case D3D12_LOGIC_OP_NOR:           *out = PIPE_LOGICOP_NOR; break;
   case D3D12_LOGIC_OP_XOR:           *out = PIPE_LOGICOP_XOR; break;
   case D3D12_LOGIC_OP_EQUIV:         *out = PIPE_LOGICOP_EQUIV; break;
   case D3D12_LOGIC_OP_AND_REVERSE:   *out = PIPE_LOGICOP_AND_REVERSE; break;
   case D3D12_LOGIC_OP_AND_INVERTED:  *out = PIPE_LOGICOP_AND_INVERTED; break;
   case D3D12_LOGIC_OP_OR_REVERSE:    *out = PIPE_LOGICOP_OR_REVERSE; break;
   case D3D12_LOGIC_OP_OR_INVERTED:   *out = PIPE_LOGICOP_OR_INVERTED; break;
   default:
      return false;
   }

   return true;
}

/*
 * ms_setup: turn the mesh dispatch dimensions into an indirect draw.
 *
 * Each workgroup owns an index segment sized by its primitive declaration.  Drawing that
 * complete segment lets the rasterization shader derive a stable group and
 * local primitive ID without a GPU prefix sum; primitives beyond the group's
 * GPU-written count are discarded there.  The bounded group count keeps every
 * generated address inside the allocation even for GPU-written dimensions.
 */
static nir_shader *
build_ms_setup(unsigned max_prims)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "d12 mesh draw setup");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.internal = false;

   nir_def *params = nir_load_preamble(&b, 1, 64, .base = D12_UNIFORM_MESH_PARAMS);

   /* NOT the dispatch slot: an amplification shader writes its
    * DispatchMesh grid there, so the draw descriptor has its own.
    */
   nir_def *draw = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_DRAW), .align_mul = 8);

   nir_def *dims_ptr = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_GROUP_DIMS),
      .align_mul = 8);
   nir_def *dims = nir_load_global(&b, 3, 32, dims_ptr, .align_mul = 4);
   nir_def *groups = nir_imul(
      &b, nir_channel(&b, dims, 0),
      nir_imul(&b, nir_channel(&b, dims, 1), nir_channel(&b, dims, 2)));
   nir_def *capacity = nir_load_global(
      &b, 1, 32, nir_iadd_imm(&b, params, D12_MESH_OFF_CAPACITY),
      .align_mul = 8);
   groups = nir_umin(&b, groups, capacity);

   /* Three expanded vertices per declared triangle in each workgroup. */
   nir_def *idx = nir_imul_imm(&b, groups, max_prims * 3);

   nir_store_global(&b, idx, draw, .write_mask = 1, .align_mul = 4);
   nir_store_global(&b, nir_imm_int(&b, 1), nir_iadd_imm(&b, draw, 4),
                    .write_mask = 1, .align_mul = 4);
   for (unsigned i = 2; i < 5; i++) {
      nir_store_global(&b, nir_imm_int(&b, 0), nir_iadd_imm(&b, draw, i * 4),
                       .write_mask = 1, .align_mul = 4);
   }

   return b.shader;
}

/*
 * as_setup: the amplification shader hands off with DispatchMesh(x,y,z),
 * a THREAD GROUP count written to D12_MESH_OFF_DISPATCH. The mesh
 * dispatch that follows must be indirect off it, and an AGX indirect
 * grid is six words -- three THREAD counts then three local sizes -- so
 * the group-to-thread multiply cannot happen on the CPU.
 */
static nir_shader *
build_as_setup(const unsigned wg[3])
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "d12 amplification setup");
   b.shader->info.workgroup_size[0] = D12_MESH_MAX_AS_GROUPS;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.internal = false;

   nir_def *params =
      nir_load_preamble(&b, 1, 64, .base = D12_UNIFORM_MESH_PARAMS);
   nir_def *disp = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_DISPATCH),
      .align_mul = 8);
   nir_def *grid = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_GRID), .align_mul = 8);
   nir_def *mesh_dims = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_GROUP_DIMS),
      .align_mul = 8);
   nir_def *map = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_GROUP_MAP),
      .align_mul = 8);
   nir_def *as_dims_ptr =
      nir_load_global(&b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_AS_DIMS),
                      .align_mul = 8);
   nir_def *as_dims = nir_load_global(&b, 3, 32, as_dims_ptr, .align_mul = 4);
   nir_def *as_groups = nir_imul(
      &b, nir_channel(&b, as_dims, 0),
      nir_imul(&b, nir_channel(&b, as_dims, 1), nir_channel(&b, as_dims, 2)));
   as_groups = nir_umin_imm(&b, as_groups, D12_MESH_MAX_AS_GROUPS);

   nir_def *tid = nir_load_local_invocation_index(&b);
   nir_def *own_addr = nir_iadd(
      &b, disp, nir_u2u64(&b, nir_imul_imm(&b, tid, 3 * sizeof(uint32_t))));
   nir_def *own_dims = nir_load_global(&b, 3, 32, own_addr, .align_mul = 4);
   nir_def *own_count = nir_imul(
      &b, nir_channel(&b, own_dims, 0),
      nir_imul(&b, nir_channel(&b, own_dims, 1), nir_channel(&b, own_dims, 2)));
   own_count = nir_umin_imm(&b, own_count, D12_MESH_MAX_AMP_GROUPS);
   nir_def *active = nir_ult(&b, tid, as_groups);
   own_count = nir_bcsel(&b, active, own_count, nir_imm_int(&b, 0));

   nir_def *prefix = nir_imm_int(&b, 0);
   nir_def *total = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < D12_MESH_MAX_AS_GROUPS; i++) {
      nir_def *addr = nir_iadd_imm(&b, disp, i * 3 * sizeof(uint32_t));
      nir_def *dims = nir_load_global(&b, 3, 32, addr, .align_mul = 4);
      nir_def *count = nir_imul(
         &b, nir_channel(&b, dims, 0),
         nir_imul(&b, nir_channel(&b, dims, 1), nir_channel(&b, dims, 2)));
      count = nir_umin_imm(&b, count, D12_MESH_MAX_AMP_GROUPS);
      count = nir_bcsel(&b, nir_ult(&b, nir_imm_int(&b, i), as_groups), count,
                        nir_imm_int(&b, 0));
      total = nir_iadd(&b, total, count);
      prefix = nir_iadd(&b, prefix,
                        nir_bcsel(&b, nir_ult(&b, nir_imm_int(&b, i), tid),
                                  count, nir_imm_int(&b, 0)));
   }
   total = nir_umin_imm(&b, total, D12_MESH_MAX_TOTAL_GROUPS);

   nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
   nir_store_global(&b, total, mesh_dims, .write_mask = 1, .align_mul = 4);
   nir_store_global(&b, nir_imm_int(&b, 1), nir_iadd_imm(&b, mesh_dims, 4),
                    .write_mask = 1, .align_mul = 4);
   nir_store_global(&b, nir_imm_int(&b, 1), nir_iadd_imm(&b, mesh_dims, 8),
                    .write_mask = 1, .align_mul = 4);
   nir_store_global(&b, nir_imul_imm(&b, total, wg[0]), grid, .write_mask = 1,
                    .align_mul = 4);
   nir_store_global(&b, nir_imm_int(&b, wg[1]), nir_iadd_imm(&b, grid, 4),
                    .write_mask = 1, .align_mul = 4);
   nir_store_global(&b, nir_imm_int(&b, wg[2]), nir_iadd_imm(&b, grid, 8),
                    .write_mask = 1, .align_mul = 4);
   for (unsigned i = 0; i < 3; i++) {
      nir_store_global(&b, nir_imm_int(&b, wg[i]),
                       nir_iadd_imm(&b, grid, (3 + i) * 4), .write_mask = 1,
                       .align_mul = 4);
   }
   nir_pop_if(&b, NULL);

   nir_def *dx = nir_umax_imm(&b, nir_channel(&b, own_dims, 0), 1);
   nir_def *dy = nir_umax_imm(&b, nir_channel(&b, own_dims, 1), 1);
   nir_variable *child =
      nir_local_variable_create(b.impl, glsl_uint_type(), "child");
   nir_store_var(&b, child, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b);
   {
      nir_def *local = nir_load_var(&b, child);
      nir_push_if(&b, nir_uge(&b, local, own_count));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *linear = nir_iadd(&b, prefix, local);
      nir_def *emit =
         nir_iand(&b, active,
                  nir_iand(&b, nir_ult(&b, local, own_count),
                           nir_ult_imm(&b, linear, D12_MESH_MAX_TOTAL_GROUPS)));

      nir_push_if(&b, emit);
      nir_def *x = nir_umod(&b, local, dx);
      nir_def *yz = nir_udiv(&b, local, dx);
      nir_def *y = nir_umod(&b, yz, dy);
      nir_def *z = nir_udiv(&b, yz, dy);
      nir_def *entry = nir_vec4(&b, x, y, z, tid);
      nir_def *addr = nir_iadd(
         &b, map,
         nir_u2u64(&b, nir_imul_imm(&b, linear, 4 * sizeof(uint32_t))));
      nir_store_global(&b, entry, addr, .write_mask = 0xf, .align_mul = 16);
      nir_pop_if(&b, NULL);
      nir_store_var(&b, child, nir_iadd_imm(&b, local, 1), 1);
   }
   nir_pop_loop(&b, NULL);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);

   return b.shader;
}

static nir_alu_type
d12_mesh_sig_type(const struct d12_input_sig_elem *sig)
{
   switch (sig->component_type) {
   case 1:
      return nir_type_uint32;
   case 2:
      return nir_type_int32;
   case 3:
      return nir_type_float32;
   default:
      return nir_type_uint32;
   }
}

static nir_def *
d12_mesh_load_output(nir_builder *b, nir_def *base, nir_def *index,
                     unsigned slot, unsigned slots)
{
   nir_def *off = nir_imul_imm(b, index, slots * D12_MESH_SLOT_STRIDE);
   off = nir_iadd_imm(b, off, slot * D12_MESH_SLOT_STRIDE);
   return nir_load_global(b, 4, 32, nir_iadd(b, base, nir_u2u64(b, off)),
                          .align_mul = 16);
}

static void
d12_mesh_store_linked_output(nir_builder *b, nir_def *value,
                             const struct d12_input_sig_elem *producer,
                             const struct d12_input_sig_elem *consumer,
                             unsigned row, bool lower_integer_transport)
{
   unsigned location = VARYING_SLOT_VAR0 + consumer->register_index + row;
   unsigned components =
      MIN2(producer->component_count, consumer->component_count);
   components = MIN2(components, 4 - consumer->start_component);

   for (unsigned c = 0; c < components; c++) {
      nir_def *component = nir_channel(b, value, c);
      nir_alu_type src_type = d12_mesh_sig_type(consumer);
      if (lower_integer_transport && consumer->component_type == 1) {
         src_type = nir_type_float32;
      } else if (lower_integer_transport && consumer->component_type == 2) {
         src_type = nir_type_float32;
      }

      nir_store_output(b, component, nir_imm_int(b, 0),
                       .base = consumer->register_index + row,
                       .component = consumer->start_component + c,
                       .write_mask = 1, .src_type = src_type,
                       .io_semantics.location = location,
                       .io_semantics.num_slots = consumer->rows);
   }

   b->shader->info.outputs_written |= BITFIELD64_BIT(location);
}

/*
 * ms_rast is the hardware vertex shader for a mesh pipeline.  Mesh output
 * stores land in the driver's global-memory ABI, so this stage links that
 * memory back to the pixel shader by SEMANTIC, not by assuming producer and
 * consumer registers are identical.  DXIL is free to repack them: the
 * interface-matching shader, for example, writes UV_VERTEX_ID in mesh slot 2
 * but the pixel shader consumes it in register 2, component 1.
 */
static nir_shader *
d12_build_mesh_rast(const void *ms_dxil, size_t ms_size, const void *ps_dxil,
                    size_t ps_size, unsigned max_prims)
{
   struct d12_input_sig_elem output[D12_MAX_SIG_ELEMENTS] = {0};
   struct d12_input_sig_elem primitive[D12_MAX_SIG_ELEMENTS] = {0};
   struct d12_input_sig_elem input[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr_output = d12_read_signature(ms_dxil, ms_size, D12_SIG_OUTPUT,
                                           output, ARRAY_SIZE(output));
   unsigned nr_primitive = d12_read_signature(ms_dxil, ms_size, D12_SIG_PATCH,
                                              primitive, ARRAY_SIZE(primitive));
   unsigned nr_input = ps_dxil
                          ? d12_read_signature(ps_dxil, ps_size, D12_SIG_INPUT,
                                               input, ARRAY_SIZE(input))
                          : 0;
   unsigned vertex_slots = 1, primitive_slots = 1;
   for (unsigned i = 0; i < nr_output; i++)
      vertex_slots = MAX2(vertex_slots, output[i].id + output[i].rows);
   for (unsigned i = 0; i < nr_primitive; i++)
      primitive_slots =
         MAX2(primitive_slots, primitive[i].id + primitive[i].rows);
   bool has_flat_integer = false, has_flat_float = false;
   for (unsigned i = 0; i < nr_input; i++) {
      if (input[i].interpolation != 1)
         continue;
      has_flat_integer |=
         input[i].component_type == 1 || input[i].component_type == 2;
      has_flat_float |= input[i].component_type == 3;
   }
   bool lower_integer_transport = has_flat_integer && has_flat_float;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "d12 mesh rast");
   b.shader->info.internal = false;

   nir_def *params = nir_load_preamble(&b, 1, 64, .base = D12_UNIFORM_MESH_PARAMS);
   nir_def *verts = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_VERTICES),
      .align_mul = 8);
   nir_def *prims = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_PRIMITIVES),
      .align_mul = 8);
   nir_def *indices =
      nir_load_global(&b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_INDICES),
                      .align_mul = 8);
   nir_def *counts = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, params, D12_MESH_OFF_COUNTS), .align_mul = 8);

   /* Rasterize an expanded, non-indexed list.  That gives every hardware
    * vertex a stable primitive number while the original index still selects
    * the mesh shader's vertex output.  Per-primitive attributes can then be
    * duplicated onto the three vertices and consumed with flat interpolation.
    */
   nir_def *linear_vid = nir_load_vertex_id(&b);
   nir_def *prim_id = nir_udiv_imm(&b, linear_vid, 3);
   nir_def *group_id = nir_udiv_imm(&b, prim_id, max_prims);
   nir_def *local_prim = nir_umod_imm(&b, prim_id, max_prims);
   nir_def *group_counts =
      nir_iadd(&b, counts, nir_u2u64(&b, nir_imul_imm(&b, group_id, 8)));
   nir_def *primitive_count = nir_load_global(
      &b, 1, 32, nir_iadd_imm(&b, group_counts, 4), .align_mul = 4);
   nir_def *vid = nir_load_global(
      &b, 1, 32,
      nir_iadd(&b, indices, nir_u2u64(&b, nir_imul_imm(&b, linear_vid, 4))),
      .align_mul = 4);
   nir_def *provoking_vid = nir_load_global(
      &b, 1, 32,
      nir_iadd(&b, indices, nir_u2u64(&b, nir_imul_imm(&b, prim_id, 12))),
      .align_mul = 4);
   nir_def *position = NULL;

   for (unsigned i = 0; i < nr_output; i++) {
      const struct d12_input_sig_elem *out = &output[i];
      const struct d12_input_sig_elem *in =
         d12_find_signature_semantic(input, nr_input, out);

      for (unsigned row = 0; row < out->rows; row++) {
         unsigned slot = out->id + row;
         if (slot >= D12_MESH_MAX_SLOTS)
            continue;

         nir_def *value =
            d12_mesh_load_output(&b, verts, vid, slot, vertex_slots);

         if (!strcasecmp(out->name, "SV_Position")) {
            position = value;
         } else if (!strcasecmp(out->name, "SV_ClipDistance")) {
            unsigned location =
               VARYING_SLOT_CLIP_DIST0 + out->semantic_index + row;
            unsigned components = MIN2(out->component_count, 4);
            for (unsigned c = 0; c < components; c++) {
               nir_store_output(&b, nir_channel(&b, value, c),
                                nir_imm_int(&b, 0), .base = out->id + row,
                                .component = c, .write_mask = 1,
                                .src_type = nir_type_float32,
                                .io_semantics.location = location,
                                .io_semantics.num_slots = out->rows);
            }
            b.shader->info.outputs_written |= BITFIELD64_BIT(location);
            b.shader->info.clip_distance_array_size += components;
         } else if (in) {
            /* D3D12 flat vertex inputs select the first indexed vertex.
             * Replicate that value across the expanded primitive, like the
             * per-primitive records, so the coefficient setup transports a
             * constant even when its integer bits encode a float subnormal. */
            if (in->interpolation == 1)
               value = d12_mesh_load_output(&b, verts, provoking_vid, slot,
                                             vertex_slots);
            d12_mesh_store_linked_output(&b, value, out, in, row,
                                         lower_integer_transport);
         }
      }
   }

   nir_def *culled = nir_inot(&b, nir_ult(&b, local_prim, primitive_count));
   for (unsigned i = 0; i < nr_primitive; i++) {
      const struct d12_input_sig_elem *out = &primitive[i];
      const struct d12_input_sig_elem *in =
         d12_find_signature_semantic(input, nr_input, out);

      for (unsigned row = 0; row < out->rows; row++) {
         unsigned slot = out->id + row;
         if (slot >= D12_MESH_MAX_SLOTS)
            continue;

         nir_def *value =
            d12_mesh_load_output(&b, prims, prim_id, slot, primitive_slots);
         if (!strcasecmp(out->name, "SV_CullPrimitive")) {
            culled = nir_ior(&b, culled,
                             nir_ine_imm(&b, nir_channel(&b, value, 0), 0));
         } else if (!strcasecmp(out->name, "SV_RenderTargetArrayIndex")) {
            nir_store_output(&b, nir_channel(&b, value, 0), nir_imm_int(&b, 0),
                             .base = 0, .write_mask = 1,
                             .src_type = nir_type_uint32,
                             .io_semantics.location = VARYING_SLOT_LAYER,
                             .io_semantics.num_slots = 1);
            b.shader->info.outputs_written |= VARYING_BIT_LAYER;
         } else if (in) {
            d12_mesh_store_linked_output(&b, value, out, in, row,
                                         lower_integer_transport);
         }
      }
   }

   if (!position)
      position = nir_imm_vec4(&b, 0.0, 0.0, 0.0, 0.0);
   position =
      nir_bcsel(&b, culled, nir_imm_vec4(&b, 0.0, 0.0, 0.0, 0.0), position);
   nir_store_output(&b, position, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);
   b.shader->info.outputs_written |= VARYING_BIT_POS;

   return b.shader;
}

/*
 * Mesh output stores are lowered to global memory, so nir_shader_info cannot
 * discover them and outputs_written is necessarily zero.  Recover the live
 * per-vertex slots from the DXIL signature instead.  The backing-memory ABI
 * is deliberately indexed by (signature id + row), exactly like the
 * STORE_VERTEX_OUTPUT lowering, rather than by the packed output register.
 */
static uint64_t
d12_mesh_output_mask(const void *dxil, size_t size, enum d12_sig_section kind)
{
   struct d12_input_sig_elem sig[D12_MAX_SIG_ELEMENTS] = {0};
   unsigned nr = d12_read_signature(dxil, size, kind, sig, ARRAY_SIZE(sig));
   uint64_t mask = 0;

   for (unsigned i = 0; i < nr; i++) {
      for (unsigned row = 0; row < sig[i].rows; row++) {
         unsigned slot = sig[i].id + row;
         if (slot < D12_MESH_MAX_SLOTS)
            mask |= BITFIELD64_BIT(slot);
      }
   }

   return mask;
}

HRESULT
d12_pipeline_state_create_mesh(struct d12_device *device, const void *ms_dxil,
                               size_t ms_size, const void *as_dxil,
                               size_t as_size, const void *ps_dxil,
                               size_t ps_size, enum pipe_format rt_format,
                               unsigned nr_rt,
                               const DXGI_FORMAT *rt_formats,
                               unsigned nr_samples, UINT sample_mask,
                               const D3D12_DEPTH_STENCIL_DESC *ds_desc,
                               DXGI_FORMAT dsv_format,
                               const D3D12_RASTERIZER_DESC *rs_desc,
                               const D3D12_BLEND_DESC *blend_desc,
                               const struct d12_binding_map *map, REFIID riid,
                               void **out)
{
   uint32_t ms_kind, as_kind;
   if (!d12_dxil_container_kind(ms_dxil, ms_size, &ms_kind) || ms_kind != 13 ||
       (as_dxil && (!d12_dxil_container_kind(as_dxil, as_size, &as_kind) ||
                    as_kind != 14))) {
      mesa_loge("CreatePipelineState: mesh/amplification bytecode kind mismatch");
      return E_INVALIDARG;
   }
   if (!d12_validate_mesh_ps_link(ms_dxil, ms_size, ps_dxil, ps_size)) {
      mesa_loge("CreatePipelineState: incompatible mesh/pixel signatures");
      return E_INVALIDARG;
   }

   /* Build the pipeline the ordinary way first: the pixel shader, the
    * depth/stencil and rasterizer state and the tilebuffer layout are
    * all identical to a vertex pipeline. The MESH-specific programs are
    * then attached alongside, in ms_rast/ms_uvs/ms_varyings, which is what
    * the draw path binds for a mesh pipeline.
    *
    * NO vertex shader is passed. Handing the mesh bytecode to the VS slot
    * (which this did until 2026-08-10) made every DXC-produced mesh
    * container fail PSO creation: kind 13 is reported as COMPUTE, and the
    * VS stage-equality check refused it.
    *
    * A mesh shader translates to plain compute writing through
    * D12_UNIFORM_MESH_PARAMS, so it needs no lowering of its own -- but
    * nothing generates its rasterization shader either, because there is
    * no poly emulation for mesh. Both internal programs are built here.
    */
   HRESULT hr = d12_pipeline_state_create_graphics(
      device, NULL, 0, true, ps_dxil, ps_size, NULL, 0, NULL, 0, NULL, 0,
      rt_format, nr_rt, rt_formats, nr_samples, sample_mask, ds_desc,
      dsv_format, rs_desc, rs_desc ? (float)rs_desc->DepthBias : 0.0f, true,
      true, ds_desc ? ds_desc->StencilReadMask : UINT8_MAX,
      ds_desc ? ds_desc->StencilWriteMask : UINT8_MAX,
      ds_desc ? ds_desc->StencilReadMask : UINT8_MAX,
      ds_desc ? ds_desc->StencilWriteMask : UINT8_MAX, blend_desc, NULL, NULL,
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
      D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
      D3D12_PIPELINE_STATE_FLAG_NONE, map, riid, out);
   if (FAILED(hr))
      return hr;

   struct d12_pipeline_state *pso =
      d12_pso_from_iface((ID3D12PipelineState *)*out);

   /* Every compiled part is released at `fail` on both paths. */
   struct agx_shader_part ms_part = {0}, setup_part = {0}, rast_part = {0};
   struct agx_shader_part as_part = {0}, as_setup_part = {0};

   /* The mesh shader itself, as compute. */
   nir_shader *ms = dxil_to_nir_stage_vis(
      map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip), ms_dxil,
      ms_size, MESA_SHADER_COMPUTE, pso->texture_shapes,
      D3D12_SHADER_VISIBILITY_MESH);
   if (!ms) {
      hr = E_INVALIDARG;
      goto fail;
   }

   const nir_lower_compute_system_values_options mesh_csv = {
      .shuffle_local_ids_for_quad_derivatives = true,
      .lower_local_invocation_index = true,
   };
   d12_dxil_bound_mesh_workgroups(ms, false);
   if (ms->info.derivative_group == DERIVATIVE_GROUP_QUADS)
      NIR_PASS(_, ms, nir_lower_compute_system_values, &mesh_csv);

   pso->ms_workgroup[0] = MAX2(ms->info.workgroup_size[0], 1);
   pso->ms_workgroup[1] = MAX2(ms->info.workgroup_size[1], 1);
   pso->ms_workgroup[2] = MAX2(ms->info.workgroup_size[2], 1);
   const unsigned ms_payload_size = ms->info.task_payload_size;
   pso->ms_max_vertices = MAX2(ms->info.mesh.max_vertices_out, 1);
   pso->ms_max_primitives = MAX2(ms->info.mesh.max_primitives_out, 1);
   pso->ms_outputs = d12_mesh_output_mask(ms_dxil, ms_size, D12_SIG_OUTPUT);
   pso->ms_primitive_outputs =
      d12_mesh_output_mask(ms_dxil, ms_size, D12_SIG_PATCH);
   if (!compile_poly_program(device, ms, NULL, &ms_part)) {
      hr = E_INVALIDARG;
      goto fail;
   }

   if (!compile_poly_program(device, build_ms_setup(pso->ms_max_primitives),
                             NULL, &setup_part)) {
      hr = E_INVALIDARG;
      goto fail;
   }

   const struct {
      struct agx_shader_part *part;
      struct agx_bo **bo;
      uint64_t *va;
      struct agx_shader_info *info;
      struct d12_native_compute_abi *abi;
      const char *name;
   } cs[] = {
      {&ms_part, &pso->ms_bo, &pso->ms_va, &pso->ms_info,
       &pso->native_stages[D12_NATIVE_MS], "d3d12 mesh"},
      {&setup_part, &pso->ms_setup_bo, &pso->ms_setup_va, &pso->ms_setup_info,
       &pso->native_stages[D12_NATIVE_MS_SETUP],
       "d3d12 mesh setup"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cs); i++) {
      HRESULT upload = d12_upload_internal_compute(
         device, cs[i].part, cs[i].bo, cs[i].va, cs[i].info,
         cs[i].abi, cs[i].name);
      if (FAILED(upload)) {
         hr = upload;
         goto fail;
      }
   }

   /* The rasterization vertex shader, linked like any other. */
   nir_shader *rast = d12_build_mesh_rast(ms_dxil, ms_size, ps_dxil, ps_size,
                                         pso->ms_max_primitives);
   if (!compile_poly_program(device, rast, &pso->ms_uvs, &rast_part)) {
      hr = E_INVALIDARG;
      goto fail;
   }

   struct agx_vs_prolog_key pk = {.hw = true, .static_vi = true};
   struct agx_shader_key pbk = {
      .dev = agx_gather_device_key(&device->dev),
      .has_scratch = true,
      .secondary = true,
      .no_stop = true,
   };
   const struct agx_shader_part *prolog =
      d12_shader_cache_part(device, agx_nir_vs_prolog, &pk, sizeof(pk), &pbk,
                            false);
   if (!prolog) {
      hr = E_INVALIDARG;
      goto fail;
   }

   bool linked = d12_link(&pso->ms_rast.linked, &device->dev, false,
                          &rast_part, prolog->info.binary_size ? prolog : NULL,
                          NULL, 0);
   if (!linked) {
      hr = E_OUTOFMEMORY;
      goto fail;
   }
   pso->ms_rast.info = rast_part.info;
   pso->ms_rast.g17p_root_only = true;
   pso->ms_rast.g17p_writes_memory = false;

   agx_assign_uvs(&pso->ms_varyings, &pso->ms_uvs, 0, 0);

   /* Same preamble rule as every other linked graphics stage. */
   if (rast_part.info.has_preamble) {
      pso->ms_rast.code_bo =
         agx_bo_create(&device->dev, rast_part.info.binary_size, 0,
                       AGX_BO_EXEC | AGX_BO_LOW_VA, "d3d12 mesh rast");
      if (!pso->ms_rast.code_bo) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
      if (!d12_upload_executable(pso->ms_rast.code_bo, rast_part.binary,
                                 rast_part.info.binary_size)) {
         hr = E_FAIL;
         goto fail;
      }
      pso->ms_rast.preamble_va =
         pso->ms_rast.code_bo->va->addr + rast_part.info.preamble_offset;
   }

   unsigned samplers = rast_part.info.sampler_state_count;
   if (rast_part.info.uses_txf)
      samplers = MAX2(samplers, 1);

   agx_pack(&pso->ms_rast.counts, COUNTS, cfg) {
      cfg.uniform_register_count = rast_part.info.push_count;
      cfg.preshader_register_count = rast_part.info.nr_preamble_gprs;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(samplers, false);
      cfg.texture_state_register_count = rast_part.info.texture_state_count;
   }

   /* The amplification shader, if there is one. */
   if (as_dxil && as_size) {
      nir_shader *as = dxil_to_nir_stage_vis(
         map, (uint32_t)agx_chip_descriptor_generation(device->dev.chip),
         as_dxil, as_size, MESA_SHADER_COMPUTE, pso->texture_shapes,
         D3D12_SHADER_VISIBILITY_AMPLIFICATION);
      if (!as) {
         hr = E_INVALIDARG;
         goto fail;
      }

      d12_dxil_bound_mesh_workgroups(as, true);
      if (as->info.derivative_group == DERIVATIVE_GROUP_QUADS)
         NIR_PASS(_, as, nir_lower_compute_system_values, &mesh_csv);

      pso->as_workgroup[0] = MAX2(as->info.workgroup_size[0], 1);
      pso->as_workgroup[1] = MAX2(as->info.workgroup_size[1], 1);
      pso->as_workgroup[2] = MAX2(as->info.workgroup_size[2], 1);

      /* compile_poly_program consumes the NIR. Save declarations first. */
      pso->as_payload_size = as->info.task_payload_size;
      if (pso->as_payload_size > D12_MESH_PAYLOAD_STRIDE ||
          (ms_payload_size && ms_payload_size != pso->as_payload_size)) {
         ralloc_free(as);
         hr = E_INVALIDARG;
         goto fail;
      }

      if (!compile_poly_program(device, as, NULL, &as_part) ||
          !compile_poly_program(device, build_as_setup(pso->ms_workgroup),
                                NULL, &as_setup_part)) {
         hr = E_INVALIDARG;
         goto fail;
      }

      const struct {
         struct agx_shader_part *part;
         struct agx_bo **bo;
         uint64_t *va;
         struct agx_shader_info *info;
         struct d12_native_compute_abi *abi;
         const char *name;
      } acs[] = {
         {&as_part, &pso->as_bo, &pso->as_va, &pso->as_info,
          &pso->native_stages[D12_NATIVE_AS],
          "d3d12 amplification"},
         {&as_setup_part, &pso->as_setup_bo, &pso->as_setup_va,
          &pso->as_setup_info, &pso->native_stages[D12_NATIVE_AS_SETUP],
          "d3d12 amplification setup"},
      };

      for (unsigned i = 0; i < ARRAY_SIZE(acs); i++) {
         HRESULT upload = d12_upload_internal_compute(
            device, acs[i].part, acs[i].bo, acs[i].va, acs[i].info,
            acs[i].abi, acs[i].name);
            if (FAILED(upload)) {
            hr = upload;
            goto fail;
         }
      }

      pso->has_amp = true;
      mesa_logi("  + amplification: as %u B, setup %u B, wg %ux%ux%u",
                as_part.info.binary_size, as_setup_part.info.binary_size,
                pso->as_workgroup[0], pso->as_workgroup[1],
                pso->as_workgroup[2]);
   }

   mesa_logi("compiled mesh pipeline: ms %u B, setup %u B, rast %u B, "
             "wg %ux%ux%u, outputs 0x%llx",
             ms_part.info.binary_size, setup_part.info.binary_size,
             rast_part.info.binary_size, pso->ms_workgroup[0],
             pso->ms_workgroup[1], pso->ms_workgroup[2],
             (unsigned long long)pso->ms_outputs);

   if (device->dev.chip == AGX_CHIP_G17P) {
      const struct agx_g17p_vertex_compile_info compiler_info = {
         .hardware_vertex = pso->ms_rast.info.stage == MESA_SHADER_VERTEX,
         .has_preamble = pso->ms_rast.info.has_preamble,
         .nr_preamble_gprs = pso->ms_rast.info.nr_preamble_gprs,
         .preamble_scratch_size = pso->ms_rast.info.preamble_scratch_size,
         .uvs_written = pso->ms_uvs.written,
         .position_off = pso->ms_uvs.group_offs[UVS_POSITION],
         .varyings_off = pso->ms_uvs.group_offs[UVS_VARYINGS],
         .psiz_off = pso->ms_uvs.group_offs[UVS_PSIZ],
         .layer_viewport_off = pso->ms_uvs.group_offs[UVS_LAYER_VIEWPORT],
         .clip_off = pso->ms_uvs.group_offs[UVS_CLIP_DIST],
         .uvs_size = pso->ms_uvs.size,
         .user_size = pso->ms_uvs.user_size,
      };
      if (!d12_agx3_g17p_vertex_program_info(
             device->dev.chip, &compiler_info,
             &pso->ms_rast.agx3_vdm_state_update)) {
         hr = E_NOTIMPL;
         goto fail;
      }
   }
   pso->has_mesh = true;
   hr = S_OK;

fail:
   free(ms_part.binary);
   free(setup_part.binary);
   free(rast_part.binary);
   free(as_part.binary);
   free(as_setup_part.binary);
   if (FAILED(hr)) {
      ID3D12PipelineState_Release((ID3D12PipelineState *)*out);
      *out = NULL;
   }
   return hr;
}

HRESULT
d12_pipeline_state_create_graphics(
   struct d12_device *device, const void *vs_dxil, size_t vs_size,
   bool vs_is_mesh, const void *ps_dxil, size_t ps_size, const void *gs_dxil,
   size_t gs_size, const void *hs_dxil, size_t hs_size, const void *ds_dxil,
   size_t ds_size, enum pipe_format rt_format, unsigned nr_rt,
   const DXGI_FORMAT *rt_formats, unsigned nr_samples, UINT sample_mask,
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
   D3D12_PIPELINE_STATE_FLAGS flags, const struct d12_binding_map *map,
   REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;

   if (so_desc && so_desc->NumEntries && !gs_size && !hs_size && !ds_size &&
       primitive_topology_type == D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED) {
      /* Streams may omit primitive topology. A passthrough GS's input
       * primitive is part of its compiled poly_nir program, so retain the
       * three real specializations and select from IA at draw time. */
      ID3D12PipelineState *variants[3] = {0};
      const D3D12_PRIMITIVE_TOPOLOGY_TYPE types[3] = {
         D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
         D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
         D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
      };
      HRESULT hr = S_OK;
      for (unsigned i = 0; i < ARRAY_SIZE(variants); i++) {
         hr = d12_pipeline_state_create_graphics(
            device, vs_dxil, vs_size, false, ps_dxil, ps_size, gs_dxil, gs_size,
            hs_dxil, hs_size, ds_dxil, ds_size, rt_format, nr_rt, rt_formats,
            nr_samples, sample_mask, ds_desc, dsv_format, rs_desc,
            rasterizer_depth_bias, front_stencil_enable, back_stencil_enable,
            front_stencil_read_mask, front_stencil_write_mask,
            back_stencil_read_mask, back_stencil_write_mask, blend_desc,
            input_layout, so_desc, types[i], strip_cut, flags, map,
            &IID_ID3D12PipelineState, (void **)&variants[i]);
         if (FAILED(hr)) {
            for (unsigned j = 0; j < i; j++)
               ID3D12PipelineState_Release(variants[j]);
            return hr;
         }
      }
      struct d12_pipeline_state *point = d12_pso_from_iface(variants[0]);
      point->so_dynamic_topology = true;
      point->so_line = d12_pso_from_iface(variants[1]);
      point->so_triangle = d12_pso_from_iface(variants[2]);
      hr = d12_pso_QueryInterface(variants[0], riid, out);
      ID3D12PipelineState_Release(variants[0]);
      return hr;
   }

   if ((vs_is_mesh ? (vs_dxil || vs_size) : (!vs_dxil || !vs_size)) ||
       (!ps_dxil != !ps_size))
      return E_INVALIDARG;
   if (strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED &&
       strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF &&
       strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF)
      return E_INVALIDARG;
   /* Skipped for mesh: the pixel shader's inputs are produced by the mesh
    * shader, which this validator has no vertex-side signature to match
    * against. d12_pipeline_state_create_mesh checks the MS itself. */
   if (!vs_is_mesh && !d12_validate_graphics_signatures(
                         vs_dxil, vs_size, ps_dxil, ps_size, gs_dxil, gs_size,
                         hs_dxil, hs_size, ds_dxil, ds_size, input_layout)) {
      mesa_loge("CreateGraphicsPipelineState: incompatible shader signatures");
      return E_INVALIDARG;
   }

   struct d12_pipeline_state *pso = calloc(1, sizeof(*pso));
   if (!pso)
      return E_OUTOFMEMORY;

   pso->has_pixel_shader = ps_dxil && ps_size;
   pso->rasterization_enabled =
      !so_desc || !so_desc->NumEntries ||
      so_desc->RasterizedStream != D3D12_SO_NO_RASTERIZED_STREAM;
   pso->strip_cut = strip_cut;
   if (!pso->has_pixel_shader) {
      /* D3D12 treats declared RTV state as inactive when there is no pixel
       * shader. Compile the empty hardware fragment stage against a
       * zero-colour tilebuffer; depth/stencil remains fully active.
       */
      rt_format = PIPE_FORMAT_NONE;
      nr_rt = 0;
      rt_formats = NULL;
      nr_samples = 1;
   }

   if (nr_samples != 1 && nr_samples != 2 && nr_samples != 4) {
      free(pso);
      return E_INVALIDARG;
   }

   if (nr_rt > AGX_MAX_RENDER_TARGETS) {
      free(pso);
      return E_INVALIDARG;
   }

   /* Depth/stencil state, resolved to hardware enums once here rather than
    * per draw. DSVFormat may legally be UNKNOWN while depth/stencil state is
    * still used with a view bound at draw time. The draw path already makes
    * that state inert when no DSV is actually attached.
    */
   pso->ds.dsv_format = dsv_format;
   pso->ds.depth_func = AGX_ZS_FUNC_ALWAYS;
   pso->ds.front_stencil_read_mask = 0xFF;
   pso->ds.front_stencil_write_mask = 0xFF;
   pso->ds.back_stencil_read_mask = 0xFF;
   pso->ds.back_stencil_write_mask = 0xFF;

   /* Stencil-disabled defaults: pass everything, change nothing. Zeroes
    * would mean NEVER/KEEP-on-fail, which rejects fragments silently.
    */
   pso->ds.front = (struct d12_stencil_face){
      .compare = AGX_ZS_FUNC_ALWAYS,
      .pass = AGX_STENCIL_OP_KEEP,
      .depth_fail = AGX_STENCIL_OP_KEEP,
      .stencil_fail = AGX_STENCIL_OP_KEEP,
   };
   pso->ds.back = pso->ds.front;

   if (ds_desc) {
      pso->ds.depth_test = ds_desc->DepthEnable;
      pso->ds.depth_write =
         ds_desc->DepthEnable &&
         ds_desc->DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL;
      if (ds_desc->DepthEnable)
         pso->ds.depth_func = d12_translate_compare(ds_desc->DepthFunc);

      pso->ds.stencil_test =
         ds_desc->StencilEnable &&
         (front_stencil_enable || back_stencil_enable);
      if (ds_desc->StencilEnable) {
         if (front_stencil_enable) {
            pso->ds.front_stencil_read_mask = front_stencil_read_mask;
            pso->ds.front_stencil_write_mask = front_stencil_write_mask;
            d12_translate_stencil_face(&pso->ds.front,
                                       &ds_desc->FrontFace);
         } else {
            pso->ds.front_stencil_read_mask = 0;
            pso->ds.front_stencil_write_mask = 0;
         }
         if (back_stencil_enable) {
            pso->ds.back_stencil_read_mask = back_stencil_read_mask;
            pso->ds.back_stencil_write_mask = back_stencil_write_mask;
            d12_translate_stencil_face(&pso->ds.back, &ds_desc->BackFace);
         } else {
            pso->ds.back_stencil_read_mask = 0;
            pso->ds.back_stencil_write_mask = 0;
         }
      }
   }

   /* D3D12 and Honeykrisp both use a Y-negative viewport here. A
    * same-hardware command-stream diff therefore gives the direct mapping:
    * the default FrontCounterClockwise=FALSE PSO uses front_face_ccw=false.
    * Inverting it culls both triangles of the standard full-screen strip.
    */
   pso->raster.front_face_ccw = false;
   pso->raster.depth_clip = true;
   pso->raster.polygon_mode = AGX_POLYGON_MODE_FILL;
   pso->raster.dynamic_depth_bias =
      flags & D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS;
   pso->raster.dynamic_strip_cut =
      flags & D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT;
   if (rs_desc) {
      pso->raster.front_face_ccw = rs_desc->FrontCounterClockwise;
      pso->raster.cull_front = rs_desc->CullMode == D3D12_CULL_MODE_FRONT;
      pso->raster.cull_back = rs_desc->CullMode == D3D12_CULL_MODE_BACK;
      pso->raster.depth_clip = rs_desc->DepthClipEnable;
      pso->raster.depth_bias = rasterizer_depth_bias;
      pso->raster.depth_bias_clamp = rs_desc->DepthBiasClamp;
      pso->raster.slope_scaled_depth_bias =
         rs_desc->SlopeScaledDepthBias;
      /* The legacy state selects alpha-antialiased lines only when
       * multisample line rasterization is disabled. AGX has no native smooth
       * mode, so the draw path widens coverage and the fragment epilog
       * attenuates alpha before applying the application's blend state. */
      pso->raster.smooth_line =
         rs_desc->AntialiasedLineEnable && !rs_desc->MultisampleEnable;

      static_assert(D3D12_FILL_MODE_WIREFRAME ==
                       D12_CORE_FILL_MODE_WIREFRAME &&
                    D3D12_FILL_MODE_SOLID == D12_CORE_FILL_MODE_SOLID,
                    "D3D12 fill-mode values");
      static_assert(AGX_POLYGON_MODE_LINE == D12_CORE_POLYGON_MODE_LINE &&
                    AGX_POLYGON_MODE_FILL == D12_CORE_POLYGON_MODE_FILL,
                    "AGX polygon-mode values");
      uint32_t polygon_mode;
      if (!d12_raster_polygon_mode(rs_desc->FillMode, &polygon_mode)) {
         free(pso);
         return E_INVALIDARG;
      }
      pso->raster.polygon_mode = (enum agx_polygon_mode)polygon_mode;

      /* Target-independent rasterization. Reject a count the hardware cannot
       * rasterize instead of quietly rasterizing at a different one: a shader
       * reading SV_Coverage would then see a mask of the wrong width. */
      if (!d12_raster_forced_sample_count_supported(
             rs_desc->ForcedSampleCount)) {
         mesa_loge("ForcedSampleCount %u is not supported; AGX rasterizes at "
                   "1, 2 or 4 samples", rs_desc->ForcedSampleCount);
         free(pso);
         return E_INVALIDARG;
      }
      pso->raster.forced_sample_count = rs_desc->ForcedSampleCount;
   }

   /* The tilebuffer sample stride is part of the hardware ABI, so use the
    * PSO's declared count and require the bound RTV to agree at draw time.
    * Keep zero colour buffers for a depth-only pipeline.
    * Inventing RGBA8 here makes the fragment shader and render command carry
    * a colour tilebuffer that has no backing attachment; Honeykrisp keeps the
    * zero-colour layout and lets ZLS stand alone.
    */
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
   if (nr_rt)
      formats[0] = rt_format;
   for (unsigned i = 1; i < nr_rt; i++) {
      if (rt_formats && d12_format_is_renderable(rt_formats[i]))
         formats[i] = d12_format_to_pipe(rt_formats[i]);
   }
   enum pipe_format layout_formats[AGX_MAX_RENDER_TARGETS];
   memcpy(layout_formats, formats, sizeof(layout_formats));

   /*
    * D3D12 permits UNKNOWN formats for attachments unused by a PSO, then
    * permits a render pass to bind real RTVs in those slots.  Compacting the
    * UNKNOWN slots makes every PSO subset a different AGX tilebuffer ABI:
    * switching masks then requires a new hardware render command and large
    * mix-and-match command lists exhaust the segment array.
    *
    * There is one useful class where the ABI can be made slot-stable without
    * guessing a format: independent-blend PSOs whose declared RTVs all use
    * the same format.  Reserve all eight slots with that known format.  The
    * epilog still exports only the PSO-declared targets, and the command path
    * only enables this merge when every actually bound RTV has that same
    * format.  Mixed-format cases retain the established compact layout.
    */
   enum pipe_format uniform_rt_format = PIPE_FORMAT_NONE;
   bool uniform_rt_formats = blend_desc && blend_desc->IndependentBlendEnable &&
                             !(vs_is_mesh && device->dev.chip == AGX_CHIP_G17P);
   for (unsigned i = 0; i < nr_rt; i++) {
      if (formats[i] == PIPE_FORMAT_NONE)
         continue;

      if (uniform_rt_format == PIPE_FORMAT_NONE)
         uniform_rt_format = formats[i];
      else if (formats[i] != uniform_rt_format)
         uniform_rt_formats = false;
   }

   uniform_rt_formats &= uniform_rt_format != PIPE_FORMAT_NONE;
   if (uniform_rt_formats) {
      for (unsigned i = 0; i < ARRAY_SIZE(layout_formats); i++)
         layout_formats[i] = uniform_rt_format;
      pso->stable_unused_rt_layout = true;
   }

   struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(
      layout_formats,
      pso->stable_unused_rt_layout ? ARRAY_SIZE(layout_formats) : nr_rt,
      nr_samples, false);
   pso->tib = tib;

   const D3D12_RENDER_TARGET_BLEND_DESC default_rt_blend = {
      .BlendEnable = false,
      .LogicOpEnable = false,
      .SrcBlend = D3D12_BLEND_ONE,
      .DestBlend = D3D12_BLEND_ZERO,
      .BlendOp = D3D12_BLEND_OP_ADD,
      .SrcBlendAlpha = D3D12_BLEND_ONE,
      .DestBlendAlpha = D3D12_BLEND_ZERO,
      .BlendOpAlpha = D3D12_BLEND_OP_ADD,
      .LogicOp = D3D12_LOGIC_OP_NOOP,
      .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL,
   };
   const D3D12_RENDER_TARGET_BLEND_DESC *rt_blends[AGX_MAX_RENDER_TARGETS];
   struct agx_blend_rt_key blend_keys[AGX_MAX_RENDER_TARGETS] = {0};
   bool use_fs_epilog =
      (blend_desc && blend_desc->AlphaToCoverageEnable) ||
      pso->raster.smooth_line;
   enum pipe_logicop logicop = PIPE_LOGICOP_NOOP;

   for (unsigned i = 0; i < nr_rt; i++) {
      const D3D12_RENDER_TARGET_BLEND_DESC *rt_blend =
         blend_desc
            ? &blend_desc->RenderTarget[
                 blend_desc->IndependentBlendEnable ? i : 0]
            : &default_rt_blend;
      rt_blends[i] = rt_blend;

      if (rt_blend->BlendEnable) {
         const D3D12_BLEND factors[4] = {
            rt_blend->SrcBlend, rt_blend->DestBlend,
            rt_blend->SrcBlendAlpha, rt_blend->DestBlendAlpha,
         };
         for (unsigned f = 0; f < 4; f++) {
            if (factors[f] == D3D12_BLEND_BLEND_FACTOR ||
                factors[f] == D3D12_BLEND_INV_BLEND_FACTOR ||
                factors[f] == D3D12_BLEND_ALPHA_FACTOR ||
                factors[f] == D3D12_BLEND_INV_ALPHA_FACTOR)
               pso->blend_constants_used = true;
         }
      }

      if (rt_blend->BlendEnable && rt_blend->LogicOpEnable) {
         free(pso);
         return E_INVALIDARG;
      }

      /* AGX exposes one logic operation for the whole output merger. D3D12
       * only permits that shape when independent blending is disabled, in
       * which case RT0's description is replicated to every active target. */
      if (rt_blend->LogicOpEnable) {
         if ((blend_desc && blend_desc->IndependentBlendEnable) ||
             !d12_translate_logic_op(rt_blend->LogicOp, &logicop)) {
            free(pso);
            return E_INVALIDARG;
         }
      }

      enum pipe_blend_func rgb_func = PIPE_BLEND_ADD;
      enum pipe_blend_func alpha_func = PIPE_BLEND_ADD;
      enum pipe_blendfactor rgb_src = PIPE_BLENDFACTOR_ONE;
      enum pipe_blendfactor rgb_dst = PIPE_BLENDFACTOR_ZERO;
      enum pipe_blendfactor alpha_src = PIPE_BLENDFACTOR_ONE;
      enum pipe_blendfactor alpha_dst = PIPE_BLENDFACTOR_ZERO;

      if (rt_blend->BlendEnable &&
          (!d12_translate_blend_op(rt_blend->BlendOp, &rgb_func) ||
           !d12_translate_blend_op(rt_blend->BlendOpAlpha, &alpha_func) ||
           !d12_translate_blend_factor(rt_blend->SrcBlend, &rgb_src) ||
           !d12_translate_blend_factor(rt_blend->DestBlend, &rgb_dst) ||
           !d12_translate_blend_factor(rt_blend->SrcBlendAlpha, &alpha_src) ||
           !d12_translate_blend_factor(rt_blend->DestBlendAlpha,
                                       &alpha_dst))) {
         free(pso);
         return E_INVALIDARG;
      }

      blend_keys[i] = (struct agx_blend_rt_key){
         .colormask = rt_blend->RenderTargetWriteMask &
                      D3D12_COLOR_WRITE_ENABLE_ALL,
         .mode = agx_pack_blend_standard(
            rgb_func, rgb_src, rgb_dst, alpha_func, alpha_src, alpha_dst),
      };

      if (formats[i] != PIPE_FORMAT_NONE &&
          (rt_blend->BlendEnable || rt_blend->LogicOpEnable ||
           rt_blend->RenderTargetWriteMask !=
              D3D12_COLOR_WRITE_ENABLE_ALL))
         use_fs_epilog = true;
   }

   const D3D12_RENDER_TARGET_BLEND_DESC *rt_blend =
      nr_rt ? rt_blends[0] : &default_rt_blend;

   /* Every compiled part is released at `cleanup` on both paths. */
   struct agx_shader_part vs_part = {0}, fs_part = {0};
   struct agx_shader_part sw_vs_part = {0}, tcs_part = {0}, tes_part = {0};
   struct agx_shader_part gs_sw_part = {0}, gs_count_part = {0};
   struct agx_shader_part gs_pre_part = {0}, gs_main_part = {0};
   struct agx_shader_part gs_rast_part = {0};
   /* Prologs and epilogs are device-cached and shared, never freed here. */
   const struct agx_shader_part *fs_prolog = NULL, *fs_epilog = NULL;
   const struct agx_shader_part *prolog = NULL;
   /* The compiled vertex and fragment stages, shared through the device
    * cache; `vs`/`fs` alias their parts. Referenced until `fail`. */
   struct d12_cached_stage *vs_entry = NULL, *fs_entry = NULL;
   const struct agx_shader_part *vs = NULL, *fs = NULL;
   struct agx_fs_prolog_key fs_prolog_key = {0};
   struct agx_fs_epilog_key epilog_key = {0};
   HRESULT hr = S_OK;
   struct d12_vs_fetch fetch;
   resolve_input_layout(pso, input_layout, vs_dxil, vs_size, &fetch);

   /* Shader dumps are a per-compile side effect; keep them per PSO. */
   const bool bypass_cache =
      d12_debug()->dump_shaders || d12_debug()->dump_tex;

   struct d12_stage_cache_key vs_key;
   d12_stage_cache_key_init(&vs_key, MESA_SHADER_VERTEX, vs_dxil, vs_size,
                            map);
   vs_key.nr_velems = fetch.nr;
   memcpy(vs_key.velems, fetch.keys, sizeof(vs_key.velems));
   if (!bypass_cache)
      vs_entry = d12_shader_cache_find_stage(device, &vs_key, vs_dxil);
   if (!vs_entry) {
      struct d12_stage_outputs o = {0};
      if (!compile_gfx_stage(device, map, vs_dxil, vs_size, MESA_SHADER_VERTEX,
                             &tib, 0xffff, 0, &fetch, false, false, &o,
                             &vs_part)) {
         hr = E_INVALIDARG;
         goto fail;
      }
      vs_entry = d12_shader_cache_add_stage(device, &vs_key, vs_dxil,
                                            &vs_part, &o, !bypass_cache);
      if (!vs_entry) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
   }
   vs = &vs_entry->part;
   if (!d12_merge_texture_shapes(pso->texture_shapes,
                                 vs_entry->out.texture_shapes)) {
      mesa_loge("graphics PSO: vertex shader declares a conflicting texture "
                "shape for a root slot");
      hr = E_INVALIDARG;
      goto fail;
   }
   pso->uvs = vs_entry->out.uvs;
   pso->vs.g17p_root_only = vs_entry->out.root_only;
   pso->vs.g17p_sample.valid = vs_entry->out.g17p_fragment_sample;
   pso->vs.g17p_sample.indexed_fetch = vs_entry->out.g17p_fragment_fetch;
   pso->vs.g17p_sample.indexed_samplers = vs_entry->out.g17p_indexed_samplers;
   pso->vs.reads_root = vs_entry->out.reads_root;
   pso->vs.g17p_writes_memory = vs_entry->out.writes_memory;
   const uint8_t cull_distance_size = vs_entry->out.cull_distance_size;

   uint64_t fs_outputs = 0;
   const uint16_t framebuffer_mask =
      (uint16_t)BITFIELD_MASK(MIN2(nr_samples, 8));
   uint16_t active_sample_mask =
      (sample_mask | (uint16_t)~framebuffer_mask) & 0xff;
   uint8_t a8_mask = 0;
   for (unsigned i = 0; rt_formats && i < nr_rt; i++) {
      if (rt_formats[i] == DXGI_FORMAT_A8_UNORM)
         a8_mask |= BITFIELD_BIT(i);
   }

   const bool allow_flat_geometry =
      ps_dxil && ps_size && !(gs_dxil && gs_size) &&
      !(hs_dxil && hs_size) && !(ds_dxil && ds_size) &&
      !(so_desc && so_desc->NumEntries);

   /* The fragment compile also depends on the tilebuffer layout (a
    * function of the layout formats, their count and the sample count),
    * the API sample mask, the A8 targets, whether an epilog was requested
    * and whether flat transport may be routed through poly. */
   struct d12_stage_cache_key fs_key;
   d12_stage_cache_key_init(&fs_key, MESA_SHADER_FRAGMENT, ps_dxil, ps_size,
                            map);
   fs_key.api_sample_mask = active_sample_mask;
   fs_key.a8_mask = a8_mask;
   fs_key.nr_samples = nr_samples;
   fs_key.nr_layout_formats =
      pso->stable_unused_rt_layout ? ARRAY_SIZE(layout_formats) : nr_rt;
   for (unsigned i = 0; i < ARRAY_SIZE(layout_formats); i++)
      fs_key.layout_formats[i] = layout_formats[i];
   fs_key.flags = (use_fs_epilog ? D12_STAGE_KEY_FS_EPILOG_REQUESTED : 0) |
                  (allow_flat_geometry ? D12_STAGE_KEY_FLAT_TRANSPORT : 0) |
                  (vs_is_mesh ? D12_STAGE_KEY_LOSSLESS_FLAT : 0);
   if (!bypass_cache)
      fs_entry = d12_shader_cache_find_stage(device, &fs_key, ps_dxil);
   if (!fs_entry) {
      struct d12_stage_outputs o = {0};
      o.use_fs_epilog = use_fs_epilog;
      if (!compile_gfx_stage(device, map, ps_dxil, ps_size,
                             MESA_SHADER_FRAGMENT, &tib, active_sample_mask,
                             a8_mask, NULL, allow_flat_geometry, vs_is_mesh, &o,
                             &fs_part)) {
         hr = E_INVALIDARG;
         goto fail;
      }
      fs_entry = d12_shader_cache_add_stage(device, &fs_key, ps_dxil,
                                            &fs_part, &o, !bypass_cache);
      if (!fs_entry) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
   }
   fs = &fs_entry->part;
   if (!d12_merge_texture_shapes(pso->texture_shapes,
                                 fs_entry->out.texture_shapes)) {
      mesa_loge("graphics PSO: pixel shader declares a conflicting texture "
                "shape for a root slot");
      hr = E_INVALIDARG;
      goto fail;
   }
   fs_outputs = fs_entry->out.outputs_written;
   use_fs_epilog = fs_entry->out.use_fs_epilog;
   const struct agx_fs_epilog_link_info fs_epilog_link =
      fs_entry->out.fs_epilog_link;
   const struct d12_flat_transport flat_transport =
      fs_entry->out.flat_transport;
   pso->fs.g17p_root_only = fs_entry->out.root_only;
   pso->fs.g17p_sample.valid = fs_entry->out.g17p_fragment_sample;
   pso->fs.g17p_sample.indexed_fetch = fs_entry->out.g17p_fragment_fetch;
   pso->fs.g17p_sample.indexed_samplers = fs_entry->out.g17p_indexed_samplers;
   pso->fs.g17p_sample.sampler_slot = fs_entry->out.g17p_sample_slot;
   pso->fs.g17p_sample.texture_offset_B = fs_entry->out.g17p_texture_offset_B;
   pso->fs.reads_root = fs_entry->out.reads_root;
   pso->fs.g17p_writes_memory = fs_entry->out.writes_memory;

   for (unsigned i = 0; i < nr_rt; i++) {
      if (formats[i] != PIPE_FORMAT_NONE && blend_keys[i].colormask &&
          (fs_outputs & BITFIELD64_BIT(FRAG_RESULT_DATA0 + i)))
         pso->rt_write_mask |= BITFIELD_BIT(i);
   }

   /* Cull distance is an emulated pre-fragment operation on AGX. Keep it in
    * the standard fragment prolog, exactly like Honeykrisp, rather than in
    * the API shader main. Besides preserving the required ordering relative
    * to depth/stencil tests, the fast linker knows how to merge the prolog's
    * private coefficient-register bindings after those used by the main. */
   if (cull_distance_size) {
      fs_prolog_key = (struct agx_fs_prolog_key){
         .api_sample_mask = 0xff,
         .cull_distance_size = cull_distance_size,
         .run_zs_tests = true,
         .cf_base = fs->info.varyings.fs.nr_cf,
      };

      struct agx_shader_key prolog_backend_key = {
         .dev = agx_gather_device_key(&device->dev),
         .secondary = true,
         .no_stop = true,
         .fs.cf_base = fs_prolog_key.cf_base,
      };
      fs_prolog = d12_shader_cache_part(device, agx_nir_fs_prolog,
                                        &fs_prolog_key, sizeof(fs_prolog_key),
                                        &prolog_backend_key, false);

      if (!fs_prolog || !fs_prolog->binary || !fs_prolog->info.binary_size) {
         hr = E_INVALIDARG;
         goto fail;
      }
   }

   /* D3D12 does not define blending for pure-integer render targets. The
    * validation is conditional on the pixel shader actually exporting that
    * target: an enabled blend state is inert, and legal, when no colour is
    * written. Capture outputs_written before tilebuffer lowering consumes the
    * fragment stores, then reject only the active invalid combination.
    */
   for (unsigned i = 0; i < nr_rt; i++) {
      if (rt_blends[i]->BlendEnable &&
          util_format_is_pure_integer(formats[i]) &&
          (fs_outputs & BITFIELD64_BIT(FRAG_RESULT_DATA0 + i))) {
         hr = E_INVALIDARG;
         goto fail;
      }
   }

   /*
    * D3D12 dual-source blending consumes the second colour export as SRC1;
    * it is not a second render target. Consequently a real RT1 is illegal,
    * and independent blend state for later attachments must stay disabled.
    * Higher targets are only a conflict when the pixel shader writes them.
    *
    * Validate after translation so the shader's actual output mask, rather
    * than a bytecode-container guess, decides the RT2+ cases.
    */
   const bool dual_source =
      d12_blend_factor_uses_src1(rt_blend->SrcBlend) ||
      d12_blend_factor_uses_src1(rt_blend->DestBlend) ||
      d12_blend_factor_uses_src1(rt_blend->SrcBlendAlpha) ||
      d12_blend_factor_uses_src1(rt_blend->DestBlendAlpha);

   if (dual_source && blend_desc && blend_desc->IndependentBlendEnable) {
      for (unsigned i = 1; i < ARRAY_SIZE(blend_desc->RenderTarget); ++i) {
         if (blend_desc->RenderTarget[i].BlendEnable ||
             blend_desc->RenderTarget[i].LogicOpEnable) {
            hr = E_INVALIDARG;
            goto fail;
         }
      }
   }

   if (dual_source && rt_formats) {
      for (unsigned i = 1; i < MIN2(nr_rt, 8); ++i) {
         if (rt_formats[i] == DXGI_FORMAT_UNKNOWN)
            continue;

         if (i == 1 ||
             (fs_outputs & BITFIELD64_BIT(FRAG_RESULT_DATA0 + i))) {
            hr = E_INVALIDARG;
            goto fail;
         }
      }
   }

   if (use_fs_epilog) {
      /* An explicit SV_Coverage output is the final coverage mask. D3D12
       * does not additionally alpha-to-coverage that mask; doing both made
       * an explicit nonzero mask disappear whenever the shader's alpha was
       * zero. */
      epilog_key = (struct agx_fs_epilog_key){
         .link = fs_epilog_link,
         .smooth_line_alpha = pso->raster.smooth_line,
         .blend.alpha_to_coverage =
            blend_desc && blend_desc->AlphaToCoverageEnable &&
            !(fs_outputs & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)),
         .blend.logicop_enable = rt_blend->LogicOpEnable,
         .blend.logicop_func = logicop,
         .nr_samples = nr_samples,
      };

      memset(epilog_key.remap, -1, sizeof(epilog_key.remap));
      for (unsigned i = 0; i < ARRAY_SIZE(epilog_key.rt_formats); i++) {
         epilog_key.blend.rt[i] = blend_keys[i];
         epilog_key.rt_formats[i] = tib.logical_format[i];
         if (i < nr_rt && formats[i] != PIPE_FORMAT_NONE)
            epilog_key.remap[i] = i;
      }

      struct agx_shader_key epilog_backend_key = {
         .dev = agx_gather_device_key(&device->dev),
         .secondary = true,
         /* The one-sample fast-link footer appends the terminating stop. */
         .no_stop = true,
      };
      fs_epilog = d12_shader_cache_part(device, agx_nir_fs_epilog,
                                        &epilog_key, sizeof(epilog_key),
                                        &epilog_backend_key, true);

      bool any_colour_write = false;
      for (unsigned i = 0; i < nr_rt; i++)
         any_colour_write |= blend_keys[i].colormask != 0;

      const bool intentionally_empty =
         !any_colour_write && !epilog_key.blend.logicop_enable &&
         !epilog_key.blend.alpha_to_coverage && !fs_epilog_link.write_z &&
         !fs_epilog_link.write_s;

      if (!fs_epilog ||
          ((!fs_epilog->binary || !fs_epilog->info.binary_size) &&
           !intentionally_empty)) {
         hr = E_INVALIDARG;
         goto fail;
      }
   }

   /* The vertex shader needs a prolog: the AGX ABI has vertex/instance IDs
    * arriving in registers that the prolog fills (AGX_ABI_VIN_*). Without
    * one, SV_VertexID reads an uninitialized register and the geometry
    * lands nowhere. We fetch no attributes, so static_vi says "just fix up
    * the vertex ID".
    */
   struct agx_vs_prolog_key prolog_key = {
      .hw = true,
      .static_vi = true,
   };

   struct agx_shader_key prolog_backend_key = {
      .dev = agx_gather_device_key(&device->dev),
      /* Allocas live in per-thread scratch, and large shaders spill. */
      .has_scratch = true,
      .secondary = true,
      .no_stop = true,
   };

   prolog = d12_shader_cache_part(device, agx_nir_vs_prolog, &prolog_key,
                                  sizeof(prolog_key), &prolog_backend_key,
                                  false);
   /* Static VI needs no fixup instructions on G13/G14. An empty successful
    * compile is a valid no-op there; failed compiles and missing nonempty
    * binaries remain errors. Keep G17P's nonempty-prolog requirement. */
   const bool empty_prolog = prolog && !prolog->info.binary_size;
   if (!prolog || (!empty_prolog && !prolog->binary) ||
       (empty_prolog && device->dev.chip == AGX_CHIP_G17P && !vs_is_mesh)) {
      hr = E_INVALIDARG;
      goto fail;
   }

   /* Link: this uploads executable memory and bakes the USC/PPP words.
    * Linked variants are cached with the stage, keyed on the prolog. */
   struct d12_link_key vs_link = {0};
   vs_link.has_prolog = !empty_prolog;
   if (!empty_prolog)
      vs_link.vs_prolog = prolog_key;
   if (!d12_cached_stage_link(device, vs_entry, &vs_link,
                              empty_prolog ? NULL : prolog, NULL,
                              &pso->vs.linked)) {
      hr = E_OUTOFMEMORY;
      goto fail;
   }

   /*
    * THE PREAMBLE RULE, which the ordinary vertex shader had never needed
    * before: agx_fast_link links main + prolog + epilog and does NOT copy
    * the preamble, so info.preamble_offset is meaningless inside
    * linked.bo. Every other stage that can have one already keeps its own
    * copy (gs_rast, tes, ms_rast); a plain VS could not have one until it
    * started fetching vertex attributes, because nothing else made it
    * compute a uniform.
    *
    * Missing this does not fail loudly. The preamble is what turns the
    * attribute parameter block into the base/clamp/stride uniforms, so
    * without it the fetch runs against whatever those registers held --
    * measured as a triangle covering the ENTIRE render target.
    */
   if (vs->info.has_preamble) {
      pso->vs.code_bo =
         d12_cached_stage_code_bo(device, vs_entry, "d3d12 vs preamble");
      if (!pso->vs.code_bo) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
      pso->vs.preamble_va =
         pso->vs.code_bo->va->addr + vs->info.preamble_offset;
   }

   struct d12_link_key fs_link = {0};
   fs_link.fragment = true;
   fs_link.has_prolog = fs_prolog && fs_prolog->binary;
   fs_link.has_epilog = use_fs_epilog && fs_epilog && fs_epilog->binary;
   fs_link.nr_samples_shaded = use_fs_epilog ? 1 : 0;
   if (fs_link.has_prolog)
      fs_link.fs_prolog = fs_prolog_key;
   if (fs_link.has_epilog)
      fs_link.fs_epilog = epilog_key;
   if (!d12_cached_stage_link(device, fs_entry, &fs_link,
                              fs_link.has_prolog ? fs_prolog : NULL,
                              fs_link.has_epilog ? fs_epilog : NULL,
                              &pso->fs.linked)) {
      hr = E_OUTOFMEMORY;
      goto fail;
   }

   /* Texture promotion gives the fragment shader a preamble that copies
    * the descriptor reached through D12_UNIFORM_TEXTURES into hardware
    * texture state. agx_fast_link does not copy that preamble, so retain
    * the original compiled part exactly as the VS/TES/GS/MS paths do.
    * Pointing preamble_va into linked.bo executes unrelated linked main
    * code and is a GPU-fault-class error, not merely a bad sample. */
   if (fs->info.has_preamble) {
      pso->fs.code_bo =
         d12_cached_stage_code_bo(device, fs_entry, "d3d12 fs preamble");
      if (!pso->fs.code_bo) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
      pso->fs.preamble_va =
         pso->fs.code_bo->va->addr + fs->info.preamble_offset;
   }

   pso->vs.info = vs->info;
   pso->fs.info = fs->info;

   /*
    * GEOMETRY. Three more programs, compiled and linked here so a draw
    * only has to bind them. The rasterization shader replaces pso->vs as
    * the hardware vertex shader; the other two are compute, uploaded the
    * same way the compute PSO uploads its single program.
    *
    * A failure anywhere here fails PSO creation. Every unsupported
    * geometry configuration already refused BY NAME inside
    * compile_gs_stage, so reaching this point and then falling back to a
    * pipeline without a GS would be the silent-wrong this whole path
    * exists to avoid.
    */
   /*
    * TESSELLATION. The vertex shader becomes the same compute prepass the
    * geometry path uses, the hull shader is compute, and the DOMAIN
    * shader replaces pso->vs as the hardware vertex stage.
    */
   if (hs_dxil && hs_size && ds_dxil && ds_size) {

      if (!compile_sw_vs_stage(device, map, vs_dxil, vs_size, &fetch,
                               &sw_vs_part, &pso->sw_vs_outputs, NULL, NULL) ||
          !compile_tcs_stage(device, map, hs_dxil, hs_size, pso, &tcs_part) ||
          !compile_tes_stage(device, map, ds_dxil, ds_size, pso, &pso->tes_uvs,
                             &tes_part)) {
         hr = E_INVALIDARG;
         goto fail;
      }

      const struct {
         struct agx_shader_part *part;
         struct agx_bo **bo;
         uint64_t *va;
         struct agx_shader_info *info;
         struct d12_native_compute_abi *abi;
         const char *name;
      } cs[] = {
         {&sw_vs_part, &pso->sw_vs_bo, &pso->sw_vs_va, &pso->sw_vs_info, &pso->native_stages[D12_NATIVE_SW_VS],
          "d3d12 sw vs (tess)"},
         {&tcs_part, &pso->tcs_bo, &pso->tcs_va, &pso->tcs_info, &pso->native_stages[D12_NATIVE_TCS],
          "d3d12 hull"},
      };

      for (unsigned i = 0; i < ARRAY_SIZE(cs); i++) {
         HRESULT upload = d12_upload_internal_compute(
            device, cs[i].part, cs[i].bo, cs[i].va, cs[i].info,
            cs[i].abi, cs[i].name);
            if (FAILED(upload)) {
            hr = upload;
            goto fail;
         }
      }

      /* The domain shader is a hardware vertex stage and takes the same
       * static_vi prolog as pso->vs. */
      bool tes_linked = d12_link(&pso->tes.linked, &device->dev, false,
                                 &tes_part, empty_prolog ? NULL : prolog,
                                 NULL, 0);
      if (!tes_linked) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }
      pso->tes.info = tes_part.info;
      agx_assign_uvs(&pso->tes_varyings, &pso->tes_uvs, 0, 0);

      /* Same preamble rule as the geometry rasterization program: the
       * linker does not copy it, so keep the compiled part addressable.
       */
      if (tes_part.info.has_preamble) {
         pso->tes.code_bo =
            agx_bo_create(&device->dev, tes_part.info.binary_size, 0,
                          AGX_BO_EXEC | AGX_BO_LOW_VA, "d3d12 domain");
         if (!pso->tes.code_bo) {
            hr = E_OUTOFMEMORY;
            goto fail;
         }
         if (!d12_upload_executable(pso->tes.code_bo, tes_part.binary,
                                    tes_part.info.binary_size)) {
            hr = E_FAIL;
            goto fail;
         }
         pso->tes.preamble_va =
            pso->tes.code_bo->va->addr + tes_part.info.preamble_offset;
      }

      unsigned tes_samplers = tes_part.info.sampler_state_count;
      if (tes_part.info.uses_txf)
         tes_samplers = MAX2(tes_samplers, 1);

      agx_pack(&pso->tes.counts, COUNTS, cfg) {
         cfg.uniform_register_count = tes_part.info.push_count;
         cfg.preshader_register_count = tes_part.info.nr_preamble_gprs;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(tes_samplers, false);
         cfg.texture_state_register_count = tes_part.info.texture_state_count;
      }

      mesa_logi("compiled tessellation pipeline: sw_vs %u B, tcs %u B, "
                "tes %u B, patch_size %u, stride %u, mode %u, spacing %u, "
                "ccw %d, points %d",
                sw_vs_part.info.binary_size, tcs_part.info.binary_size,
                tes_part.info.binary_size, pso->tcs_output_patch_size,
                pso->tcs_output_stride, pso->tess_mode, pso->tess_spacing,
                pso->tess_ccw, pso->tess_points);

      pso->has_tess = true;
   }

   bool has_app_gs = gs_dxil && gs_size;
   bool has_so = so_desc && so_desc->NumEntries;
   bool has_flat_geometry = flat_transport.enabled;
   if (!vs_is_mesh && (has_app_gs || has_so || has_flat_geometry)) {
      uint8_t clip_size = 0, cull_size = 0;

      nir_xfb_info *xfb = NULL;
      if (has_so) {
         bool valid = false;
         const void *capture = has_app_gs ? gs_dxil
                               : pso->has_tess ? ds_dxil
                                               : vs_dxil;
         size_t capture_size = has_app_gs ? gs_size
                               : pso->has_tess ? ds_size
                                               : vs_size;
         xfb = d12_build_xfb_info(NULL, so_desc, capture, capture_size,
                                  &valid);
         if (!valid || !xfb) {
            mesa_loge("graphics PSO: invalid stream-output declaration or "
                      "unmatched output semantic");
            hr = E_INVALIDARG;
            goto fail;
         }
      }

      bool compiled;
      uint64_t gs_input_outputs;
      if (pso->has_tess) {
         compiled = compile_sw_tes_stage(
            device, map, ds_dxil, ds_size, &gs_sw_part,
            &pso->sw_tes_outputs, &clip_size, &cull_size, has_app_gs);
         gs_input_outputs = pso->sw_tes_outputs;
      } else {
         compiled = compile_sw_vs_stage(
            device, map, vs_dxil, vs_size, &fetch, &gs_sw_part,
            &pso->sw_vs_outputs, &clip_size, &cull_size);
         gs_input_outputs = pso->sw_vs_outputs;
      }

      if (compiled && has_app_gs) {
         compiled = compile_gs_stage(
            device, map, gs_dxil, gs_size, xfb, primitive_topology_type, pso,
            &pso->gs_info,
            &pso->gs_uvs, &gs_count_part, &gs_pre_part, &gs_main_part,
            &gs_rast_part);
      } else if (compiled) {
         enum mesa_prim prim =
            pso->has_tess
               ? (pso->tess_points
                     ? MESA_PRIM_POINTS
                     : pso->tess_mode == TESS_PRIMITIVE_ISOLINES
                          ? MESA_PRIM_LINES
                          : MESA_PRIM_TRIANGLES)
               : d12_so_primitive(primitive_topology_type);
         if (prim == MESA_PRIM_UNKNOWN) {
            /*
             * D3D12 CREATES this pipeline. A stream-output PSO that omits
             * the PRIMITIVE_TOPOLOGY subobject leaves the type UNDEFINED,
             * and the runtime accepts it -- the topology only has to be
             * valid by the time something is drawn with it. Refusing
             * creation here failed test_create_pipeline_state's third
             * stream, which is exactly that pipeline and expects S_OK.
             *
             * Assume triangles for the passthrough geometry shader, which
             * is the topology every such pipeline in the suite goes on to
             * draw, and say so rather than assuming silently.
             */
            mesa_logw("graphics PSO: stream output with an UNDEFINED "
                      "primitive topology type; assuming triangles");
            prim = MESA_PRIM_TRIANGLES;
         }
         {
            nir_shader *passthrough = d12_build_so_passthrough_gs(
               gs_input_outputs, clip_size, cull_size, prim, xfb,
               has_flat_geometry);
            compiled = compile_gs_nir(
               device, passthrough, &pso->gs_info, &pso->gs_uvs,
               has_flat_geometry ? &flat_transport : NULL, &gs_count_part,
               &gs_pre_part, &gs_main_part, &gs_rast_part);
         }
      }
      ralloc_free(xfb);

      if (!compiled) {
         hr = E_INVALIDARG;
         goto fail;
      }

      /* The software VS, optional count/pre-GS, and geometry body are
       * ordinary compute programs. Count/pre-GS are present when XFB needs
       * capacity accounting and filled-size updates. */
      struct agx_bo **sw_bo =
         pso->has_tess ? &pso->sw_tes_bo : &pso->sw_vs_bo;
      uint64_t *sw_va = pso->has_tess ? &pso->sw_tes_va : &pso->sw_vs_va;
      struct agx_shader_info *sw_info =
         pso->has_tess ? &pso->sw_tes_info : &pso->sw_vs_info;

      const struct {
         struct agx_shader_part *part;
         struct agx_bo **bo;
         uint64_t *va;
         struct agx_shader_info *info;
         struct d12_native_compute_abi *abi;
         const char *name;
      } cs[] = {
         {&gs_sw_part, sw_bo, sw_va, sw_info,
          &pso->native_stages[pso->has_tess ? D12_NATIVE_SW_TES : D12_NATIVE_SW_VS],
          pso->has_tess ? "d3d12 sw domain" : "d3d12 sw vs"},
         {&gs_count_part, &pso->gs_count_bo, &pso->gs_count_va,
          &pso->gs_count_info, &pso->native_stages[D12_NATIVE_GS_COUNT],
          "d3d12 gs count"},
         {&gs_pre_part, &pso->gs_pre_bo, &pso->gs_pre_va,
          &pso->gs_pre_info, &pso->native_stages[D12_NATIVE_GS_PRE], "d3d12 gs pre"},
         {&gs_main_part, &pso->gs_main_bo, &pso->gs_main_va,
          &pso->gs_main_info, &pso->native_stages[D12_NATIVE_GS_MAIN],
          "d3d12 gs main"},
      };

      for (unsigned i = 0; i < ARRAY_SIZE(cs); i++) {
         if (!cs[i].part->binary || !cs[i].part->info.binary_size)
            continue;

         /* Same rule as the compute PSO: executable, low-VA, so the USC
          * can address it 32-bit-relative to the shader base. */
         HRESULT upload = d12_upload_internal_compute(
            device, cs[i].part, cs[i].bo, cs[i].va, cs[i].info,
            cs[i].abi, cs[i].name);
            if (FAILED(upload)) {
            hr = upload;
            goto fail;
         }
      }

      /* The rasterization program is an ordinary hardware vertex shader
       * and needs the SAME prolog pso->vs gets. It reads no vertex
       * attributes, but it does read load_vertex_id and load_instance_id
       * to work out which emitted vertex it represents, and fixing those
       * up is exactly what the static_vi prolog does. Linking it with no
       * prolog leaves both IDs undefined, so every invocation reads the
       * wrong slot of the vertex buffer.
       */
      bool gs_rast_linked = d12_link(
         &pso->gs_rast.linked, &device->dev, false, &gs_rast_part,
         empty_prolog ? NULL : prolog, NULL, 0);
      if (!gs_rast_linked) {
         hr = E_OUTOFMEMORY;
         goto fail;
      }

      /* Keep the compiled part so its PREAMBLE has an address. The linker
       * copies main/prolog/epilog into linked.bo and leaves the preamble
       * behind, so info.preamble_offset does not apply there -- the USC
       * then points the preshader at whatever happens to sit at that
       * offset, which decoded as a run of `jmp_incomplete 0` and killed
       * the render command outright: the target was not even cleared.
       */
      if (gs_rast_part.info.has_preamble) {
         pso->gs_rast.code_bo =
            agx_bo_create(&device->dev, gs_rast_part.info.binary_size, 0,
                          AGX_BO_EXEC | AGX_BO_LOW_VA, "d3d12 gs rast");
         if (!pso->gs_rast.code_bo) {
            hr = E_OUTOFMEMORY;
            goto fail;
         }
         if (!d12_upload_executable(pso->gs_rast.code_bo,
                                    gs_rast_part.binary,
                                    gs_rast_part.info.binary_size)) {
            hr = E_FAIL;
            goto fail;
         }
         pso->gs_rast.preamble_va = pso->gs_rast.code_bo->va->addr +
                                    gs_rast_part.info.preamble_offset;
      }
      pso->gs_rast.info = gs_rast_part.info;

      /* The VDM reads these counts to know how much state to load. Leave
       * them zero and the rasterization program runs with no uniforms,
       * no textures and no preamble -- which is not a crash, just a
       * shader computing from whatever was already in the registers.
       */
      unsigned rast_samplers = gs_rast_part.info.sampler_state_count;
      if (gs_rast_part.info.uses_txf)
         rast_samplers = MAX2(rast_samplers, 1);

      agx_pack(&pso->gs_rast.counts, COUNTS, cfg) {
         cfg.uniform_register_count = gs_rast_part.info.push_count;
         cfg.preshader_register_count = gs_rast_part.info.nr_preamble_gprs;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(rast_samplers, false);
         cfg.texture_state_register_count =
            gs_rast_part.info.texture_state_count;
      }
      agx_assign_uvs(&pso->gs_varyings, &pso->gs_uvs, 0, 0);
      pso->flat_outputs =
         has_flat_geometry ? flat_transport.flat_outputs : 0;

      mesa_logi("compiled geometry pipeline: sw_vs %u B / %u GPRs, "
                "gs_main %u B, gs_rast %u B / %u GPRs, outputs 0x%llx",
                gs_sw_part.info.binary_size, gs_sw_part.info.nr_gprs,
                gs_main_part.info.binary_size, gs_rast_part.info.binary_size,
                gs_rast_part.info.nr_gprs,
                (unsigned long long)(pso->has_tess ? pso->sw_tes_outputs
                                                   : pso->sw_vs_outputs));

      pso->has_gs = true;
   }

   /*
    * A texel fetch consumes a sampler even though HLSL's Load names none:
    * the backend lowers txf to `smpi ... ss0` and the driver supplies that
    * state. The count here is what makes the hardware FETCH it -- leave it
    * at zero and sampler state 0 is never loaded, so the fetch reads
    * undefined state and returns mostly out-of-bounds. Honeykrisp does the
    * same MAX2 (hk_shader.c).
    */
   unsigned vs_samplers = vs->info.sampler_state_count;
   if (vs->info.uses_txf)
      vs_samplers = MAX2(vs_samplers, 1);

   unsigned fs_samplers = fs->info.sampler_state_count;
   if (fs->info.uses_txf)
      fs_samplers = MAX2(fs_samplers, 1);

   agx_pack(&pso->vs.counts, COUNTS, cfg) {
      cfg.uniform_register_count = vs->info.push_count;
      cfg.preshader_register_count = vs->info.nr_preamble_gprs;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(vs_samplers, false);
      cfg.texture_state_register_count = vs->info.texture_state_count;
   }

   agx_pack_fragment_face_2(&pso->fs.frag_face, 0, &pso->fs.info);

   agx_pack(&pso->fs.fs_counts, FRAGMENT_SHADER_WORD_0, cfg) {
      cfg.cf_binding_count = pso->fs.linked.cf.nr_bindings;
      cfg.uniform_register_count = fs->info.push_count;
      cfg.preshader_register_count = fs->info.nr_preamble_gprs;
      cfg.texture_state_register_count = fs->info.texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(fs_samplers, false);
   }

   agx_assign_uvs(&pso->varyings, &pso->uvs, 0, 0);
   if (device->dev.chip == AGX_CHIP_G17P && !pso->has_gs && !pso->has_tess &&
       !pso->has_mesh && (pso->vs.g17p_root_only ||
                         pso->vs.g17p_sample.indexed_fetch) &&
       !pso->vs.info.has_preamble) {
      const struct agx_g17p_vertex_compile_info compiler_info = {
         .hardware_vertex = pso->vs.info.stage == MESA_SHADER_VERTEX,
         .has_preamble = pso->vs.info.has_preamble,
         .nr_preamble_gprs = pso->vs.info.nr_preamble_gprs,
         .preamble_scratch_size = pso->vs.info.preamble_scratch_size,
         .uvs_written = pso->uvs.written,
         .position_off = pso->uvs.group_offs[UVS_POSITION],
         .varyings_off = pso->uvs.group_offs[UVS_VARYINGS],
         .psiz_off = pso->uvs.group_offs[UVS_PSIZ],
         .layer_viewport_off = pso->uvs.group_offs[UVS_LAYER_VIEWPORT],
         .clip_off = pso->uvs.group_offs[UVS_CLIP_DIST],
         .uvs_size = pso->uvs.size,
         .user_size = pso->uvs.user_size,
      };
      if (!d12_agx3_g17p_vertex_program_info(device->dev.chip, &compiler_info,
                                             &pso->vs.agx3_vdm_state_update))
         mesa_loge("native graphics final UVS state producer refused layout");
   }


   pso->iface.lpVtbl = (ID3D12PipelineStateVtbl *)d12_api_trace_vtbl(
      D12_API_TRACE_PIPELINE_STATE, &d12_pso_vtbl);
   pso->refcount = 1;
   pso->lifetime = 1;
   pso->device = device;
   pso->is_compute = false;
   d12_private_data_store_init(&pso->private_data, (IUnknown *)&pso->iface);
   d12_device_add_ref(device);

   mesa_logi("compiled graphics pipeline: VS %u B / %u GPRs, FS %u B / %u "
             "GPRs, uvs size %u, cf bindings %u",
             vs->info.binary_size, vs->info.nr_gprs,
             fs->info.binary_size, fs->info.nr_gprs, pso->uvs.size,
             pso->fs.linked.cf.nr_bindings);

   if (d12_debug()->dump_shaders) {
      fprintf(stderr, "; VS main +0x%x\n", vs->info.main_offset);
      d12_disassemble_shader(device->dev.chip == AGX_CHIP_G17P, (uint8_t *)vs->binary +
                          vs->info.main_offset,
                       vs->info.binary_size - vs->info.main_offset,
                       stderr);
      fprintf(stderr, "; FS main +0x%x\n", fs->info.main_offset);
      d12_disassemble_shader(device->dev.chip == AGX_CHIP_G17P, (uint8_t *)fs->binary +
                          fs->info.main_offset,
                       fs->info.binary_size - fs->info.main_offset,
                       stderr);
   }


   hr = S_OK;

fail:
   free(vs_part.binary);
   free(fs_part.binary);
   free(sw_vs_part.binary);
   free(tcs_part.binary);
   free(tes_part.binary);
   free(gs_sw_part.binary);
   free(gs_count_part.binary);
   free(gs_pre_part.binary);
   free(gs_main_part.binary);
   free(gs_rast_part.binary);
   if (vs_entry)
      d12_cached_stage_unref(device, vs_entry);
   if (fs_entry)
      d12_cached_stage_unref(device, fs_entry);

   if (FAILED(hr)) {
      d12_pso_release_bos(device, pso);
      free(pso);
      return hr;
   }

   hr = d12_pso_QueryInterface(&pso->iface, riid, out);
   d12_pso_Release(&pso->iface);
   return hr;
}
