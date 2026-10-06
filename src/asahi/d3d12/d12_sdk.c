/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Agility selection is advisory: every factory uses this driver. In particular,
 * neither the executable's D3D12SDKVersion/D3D12SDKPath nor SetSDKVersion may
 * redirect us to Microsoft's D3D12Core.dll. Interface layouts are specified by
 * Microsoft's MIT DirectX-Headers; vkd3d-proton is a behavior reference only.
 */

#include "d12_private.h"

struct d12_sdk {
   ID3D12SDKConfiguration1 iface;
   uint32_t refs;
};

struct d12_factory {
   ID3D12DeviceFactory iface;
   ID3D12DeviceConfiguration1 config;
   uint32_t refs;
   uint32_t flags;
};

/* DRED policy is accepted, but this driver does not produce DRED reports.
 * Do not advertise the data interfaces or enable a nonexistent debug layer. */
struct d12_dred_settings {
   ID3D12DeviceRemovedExtendedDataSettings iface;
   uint32_t refs;
};

static ULONG STDMETHODCALLTYPE
d12_dred_AddRef(ID3D12DeviceRemovedExtendedDataSettings *iface)
{
   return p_atomic_inc_return(&((struct d12_dred_settings *)iface)->refs);
}

static ULONG STDMETHODCALLTYPE
d12_dred_Release(ID3D12DeviceRemovedExtendedDataSettings *iface)
{
   uint32_t refs = p_atomic_dec_return(&((struct d12_dred_settings *)iface)->refs);
   if (!refs)
      free(iface);
   return refs;
}

static HRESULT STDMETHODCALLTYPE
d12_dred_QueryInterface(ID3D12DeviceRemovedExtendedDataSettings *iface,
                        REFIID iid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!d12_guid_equal(iid, &IID_IUnknown) &&
       !d12_guid_equal(iid, &IID_ID3D12DeviceRemovedExtendedDataSettings))
      return E_NOINTERFACE;
   d12_dred_AddRef(iface);
   *out = iface;
   return S_OK;
}

static void STDMETHODCALLTYPE
d12_dred_SetEnablement(ID3D12DeviceRemovedExtendedDataSettings *iface,
                       D3D12_DRED_ENABLEMENT enablement)
{
   /* Advisory until DRED collection is supported. */
}

static const ID3D12DeviceRemovedExtendedDataSettingsVtbl d12_dred_vtbl = {
   .QueryInterface = d12_dred_QueryInterface,
   .AddRef = d12_dred_AddRef,
   .Release = d12_dred_Release,
   .SetAutoBreadcrumbsEnablement = d12_dred_SetEnablement,
   .SetPageFaultEnablement = d12_dred_SetEnablement,
   .SetWatsonDumpEnablement = d12_dred_SetEnablement,
};

static ULONG STDMETHODCALLTYPE
d12_sdk_AddRef(ID3D12SDKConfiguration1 *iface)
{
   return p_atomic_inc_return(&((struct d12_sdk *)iface)->refs);
}

static ULONG STDMETHODCALLTYPE
d12_sdk_Release(ID3D12SDKConfiguration1 *iface)
{
   uint32_t refs = p_atomic_dec_return(&((struct d12_sdk *)iface)->refs);
   if (!refs)
      free(iface);
   return refs;
}

static HRESULT STDMETHODCALLTYPE
d12_sdk_QueryInterface(ID3D12SDKConfiguration1 *iface, REFIID iid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!d12_guid_equal(iid, &IID_IUnknown) &&
       !d12_guid_equal(iid, &IID_ID3D12SDKConfiguration) &&
       !d12_guid_equal(iid, &IID_ID3D12SDKConfiguration1))
      return E_NOINTERFACE;
   d12_sdk_AddRef(iface);
   *out = iface;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_sdk_SetSDKVersion(ID3D12SDKConfiguration1 *iface, UINT version,
                      LPCSTR path)
{
   /* Accept the request without storing or opening the SDK path. */
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_sdk_CreateDeviceFactory(ID3D12SDKConfiguration1 *iface, UINT version,
                            LPCSTR path, REFIID iid, void **out)
{
   return D3D12GetInterface(&CLSID_D3D12DeviceFactory, iid, out);
}

static void STDMETHODCALLTYPE
d12_sdk_FreeUnusedSDKs(ID3D12SDKConfiguration1 *iface)
{
   /* No external SDK modules are loaded. */
}

static const ID3D12SDKConfiguration1Vtbl d12_sdk_vtbl = {
   .QueryInterface = d12_sdk_QueryInterface,
   .AddRef = d12_sdk_AddRef,
   .Release = d12_sdk_Release,
   .SetSDKVersion = d12_sdk_SetSDKVersion,
   .CreateDeviceFactory = d12_sdk_CreateDeviceFactory,
   .FreeUnusedSDKs = d12_sdk_FreeUnusedSDKs,
};

static ULONG STDMETHODCALLTYPE
d12_factory_AddRef(ID3D12DeviceFactory *iface)
{
   return p_atomic_inc_return(&((struct d12_factory *)iface)->refs);
}

static ULONG STDMETHODCALLTYPE
d12_factory_Release(ID3D12DeviceFactory *iface)
{
   uint32_t refs = p_atomic_dec_return(&((struct d12_factory *)iface)->refs);
   if (!refs)
      free(iface);
   return refs;
}

static HRESULT STDMETHODCALLTYPE
d12_factory_QueryInterface(ID3D12DeviceFactory *iface, REFIID iid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   struct d12_factory *factory = (struct d12_factory *)iface;
   if (d12_guid_equal(iid, &IID_IUnknown) ||
       d12_guid_equal(iid, &IID_ID3D12DeviceFactory))
      *out = iface;
   else if (d12_guid_equal(iid, &IID_ID3D12DeviceConfiguration) ||
            d12_guid_equal(iid, &IID_ID3D12DeviceConfiguration1))
      *out = &factory->config;
   else
      return E_NOINTERFACE;
   d12_factory_AddRef(iface);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_factory_InitializeFromGlobalState(ID3D12DeviceFactory *iface)
{
   /* Factory flags are local; global D3D12CreateDevice remains a singleton. */
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_factory_ApplyToGlobalState(ID3D12DeviceFactory *iface)
{
   /* Global and per-factory creation already use the same driver. */
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_factory_SetFlags(ID3D12DeviceFactory *iface, D3D12_DEVICE_FACTORY_FLAGS flags)
{
   p_atomic_set(&((struct d12_factory *)iface)->flags, flags);
   return S_OK;
}

static D3D12_DEVICE_FACTORY_FLAGS STDMETHODCALLTYPE
d12_factory_GetFlags(ID3D12DeviceFactory *iface)
{
   return p_atomic_read(&((struct d12_factory *)iface)->flags);
}

static HRESULT STDMETHODCALLTYPE
d12_factory_GetConfigurationInterface(ID3D12DeviceFactory *iface,
                                      REFCLSID clsid, REFIID iid, void **out)
{
   return D3D12GetInterface(clsid, iid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_factory_EnableExperimentalFeatures(ID3D12DeviceFactory *iface, UINT count,
                                       const IID *iids, void *configs,
                                       UINT *sizes)
{
   return D3D12EnableExperimentalFeatures(count, iids, configs, sizes);
}

static HRESULT STDMETHODCALLTYPE
d12_factory_CreateDevice(ID3D12DeviceFactory *iface, IUnknown *adapter,
                         D3D_FEATURE_LEVEL level, REFIID iid, void **out)
{
   /* Independent devices are supported, so the ALLOW_RETURNING_* flags
    * never require falling back to the global singleton. */
   return d12_create_device(adapter, level, iid, out, false);
}

static const ID3D12DeviceFactoryVtbl d12_factory_vtbl = {
   .QueryInterface = d12_factory_QueryInterface,
   .AddRef = d12_factory_AddRef,
   .Release = d12_factory_Release,
   .InitializeFromGlobalState = d12_factory_InitializeFromGlobalState,
   .ApplyToGlobalState = d12_factory_ApplyToGlobalState,
   .SetFlags = d12_factory_SetFlags,
   .GetFlags = d12_factory_GetFlags,
   .GetConfigurationInterface = d12_factory_GetConfigurationInterface,
   .EnableExperimentalFeatures = d12_factory_EnableExperimentalFeatures,
   .CreateDevice = d12_factory_CreateDevice,
};

static struct d12_factory *
d12_factory_from_config(ID3D12DeviceConfiguration1 *iface)
{
   return container_of(iface, struct d12_factory, config);
}

static HRESULT STDMETHODCALLTYPE
d12_factory_config_QueryInterface(ID3D12DeviceConfiguration1 *iface,
                                  REFIID iid, void **out)
{
   return d12_factory_QueryInterface(&d12_factory_from_config(iface)->iface,
                                     iid, out);
}

static ULONG STDMETHODCALLTYPE
d12_factory_config_AddRef(ID3D12DeviceConfiguration1 *iface)
{
   return d12_factory_AddRef(&d12_factory_from_config(iface)->iface);
}

static ULONG STDMETHODCALLTYPE
d12_factory_config_Release(ID3D12DeviceConfiguration1 *iface)
{
   return d12_factory_Release(&d12_factory_from_config(iface)->iface);
}

static const ID3D12DeviceConfiguration1Vtbl d12_factory_config_vtbl = {
   .QueryInterface = d12_factory_config_QueryInterface,
   .AddRef = d12_factory_config_AddRef,
   .Release = d12_factory_config_Release,
   .GetDesc = d12_config_GetDesc,
   .GetEnabledExperimentalFeatures = d12_config_GetEnabledExperimentalFeatures,
   .SerializeVersionedRootSignature = d12_config_SerializeVersionedRootSignature,
   .CreateVersionedRootSignatureDeserializer =
      d12_config_CreateVersionedRootSignatureDeserializer,
   .CreateVersionedRootSignatureDeserializerFromSubobjectInLibrary =
      d12_config_CreateVersionedRootSignatureDeserializerFromSubobjectInLibrary,
};

PUBLIC HRESULT WINAPI
D3D12GetInterface(REFCLSID clsid, REFIID iid, void **out)
{
   if (out)
      *out = NULL;
   if (!clsid || !iid)
      return E_INVALIDARG;

   if (d12_guid_equal(clsid, &CLSID_D3D12DeviceRemovedExtendedData)) {
      /* Configuration queries with no output are validation-only, including
       * unsupported data IIDs, matching the runtime's S_FALSE convention. */
      if (!out)
         return S_FALSE;
      struct d12_dred_settings *settings = calloc(1, sizeof(*settings));
      if (!settings)
         return E_OUTOFMEMORY;
      settings->iface.lpVtbl =
         (ID3D12DeviceRemovedExtendedDataSettingsVtbl *)&d12_dred_vtbl;
      settings->refs = 1;
      HRESULT hr = d12_dred_QueryInterface(&settings->iface, iid, out);
      d12_dred_Release(&settings->iface);
      return hr;
   }
   if (!out)
      return E_POINTER;

   if (d12_guid_equal(clsid, &CLSID_D3D12SDKConfiguration)) {
      struct d12_sdk *sdk = calloc(1, sizeof(*sdk));
      if (!sdk)
         return E_OUTOFMEMORY;
      sdk->iface.lpVtbl = (ID3D12SDKConfiguration1Vtbl *)&d12_sdk_vtbl;
      sdk->refs = 1;
      HRESULT hr = d12_sdk_QueryInterface(&sdk->iface, iid, out);
      d12_sdk_Release(&sdk->iface);
      return hr;
   }
   if (d12_guid_equal(clsid, &CLSID_D3D12DeviceFactory)) {
      struct d12_factory *factory = calloc(1, sizeof(*factory));
      if (!factory)
         return E_OUTOFMEMORY;
      factory->iface.lpVtbl = (ID3D12DeviceFactoryVtbl *)&d12_factory_vtbl;
      factory->config.lpVtbl =
         (ID3D12DeviceConfiguration1Vtbl *)&d12_factory_config_vtbl;
      factory->refs = 1;
      HRESULT hr = d12_factory_QueryInterface(&factory->iface, iid, out);
      d12_factory_Release(&factory->iface);
      return hr;
   }
   return E_NOINTERFACE;
}
