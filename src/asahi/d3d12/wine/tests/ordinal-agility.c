/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Link with d3d12-ordinal-import.def's import library, not mingw's d3d12.
 * This is a real x64 PE ordinal consumer, with Deadzone's Agility exports.
 */
#define COBJMACROS
#include <windows.h>
#include <directx/d3d12.h>
#include <stdio.h>

__declspec(dllexport) const UINT D3D12SDKVersion = 614;
__declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\x64\\";

static int failures;
#define CHECK(test) do { \
   if (!(test)) { printf("FAIL: %s (line %d)\n", #test, __LINE__); failures++; } \
   else printf("PASS: %s\n", #test); \
} while (0)

static void check_device(HRESULT hr, ID3D12Device *device)
{
   printf("CreateDevice HRESULT=0x%08lx device=%p\n", (unsigned long)hr, device);
   CHECK(hr == S_OK && device);
   if (device) {
      CHECK(ID3D12Device_GetNodeCount(device) == 1);
      ID3D12Device_Release(device);
   }
}

static void check_factory(ID3D12DeviceFactory *factory)
{
   ID3D12DeviceConfiguration1 *config1 = NULL, *config2 = NULL;
   CHECK(ID3D12DeviceFactory_QueryInterface(factory,
      &IID_ID3D12DeviceConfiguration1, (void **)&config1) == S_OK);
   CHECK(ID3D12DeviceFactory_QueryInterface(factory,
      &IID_ID3D12DeviceConfiguration1, (void **)&config2) == S_OK);
   CHECK(config1 && config1 == config2);
   if (config1) {
      IUnknown *identity = NULL;
      CHECK(ID3D12DeviceConfiguration1_QueryInterface(config1, &IID_IUnknown,
         (void **)&identity) == S_OK);
      CHECK(identity == (IUnknown *)factory);
      if (identity)
         CHECK(IUnknown_Release(identity) == 3);
      D3D12_DEVICE_CONFIGURATION_DESC desc;
      ID3D12DeviceConfiguration1_GetDesc(config1, &desc);
      CHECK(desc.SDKVersion == D3D12_SDK_VERSION);
      CHECK(desc.Flags == D3D12_DEVICE_FLAG_NONE);
      CHECK(desc.NumEnabledExperimentalFeatures == 0);
      CHECK(ID3D12DeviceConfiguration1_GetEnabledExperimentalFeatures(
         config1, NULL, 0) == S_OK);
      D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs = {0};
      rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
      ID3DBlob *blob = NULL;
      CHECK(ID3D12DeviceConfiguration1_SerializeVersionedRootSignature(
         config1, &rs, &blob, NULL) == S_OK);
      if (blob)
         blob->lpVtbl->Release(blob);
   }
   if (config2)
      CHECK(ID3D12DeviceConfiguration1_Release(config2) == 2);
   if (config1)
      CHECK(ID3D12DeviceConfiguration1_Release(config1) == 1);

   ID3D12DeviceRemovedExtendedDataSettings *dred = NULL;
   CHECK(ID3D12DeviceFactory_GetConfigurationInterface(factory,
      &CLSID_D3D12DeviceRemovedExtendedData,
      &IID_ID3D12DeviceRemovedExtendedDataSettings, (void **)&dred) == S_OK);
   if (dred) {
      ID3D12DeviceRemovedExtendedDataSettings_SetAutoBreadcrumbsEnablement(
         dred, D3D12_DRED_ENABLEMENT_FORCED_ON);
      CHECK(ID3D12DeviceRemovedExtendedDataSettings_Release(dred) == 0);
   }
   CHECK(ID3D12DeviceFactory_GetConfigurationInterface(factory,
      &CLSID_D3D12DeviceRemovedExtendedData,
      &IID_ID3D12DeviceRemovedExtendedData, NULL) == S_FALSE);

   CHECK(ID3D12DeviceFactory_SetFlags(factory,
      D3D12_DEVICE_FACTORY_FLAG_DISALLOW_STORING_NEW_DEVICE_AS_SINGLETON) == S_OK);
   CHECK(ID3D12DeviceFactory_GetFlags(factory) ==
      D3D12_DEVICE_FACTORY_FLAG_DISALLOW_STORING_NEW_DEVICE_AS_SINGLETON);
   CHECK(ID3D12DeviceFactory_ApplyToGlobalState(factory) == S_OK);
   ID3D12Device *a = NULL, *b = NULL, *c = NULL, *d = NULL;
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
      &IID_ID3D12Device, (void **)&a) == S_OK);
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
      &IID_ID3D12Device, (void **)&b) == S_OK);
   CHECK(a && a == b);
   CHECK(ID3D12DeviceFactory_CreateDevice(factory, NULL, D3D_FEATURE_LEVEL_11_0,
      &IID_ID3D12Device, (void **)&c) == S_OK);
   CHECK(ID3D12DeviceFactory_CreateDevice(factory, NULL, D3D_FEATURE_LEVEL_11_0,
      &IID_ID3D12Device, (void **)&d) == S_OK);
   CHECK(c && d && c != d && c != a && d != a);
   if (c)
      CHECK(ID3D12Device_Release(c) == 0);
   if (d)
      CHECK(ID3D12Device_Release(d) == 0);
   if (b)
      CHECK(ID3D12Device_Release(b) == 1);
   if (a)
      CHECK(ID3D12Device_Release(a) == 0);
}

int main(void)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   const char *names[] = {
      "D3D12CreateDevice", "D3D12GetDebugInterface",
      "D3D12CreateRootSignatureDeserializer",
      "D3D12CreateVersionedRootSignatureDeserializer",
      "D3D12EnableExperimentalFeatures", "D3D12GetInterface",
      "D3D12SerializeRootSignature", "D3D12SerializeVersionedRootSignature",
   };
   HMODULE dll = GetModuleHandleA("d3d12.dll");
   char path[MAX_PATH];
   CHECK(dll && GetModuleFileNameA(dll, path, sizeof(path)));
   if (!dll)
      return 1;
   printf("d3d12=%s SDKVersion=%u SDKPath=%s\n", path, D3D12SDKVersion, D3D12SDKPath);
   for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
      FARPROC named = GetProcAddress(dll, names[i]);
      CHECK(named && named == GetProcAddress(dll, (LPCSTR)(ULONG_PTR)(101 + i)));
   }

   ID3D12Device *device = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                 &IID_ID3D12Device, (void **)&device);
   check_device(hr, device);
   /* Invoke ordinal 102 too; a missing debug layer is a supported outcome. */
   IUnknown *debug = NULL;
   hr = D3D12GetDebugInterface(&IID_IUnknown, (void **)&debug);
   printf("ordinal 102 HRESULT=0x%08lx\n", (unsigned long)hr);
   if (debug)
      IUnknown_Release(debug);

   D3D12_ROOT_SIGNATURE_DESC desc = {0};
   ID3DBlob *blob = NULL;
   CHECK(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                    &blob, NULL) == S_OK);
   if (blob)
      blob->lpVtbl->Release(blob);
   D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned = {0};
   versioned.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
   blob = NULL;
   CHECK(D3D12SerializeVersionedRootSignature(&versioned, &blob, NULL) == S_OK);
   if (blob)
      blob->lpVtbl->Release(blob);

   ID3D12SDKConfiguration1 *sdk = NULL;
   CHECK(D3D12GetInterface(&CLSID_D3D12SDKConfiguration,
                           &IID_ID3D12SDKConfiguration1, (void **)&sdk) == S_OK);
   if (sdk) {
      CHECK(ID3D12SDKConfiguration1_SetSDKVersion(sdk, D3D12SDKVersion,
                                                 D3D12SDKPath) == S_OK);
      /* An unavailable SDK is deliberately ignored, too. */
      CHECK(ID3D12SDKConfiguration1_SetSDKVersion(sdk, 0xffffffff,
                                                 "Z:\\missing-sdk\\") == S_OK);
      ID3D12DeviceFactory *factory = NULL;
      CHECK(ID3D12SDKConfiguration1_CreateDeviceFactory(
         sdk, 614, D3D12SDKPath, &IID_ID3D12DeviceFactory,
         (void **)&factory) == S_OK);
      if (factory) {
         CHECK(ID3D12DeviceFactory_InitializeFromGlobalState(factory) == S_OK);
         CHECK(ID3D12DeviceFactory_ApplyToGlobalState(factory) == S_OK);
         device = NULL;
         hr = ID3D12DeviceFactory_CreateDevice(factory, NULL,
            D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&device);
         check_device(hr, device);
         check_factory(factory);
         CHECK(ID3D12DeviceFactory_Release(factory) == 0);
      }
      ID3D12SDKConfiguration1_FreeUnusedSDKs(sdk);
      CHECK(ID3D12SDKConfiguration1_Release(sdk) == 0);
   }
   ID3D12DeviceFactory *factory = NULL;
   CHECK(D3D12GetInterface(&CLSID_D3D12DeviceFactory,
                           &IID_ID3D12DeviceFactory, (void **)&factory) == S_OK);
   if (factory) {
      device = NULL;
      hr = ID3D12DeviceFactory_CreateDevice(factory, NULL,
         D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&device);
      check_device(hr, device);
      void *out = (void *)1;
      CHECK(ID3D12DeviceFactory_QueryInterface(factory, &IID_ID3D12Device,
                                              &out) == E_NOINTERFACE && !out);
      CHECK(ID3D12DeviceFactory_Release(factory) == 0);
   }
   void *out = (void *)1;
   CHECK(D3D12GetInterface(&CLSID_D3D12SDKConfiguration, &IID_ID3D12Device,
                           &out) == E_NOINTERFACE && !out);
   CHECK(D3D12GetInterface(&CLSID_D3D12SDKConfiguration,
                           &IID_ID3D12SDKConfiguration1, NULL) == E_POINTER);
   CHECK(!GetModuleHandleA("D3D12Core.dll"));
   printf("ORDINAL_AGILITY_RESULT failures=%d\n", failures);
   return failures ? 1 : 0;
}
