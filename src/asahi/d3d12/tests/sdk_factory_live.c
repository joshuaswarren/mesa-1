/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Device identity and lifetime checks. No command queues or GPU work.
 */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(test) do { \
   if (!(test)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #test); abort(); } \
} while (0)

static ID3D12Device *
create_device(void)
{
   ID3D12Device *device = NULL;
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12Device, (void **)&device) == S_OK);
   CHECK(device);
   return device;
}

static void *
create_thread(void *arg)
{
   return create_device();
}

static void *
release_thread(void *arg)
{
   CHECK(ID3D12Device_Release((ID3D12Device *)arg) == 0);
   return NULL;
}

int main(void)
{
   pthread_t threads[4];
   ID3D12Device *devices[4];
   /* All concurrent first callers must converge on the same adapter object. */
   for (unsigned i = 0; i < 4; ++i)
      CHECK(!pthread_create(&threads[i], NULL, create_thread, NULL));
   for (unsigned i = 0; i < 4; ++i)
      CHECK(!pthread_join(threads[i], (void **)&devices[i]));
   for (unsigned i = 1; i < 4; ++i)
      CHECK(devices[i] == devices[0]);
   for (unsigned i = 0; i < 4; ++i)
      CHECK(ID3D12Device_Release(devices[i]) == 3 - i);

   /* Repeatedly race the last reference against new adapter lookup. Either
    * retaining the old device or constructing a new one is legal. */
   for (unsigned i = 0; i < 16; ++i) {
      ID3D12Device *device = create_device();
      CHECK(!pthread_create(&threads[0], NULL, create_thread, NULL));
      ID3D12Device_Release(device);
      CHECK(!pthread_join(threads[0], (void **)&devices[0]));
      CHECK(ID3D12Device_GetNodeCount(devices[0]) == 1);
      CHECK(!pthread_create(&threads[0], NULL, release_thread, devices[0]));
      CHECK(!pthread_join(threads[0], NULL));
   }

   ID3D12DeviceFactory *factory = NULL;
   CHECK(D3D12GetInterface(&CLSID_D3D12DeviceFactory, &IID_ID3D12DeviceFactory,
                          (void **)&factory) == S_OK);
   ID3D12Device *global = create_device();
   ID3D12Device *independent = NULL;
   CHECK(ID3D12DeviceFactory_CreateDevice(factory, NULL, D3D_FEATURE_LEVEL_11_0,
      &IID_ID3D12Device, (void **)&independent) == S_OK);
   CHECK(independent && independent != global);

   ID3D12DeviceConfiguration1 *config = NULL;
   CHECK(ID3D12DeviceFactory_QueryInterface(factory,
      &IID_ID3D12DeviceConfiguration1, (void **)&config) == S_OK);
   IUnknown *identity = NULL;
   CHECK(ID3D12DeviceConfiguration1_QueryInterface(config, &IID_IUnknown,
      (void **)&identity) == S_OK);
   CHECK(identity == (IUnknown *)factory);
   CHECK(IUnknown_Release(identity) == 2);
   CHECK(ID3D12DeviceFactory_Release(factory) == 1);
   CHECK(ID3D12DeviceConfiguration1_GetDesc(config).SDKVersion == D3D12_SDK_VERSION);
   CHECK(ID3D12DeviceConfiguration1_Release(config) == 0);

   /* Unsupported IID and feature-level checks must not poison the singleton. */
   void *out = (void *)1;
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12DeviceFactory, &out) == E_NOINTERFACE);
   CHECK(!out);
   CHECK(D3D12CreateDevice(NULL, (D3D_FEATURE_LEVEL)0x7fffffff,
                          &IID_ID3D12Device, &out) == E_INVALIDARG);
   CHECK(!out);
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12Device, NULL) == S_FALSE);
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, NULL, NULL) == S_FALSE);

   /* RemoveDevice is a CPU-side state change. A removed global device cannot
    * be returned again, even for a validation-only request. */
   ID3D12Device5 *device5 = NULL;
   CHECK(ID3D12Device_QueryInterface(global, &IID_ID3D12Device5,
                                    (void **)&device5) == S_OK);
   ID3D12Device5_RemoveDevice(device5);
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12Device, &out) == DXGI_ERROR_DEVICE_REMOVED);
   CHECK(!out);
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12Device, NULL) == DXGI_ERROR_DEVICE_REMOVED);
   CHECK(ID3D12Device_GetDeviceRemovedReason(independent) == S_OK);
   CHECK(ID3D12Device5_Release(device5) == 1);
   CHECK(ID3D12Device_Release(global) == 0);
   CHECK(ID3D12Device_Release(independent) == 0);
   global = create_device();
   CHECK(ID3D12Device_GetDeviceRemovedReason(global) == S_OK);
   CHECK(ID3D12Device_Release(global) == 0);
   puts("SDK_FACTORY_LIVE_RESULT failures=0");
   return 0;
}
