/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Wine platform backend: the Linux DRM driver body running inside an ARM64EC
 * PE, with the render node reached through the unixlib (d12_wine_drm.c).
 * Mirrors d12_platform_linux.c; Win32 event signalling as on Windows.
 */

#include "d12_private.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <windows.h>
#include <dxgi.h>

#include "agx_wine_unixlib.h"

/* The IDXGIAdapter the application created the device from, kept for
 * GetAdapterLuid and for the DXVK presenter's GetAdapter. One entry per live
 * device; devices are rare so a small table is enough. */
struct d12_wine_device_entry {
   struct d12_device *device;
   IDXGIAdapter *adapter;
};

static simple_mtx_t d12_wine_devices_lock = SIMPLE_MTX_INITIALIZER;
static struct d12_wine_device_entry d12_wine_devices[16];

IDXGIAdapter *d12_wine_device_adapter(struct d12_device *device);

IDXGIAdapter *
d12_wine_device_adapter(struct d12_device *device)
{
   IDXGIAdapter *adapter = NULL;
   simple_mtx_lock(&d12_wine_devices_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(d12_wine_devices); i++) {
      if (d12_wine_devices[i].device == device) {
         adapter = d12_wine_devices[i].adapter;
         if (adapter)
            IDXGIAdapter_AddRef(adapter);
         break;
      }
   }
   simple_mtx_unlock(&d12_wine_devices_lock);
   return adapter;
}

static void
d12_wine_device_register(struct d12_device *device, IUnknown *adapter_unk)
{
   IDXGIAdapter *adapter = NULL;
   if (adapter_unk)
      IUnknown_QueryInterface(adapter_unk, &IID_IDXGIAdapter, (void **)&adapter);

   simple_mtx_lock(&d12_wine_devices_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(d12_wine_devices); i++) {
      if (!d12_wine_devices[i].device) {
         d12_wine_devices[i].device = device;
         d12_wine_devices[i].adapter = adapter;
         adapter = NULL;
         break;
      }
   }
   simple_mtx_unlock(&d12_wine_devices_lock);
   if (adapter)
      IDXGIAdapter_Release(adapter);
}

static void
d12_wine_device_unregister(struct d12_device *device)
{
   IDXGIAdapter *adapter = NULL;
   simple_mtx_lock(&d12_wine_devices_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(d12_wine_devices); i++) {
      if (d12_wine_devices[i].device == device) {
         adapter = d12_wine_devices[i].adapter;
         d12_wine_devices[i].device = NULL;
         d12_wine_devices[i].adapter = NULL;
         break;
      }
   }
   simple_mtx_unlock(&d12_wine_devices_lock);
   if (adapter)
      IDXGIAdapter_Release(adapter);
}

bool
d12_platform_is_real_asahi_render_fd(int fd)
{
   return d12_wine_fd_is_asahi_render_node(fd);
}

bool
d12_platform_open_device(struct d12_device *device, IUnknown *adapter)
{
   int fd = d12_wine_open_render_node();
   if (fd < 0) {
      mesa_loge("no asahi render node reachable through the Wine unixlib: %s",
                strerror(errno));
      return false;
   }

   device->dev.fd = fd;
   if (!agx_open_device(NULL, &device->dev)) {
      d12_wine_close_fd(fd);
      device->dev.fd = -1;
      return false;
   }

   device->sync = util_sync_provider_drm(fd);
   if (!device->sync) {
      d12_platform_close_device(device);
      return false;
   }

   d12_wine_device_register(device, adapter);

   /* Report the LUID DXGI gave the application so adapter matching works. */
   device->adapter_luid = (LUID){.LowPart = 0xA6A00001u, .HighPart = 0};
   IDXGIAdapter *dxgi_adapter = d12_wine_device_adapter(device);
   if (dxgi_adapter) {
      DXGI_ADAPTER_DESC desc;
      if (SUCCEEDED(IDXGIAdapter_GetDesc(dxgi_adapter, &desc)))
         device->adapter_luid = desc.AdapterLuid;
      IDXGIAdapter_Release(dxgi_adapter);
   }
   return true;
}

void
d12_platform_close_device(struct d12_device *device)
{
   if (device->sync) {
      device->sync->finalize(device->sync);
      device->sync = NULL;
   }

   int fd = device->dev.fd;
   agx_close_device(&device->dev);
   if (fd >= 0)
      d12_wine_close_fd(fd);
   device->dev.fd = -1;
   d12_wine_device_unregister(device);
}

void
d12_platform_signal_event(HANDLE event)
{
   if (!SetEvent(event))
      mesa_loge("SetEvent failed: %lu", (unsigned long)GetLastError());
}

/* Shared resources and cross-process sync are not implemented on this
 * backend yet, exactly like the native Linux backend. */
HRESULT
d12_platform_create_shared_bo(struct d12_device *device, size_t size,
                              unsigned align, enum agx_bo_flags flags,
                              const void *runtime_data,
                              uint32_t runtime_data_size, const char *label,
                              struct agx_bo **bo)
{
   (void)device; (void)size; (void)align; (void)flags; (void)runtime_data;
   (void)runtime_data_size; (void)label;
   if (bo)
      *bo = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_import_shared_bo(struct d12_device *device, HANDLE handle,
                              void *runtime_data,
                              uint32_t *runtime_data_size, const char *label,
                              struct agx_bo **bo)
{
   (void)device; (void)handle; (void)runtime_data; (void)runtime_data_size;
   (void)label;
   if (bo)
      *bo = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_export_shared_bo(struct d12_device *device, struct agx_bo *bo,
                              const SECURITY_ATTRIBUTES *attributes,
                              DWORD access, LPCWSTR name, HANDLE *handle)
{
   (void)device; (void)bo; (void)attributes; (void)access; (void)name;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_create_shared_sync(struct d12_device *device,
                                uint64_t initial_value, uint32_t *syncobj)
{
   (void)device; (void)initial_value;
   if (syncobj)
      *syncobj = 0;
   return E_NOTIMPL;
}

HRESULT
d12_platform_import_shared_sync(struct d12_device *device, HANDLE handle,
                                uint32_t *syncobj)
{
   (void)device; (void)handle;
   if (syncobj)
      *syncobj = 0;
   return E_NOTIMPL;
}

HRESULT
d12_platform_export_shared_sync(
   struct d12_device *device, uint32_t syncobj,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle)
{
   (void)device; (void)syncobj; (void)attributes; (void)access; (void)name;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_query_shared_sync(struct d12_device *device, uint32_t syncobj,
                               uint64_t *value)
{
   (void)device; (void)syncobj; (void)value;
   return E_NOTIMPL;
}

HRESULT
d12_platform_set_bo_resident(struct d12_device *device, struct agx_bo *bo,
                             bool resident)
{
   (void)device; (void)bo; (void)resident;
   return S_OK;
}

HRESULT
d12_platform_open_shared_handle_by_name(struct d12_device *device,
                                        LPCWSTR name, DWORD access,
                                        HANDLE *handle)
{
   (void)device; (void)name; (void)access;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}
