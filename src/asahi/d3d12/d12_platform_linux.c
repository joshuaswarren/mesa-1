/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_private.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/poll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>

extern bool asahi_noop_is_registered_fd(int fd)
   __attribute__((weak, visibility("default")));
extern bool asahi_noop_d3d12_graphics_qualification_allowed(int fd)
   __attribute__((weak, visibility("default")));

/* Physical qualification must not accept noop memfds or arbitrary handles.
 * Shim fstat and DRM version deliberately emulate a real character node. */
bool
d12_platform_is_real_asahi_render_fd(int fd)
{
   if ((asahi_noop_is_registered_fd && asahi_noop_is_registered_fd(fd)) ||
       (asahi_noop_d3d12_graphics_qualification_allowed &&
        !asahi_noop_is_registered_fd))
      return false; /* Older loaded shim without identity hook also fails closed. */
   struct stat st;
   if (fd < 0 || fstat(fd, &st) || !S_ISCHR(st.st_mode) ||
       drmGetNodeTypeFromFd(fd) != DRM_NODE_RENDER)
      return false;
   drmVersionPtr version = drmGetVersion(fd);
   bool valid = version && version->name && !strcmp(version->name, "asahi");
   drmFreeVersion(version);
   return valid;
}

static int
d12_open_asahi_fd(void)
{
   drmDevicePtr devices[8];
   int count = drmGetDevices2(0, devices, ARRAY_SIZE(devices));
   int result = -1;

   for (int i = 0; i < count; i++) {
      if (!(devices[i]->available_nodes & (1 << DRM_NODE_RENDER)))
         continue;

      int fd = open(devices[i]->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;

      drmVersionPtr version = drmGetVersion(fd);
      bool is_asahi = version && !strcmp(version->name, "asahi");
      drmFreeVersion(version);

      if (is_asahi) {
         result = fd;
         break;
      }
      close(fd);
   }

   if (count > 0)
      drmFreeDevices(devices, count);

   return result;
}

bool
d12_platform_open_device(struct d12_device *device, IUnknown *adapter)
{
   (void)adapter;

   int fd = d12_open_asahi_fd();
   if (fd < 0)
      return false;

   device->dev.fd = fd;
   if (!agx_open_device(NULL, &device->dev)) {
      close(fd);
      return false;
   }

   device->sync = util_sync_provider_drm(fd);
   if (!device->sync) {
      agx_close_device(&device->dev);
      return false;
   }

   device->adapter_luid = (LUID){.LowPart = 0xA6A00001u, .HighPart = 0};
   return true;
}

void
d12_platform_close_device(struct d12_device *device)
{
   if (device->sync) {
      device->sync->finalize(device->sync);
      device->sync = NULL;
   }

   agx_close_device(&device->dev);
}

void
d12_platform_signal_event(HANDLE event)
{
   int fd = (int)(intptr_t)event;

   /* Native Linux tests model a Win32 auto-reset event with eventfd. Avoid
    * accumulating another token when the event is already signalled. */
   struct pollfd pfd = {.fd = fd, .events = POLLIN};
   if (poll(&pfd, 1, 0) > 0)
      return;

   const uint64_t one = 1;
   if (write(fd, &one, sizeof(one)) != sizeof(one) && errno != EAGAIN)
      mesa_loge("fence event write failed: %s", strerror(errno));
}

HRESULT
d12_platform_create_shared_bo(struct d12_device *device, size_t size,
                              unsigned align, enum agx_bo_flags flags,
                              const void *runtime_data,
                              uint32_t runtime_data_size, const char *label,
                              struct agx_bo **bo)
{
   (void)device;
   (void)size;
   (void)align;
   (void)flags;
   (void)runtime_data;
   (void)runtime_data_size;
   (void)label;
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
   (void)device;
   (void)handle;
   (void)runtime_data;
   (void)runtime_data_size;
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
   (void)device;
   (void)bo;
   (void)attributes;
   (void)access;
   (void)name;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_create_shared_sync(struct d12_device *device,
                                uint64_t initial_value, uint32_t *syncobj)
{
   (void)device;
   (void)initial_value;
   if (syncobj)
      *syncobj = 0;
   return E_NOTIMPL;
}

HRESULT
d12_platform_import_shared_sync(struct d12_device *device, HANDLE handle,
                                uint32_t *syncobj)
{
   (void)device;
   (void)handle;
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
   (void)device;
   (void)syncobj;
   (void)attributes;
   (void)access;
   (void)name;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}

HRESULT
d12_platform_query_shared_sync(struct d12_device *device, uint32_t syncobj,
                               uint64_t *value)
{
   (void)device;
   (void)syncobj;
   (void)value;
   return E_NOTIMPL;
}

HRESULT
d12_platform_set_bo_resident(struct d12_device *device, struct agx_bo *bo,
                             bool resident)
{
   (void)device;
   (void)bo;
   (void)resident;
   /* DRM manages residency at submit time; there is no explicit equivalent
    * to D3DKMTMakeResident/Evict for this driver. */
   return S_OK;
}

HRESULT
d12_platform_open_shared_handle_by_name(struct d12_device *device,
                                        LPCWSTR name, DWORD access,
                                        HANDLE *handle)
{
   (void)device;
   (void)name;
   (void)access;
   if (handle)
      *handle = NULL;
   return E_NOTIMPL;
}
