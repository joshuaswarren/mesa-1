/*
 * Copyright 2026 the Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * agx_device_ops_t over D3DKMT. See agx_device_d3dkmt.h for why, and
 * agx_d3dkmt.h's CONTRACT ASSUMPTIONS block for every kernel-side guess this
 * file makes, tagged with the contract question it answers.
 *
 * Each op below is annotated with the AGX ioctl it replaces and with the one
 * thing that changes semantically, because in three places the change is real
 * rather than cosmetic:
 *
 *   VM_BIND    DRM binds synchronously; WDDM page-table updates complete on a
 *              paging fence. This file restores synchrony by waiting.
 *   SUBMIT     DRM submits waits, work and signals as one atomic ioctl; WDDM
 *              has no sync list on SubmitCommand and expresses the same thing
 *              as three ordered calls on one context.
 *   GEM handle DRM GEM handles are dense small integers; D3DKMT handles are
 *              not, and agx_device.c iterates the handle space.
 *
 * RUNTIME STATUS: this path cross-compiles for aarch64-w64-mingw32 as part
 * of the real driver. The hardware acceptance test lives in
 * src/asahi/d3d12/tests/shared_interop.c; run it in a matching Windows/AGX
 * session whenever resource, synchronization or private-data ABI changes.
 */

#include "agx_device_d3dkmt.h"
#include "agx_d3dkmt_priv.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winternl.h>

#include "clc/asahi_clc.h"
#include "util/macros.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "agx_abi.h"
#include "agx_bo.h"
#include "layout.h"

/* The mirrored constants in agx_d3dkmt_core.h must equal the real ones. This
 * is the tripwire that keeps the Mesa-free core honest. */
static_assert(AGX_D3DKMT_BIND_UNBIND == DRM_ASAHI_BIND_UNBIND, "BIND_UNBIND");
static_assert(AGX_D3DKMT_BIND_READ == DRM_ASAHI_BIND_READ, "BIND_READ");
static_assert(AGX_D3DKMT_BIND_WRITE == DRM_ASAHI_BIND_WRITE, "BIND_WRITE");
static_assert(AGX_D3DKMT_BIND_SINGLE_PAGE == DRM_ASAHI_BIND_SINGLE_PAGE,
              "BIND_SINGLE_PAGE");
static_assert(AGX_D3DKMT_BO_SHARED == AGX_BO_SHARED, "BO_SHARED");
static_assert(AGX_D3DKMT_BO_LOW_VA == AGX_BO_LOW_VA, "BO_LOW_VA");
static_assert(AGX_D3DKMT_BO_EXEC == AGX_BO_EXEC, "BO_EXEC");
static_assert(AGX_D3DKMT_BO_WRITEBACK == AGX_BO_WRITEBACK, "BO_WRITEBACK");
static_assert(AGX_D3DKMT_BO_SHAREABLE == AGX_BO_SHAREABLE, "BO_SHAREABLE");
static_assert(AGX_D3DKMT_BO_READONLY == AGX_BO_READONLY, "BO_READONLY");
static_assert(AGX_D3DKMT_BO_PRIMARY == AGX_BO_PRIMARY, "BO_PRIMARY");
static_assert(AGX_D3DKMT_PAGE_SIZE == AIL_PAGESIZE, "page size");
static_assert(AGXW_PARAMS_GLOBAL_SIZE == sizeof(struct drm_asahi_params_global),
              "GET_PARAMS escape payload size");
static_assert(AGXW_GET_TIME_SIZE == sizeof(struct drm_asahi_get_time),
              "GET_TIME escape payload size");
static_assert((int)AGXW_SYNC_BINARY == (int)DRM_ASAHI_SYNC_SYNCOBJ,
              "SYNC_SYNCOBJ");
static_assert((int)AGXW_SYNC_TIMELINE == (int)DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
              "SYNC_TIMELINE_SYNCOBJ");
static_assert(sizeof(struct agxw_sync) == sizeof(struct drm_asahi_sync),
              "agxw_sync mirrors drm_asahi_sync");
static_assert((int)AGXW_PRIORITY_LOW == (int)DRM_ASAHI_PRIORITY_LOW,
              "priority LOW");
static_assert((int)AGXW_PRIORITY_REALTIME == (int)DRM_ASAHI_PRIORITY_REALTIME,
              "priority REALTIME");

static void d3dkmt_log(const char *format, ...);

/* -------------------------------------------------------------------------
 * Process-wide D3DKMT device lease.
 *
 * Dxgkrnl exposes one GPU-VA root per process and adapter.  The KMD also
 * associates its VM/ASID with D3DKMTCreateDevice, so sharing only the VA
 * reservation is insufficient: a logical Vulkan device created through a
 * second KMT device can update the process VA while its render context points
 * at a different, unpublished KMD root.
 *
 * Mesa is statically linked into the native D3D12 DLL and the Vulkan ICD.
 * A named section therefore carries the process-global KMT device handle and
 * its owning adapter handle across module boundaries.  Paging queues,
 * contexts, fences, and frontend state remain private to each winsys.
 * ------------------------------------------------------------------------- */
#define AGX_D3DKMT_PROCESS_DEVICE_MAGIC   0x56445841u /* 'AXDV' */
#define AGX_D3DKMT_PROCESS_DEVICE_VERSION 1u

struct agx_d3dkmt_process_device_shared {
   uint32_t magic;
   uint32_t version;
   uint32_t mapping_bytes;
   uint32_t clients;
   uint32_t device;
   uint32_t owner_adapter;
};

struct agx_d3dkmt_process_device {
   HANDLE mapping;
   HANDLE mutex;
   struct agx_d3dkmt_process_device_shared *shared;
   bool attached;
};

static bool
process_device_lock(struct agx_d3dkmt_process_device *process)
{
   DWORD result = WaitForSingleObject(process->mutex, INFINITE);
   if (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED)
      return true;

   errno = EIO;
   return false;
}

static void
process_device_unlock(struct agx_d3dkmt_process_device *process)
{
   ReleaseMutex(process->mutex);
}

static void
process_device_close_mapping(struct agx_d3dkmt_process_device *process,
                             bool preserve_failed_state)
{
   if (process == NULL)
      return;
   if (process->shared != NULL)
      UnmapViewOfFile(process->shared);
   if (!preserve_failed_state && process->mapping != NULL)
      CloseHandle(process->mapping);
   if (process->mutex != NULL)
      CloseHandle(process->mutex);
   free(process);
}

static bool
process_device_open(struct agx_d3dkmt_winsys *ws)
{
   struct agx_d3dkmt_process_device *process = calloc(1, sizeof(*process));
   if (process == NULL)
      return false;

   wchar_t section_name[96];
   wchar_t mutex_name[96];
   DWORD pid = GetCurrentProcessId();
   swprintf(section_name, ARRAY_SIZE(section_name),
            L"Local\\MesaAgxDevice-%08lx-%08x-%08x", (unsigned long)pid,
            (unsigned)ws->luid.HighPart, (unsigned)ws->luid.LowPart);
   swprintf(mutex_name, ARRAY_SIZE(mutex_name),
            L"Local\\MesaAgxDeviceMutex-%08lx-%08x-%08x", (unsigned long)pid,
            (unsigned)ws->luid.HighPart, (unsigned)ws->luid.LowPart);

   process->mutex = CreateMutexW(NULL, FALSE, mutex_name);
   if (process->mutex == NULL)
      goto fail;

   process->mapping = CreateFileMappingW(
      INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
      sizeof(struct agx_d3dkmt_process_device_shared), section_name);
   if (process->mapping == NULL)
      goto fail;

   process->shared =
      MapViewOfFile(process->mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
   if (process->shared == NULL)
      goto fail;

   MEMORY_BASIC_INFORMATION info;
   if (VirtualQuery(process->shared, &info, sizeof(info)) != sizeof(info) ||
       info.RegionSize < sizeof(*process->shared)) {
      errno = EIO;
      goto fail;
   }

   if (!process_device_lock(process))
      goto fail;

   struct agx_d3dkmt_process_device_shared *shared = process->shared;
   if (shared->magic == 0) {
      memset(shared, 0, sizeof(*shared));
      shared->magic = AGX_D3DKMT_PROCESS_DEVICE_MAGIC;
      shared->version = AGX_D3DKMT_PROCESS_DEVICE_VERSION;
      shared->mapping_bytes = sizeof(*shared);
   }

   bool valid = shared->magic == AGX_D3DKMT_PROCESS_DEVICE_MAGIC &&
                shared->version == AGX_D3DKMT_PROCESS_DEVICE_VERSION &&
                shared->mapping_bytes == sizeof(*shared) &&
                info.RegionSize >= sizeof(*shared) &&
                shared->clients != UINT32_MAX &&
                ((shared->device == 0 && shared->clients == 0) ||
                 (shared->device != 0 && shared->owner_adapter != 0 &&
                  shared->clients != 0));
   if (!valid) {
      process_device_unlock(process);
      errno = EPROTO;
      goto fail;
   }

   if (shared->device == 0) {
      struct agx_kmt_createdevice create = {.hAdapter = ws->adapter};
      d3dkmt_log("create-device begin adapter=0x%08x shared=0",
                 (unsigned)ws->adapter);
      agx_kmt_status status = ws->fns.CreateDevice(&create);
      d3dkmt_log("create-device end status=0x%08x device=0x%08x shared=0",
                 (unsigned)status, (unsigned)create.hDevice);
      if (!AGX_KMT_SUCCESS(status)) {
         process_device_unlock(process);
         agx_d3dkmt_fail(status);
         goto fail;
      }

      shared->device = create.hDevice;
      shared->owner_adapter = ws->adapter;
   } else {
      d3dkmt_log("create-device reuse device=0x%08x owner-adapter=0x%08x "
                 "clients=%u",
                 shared->device, shared->owner_adapter, shared->clients);
   }

   shared->clients++;
   ws->device = shared->device;
   process->attached = true;
   ws->process_device = process;
   process_device_unlock(process);
   return true;

fail:
   process_device_close_mapping(process, false);
   return false;
}

static void
process_device_release(struct agx_d3dkmt_winsys *ws)
{
   struct agx_d3dkmt_process_device *process = ws->process_device;
   if (process == NULL)
      return;

   bool close_adapter = true;
   bool preserve_failed_state = false;
   if (!process->attached || !process_device_lock(process)) {
      preserve_failed_state = true;
      close_adapter = false;
      goto out;
   }

   struct agx_d3dkmt_process_device_shared *shared = process->shared;
   if (shared->magic != AGX_D3DKMT_PROCESS_DEVICE_MAGIC ||
       shared->version != AGX_D3DKMT_PROCESS_DEVICE_VERSION ||
       shared->device != ws->device || shared->clients == 0) {
      fprintf(stderr, "D3DKMT process device invariant failure\n");
      preserve_failed_state = true;
      close_adapter = shared->owner_adapter != ws->adapter;
      process_device_unlock(process);
      goto out;
   }

   shared->clients--;
   if (shared->clients == 0) {
      const agx_kmt_handle device = shared->device;
      const agx_kmt_handle owner_adapter = shared->owner_adapter;
      struct agx_kmt_destroydevice destroy = {.hDevice = device};
      agx_kmt_status status = ws->fns.DestroyDevice(&destroy);
      if (AGX_KMT_SUCCESS(status)) {
         shared->device = 0;
         shared->owner_adapter = 0;
         if (owner_adapter != 0 && owner_adapter != ws->adapter) {
            struct agx_kmt_closeadapter close = {
               .hAdapter = owner_adapter,
            };
            ws->fns.CloseAdapter(&close);
         }
      } else {
         fprintf(stderr, "D3DKMTDestroyDevice failed status=0x%08x\n",
                 (unsigned)status);
         preserve_failed_state = true;
         close_adapter = owner_adapter != ws->adapter;
      }
   } else if (shared->owner_adapter == ws->adapter) {
      /* The shared KMT device was created through this adapter handle. Keep
       * that parent alive until the final device lease is released. */
      close_adapter = false;
   }

   process->attached = false;
   process_device_unlock(process);

out:
   if (close_adapter && ws->adapter != 0) {
      struct agx_kmt_closeadapter close = {.hAdapter = ws->adapter};
      ws->fns.CloseAdapter(&close);
   }
   process_device_close_mapping(process, preserve_failed_state);
   ws->process_device = NULL;
   ws->device = 0;
   ws->adapter = 0;
}

/* -------------------------------------------------------------------------
 * Process-wide GPU VA broker.
 *
 * D3DKMT devices on one adapter do not own independent page-table roots:
 * VidMm's root is keyed by process.  Mesa is statically linked into both the
 * native D3D12 DLL and Honeykrisp (including the copy embedded by the D3D11
 * presentation UMD), so C globals cannot coordinate those modules.  A named
 * section and mutex keyed by PID+adapter LUID provide that missing process
 * scope without adding a service, a translation layer, or Linux code paths.
 *
 * The section is deliberately pointer-free.  NT and D3DKMT handles are valid
 * process-wide and may be published as integer values because the name also
 * includes the PID; no handle is ever consumed by another process.
 * ------------------------------------------------------------------------- */
#define AGX_D3DKMT_PROCESS_VA_MAGIC   0x41565841u /* 'AXVA' */
#define AGX_D3DKMT_PROCESS_VA_VERSION 3u
#define AGX_D3DKMT_FIXED_BO_COUNT     3u

struct agx_d3dkmt_shared_fixed_bo {
   uint64_t address;
   uint64_t size;
   uint64_t nt_handle;
   uint32_t allocation;
   uint32_t resource;
   uint32_t flags;
   uint32_t bind_flags;
};

struct agx_d3dkmt_process_va_shared {
   uint32_t magic;
   uint32_t version;
   uint32_t mapping_bytes;
   uint32_t word_count;
   uint64_t va_base;
   uint64_t va_size;
   uint32_t clients;
   uint32_t reservation_ready;
   uint32_t reservation_count;
   uint32_t layout_ready;
   struct agx_d3dkmt_va_range reservations[2];
   uint64_t main_base;
   uint64_t main_size;
   uint64_t usc_base;
   uint64_t usc_size;
   uint32_t anchor_device;
   struct agx_d3dkmt_shared_fixed_bo fixed[AGX_D3DKMT_FIXED_BO_COUNT];
   uint64_t bitmap[];
};

struct agx_d3dkmt_process_va {
   HANDLE mapping;
   HANDLE mutex;
   struct agx_d3dkmt_process_va_shared *shared;
   size_t mapping_bytes;
   bool attached;
};

static bool
process_va_lock(struct agx_d3dkmt_process_va *process)
{
   DWORD result = WaitForSingleObject(process->mutex, INFINITE);
   if (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED)
      return true;

   errno = EIO;
   return false;
}

static void
process_va_unlock(struct agx_d3dkmt_process_va *process)
{
   ReleaseMutex(process->mutex);
}

static void
process_va_close_mapping(struct agx_d3dkmt_process_va *process)
{
   if (process == NULL)
      return;
   if (process->shared != NULL)
      UnmapViewOfFile(process->shared);
   if (process->mapping != NULL)
      CloseHandle(process->mapping);
   if (process->mutex != NULL)
      CloseHandle(process->mutex);
   free(process);
}

static bool
process_va_open(struct agx_d3dkmt_winsys *ws, uint64_t base, uint64_t size)
{
   uint32_t words = agx_d3dkmt_va_bitmap_word_count(base, size);
   if (words == 0 ||
       words > (UINT32_MAX - sizeof(struct agx_d3dkmt_process_va_shared)) /
                  sizeof(uint64_t)) {
      errno = EOVERFLOW;
      return false;
   }

   size_t bytes = sizeof(struct agx_d3dkmt_process_va_shared) +
                  (size_t)words * sizeof(uint64_t);
   struct agx_d3dkmt_process_va *process = calloc(1, sizeof(*process));
   if (process == NULL)
      return false;

   wchar_t section_name[96];
   wchar_t mutex_name[96];
   DWORD pid = GetCurrentProcessId();
   swprintf(section_name, ARRAY_SIZE(section_name),
            L"Local\\MesaAgxVa-%08lx-%08x-%08x", (unsigned long)pid,
            (unsigned)ws->luid.HighPart, (unsigned)ws->luid.LowPart);
   swprintf(mutex_name, ARRAY_SIZE(mutex_name),
            L"Local\\MesaAgxVaMutex-%08lx-%08x-%08x", (unsigned long)pid,
            (unsigned)ws->luid.HighPart, (unsigned)ws->luid.LowPart);

   process->mutex = CreateMutexW(NULL, FALSE, mutex_name);
   if (process->mutex == NULL)
      goto fail;

   process->mapping =
      CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                         (DWORD)(bytes >> 32), (DWORD)bytes, section_name);
   if (process->mapping == NULL)
      goto fail;

   process->shared =
      MapViewOfFile(process->mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
   if (process->shared == NULL)
      goto fail;

   MEMORY_BASIC_INFORMATION info;
   if (VirtualQuery(process->shared, &info, sizeof(info)) != sizeof(info) ||
       info.RegionSize < sizeof(*process->shared)) {
      errno = EIO;
      goto fail;
   }

   process->mapping_bytes = bytes;
   if (!process_va_lock(process))
      goto fail;

   struct agx_d3dkmt_process_va_shared *shared = process->shared;
   if (shared->magic == 0) {
      if (info.RegionSize < bytes) {
         process_va_unlock(process);
         errno = EOVERFLOW;
         goto fail;
      }
      memset(shared, 0, bytes);
      shared->magic = AGX_D3DKMT_PROCESS_VA_MAGIC;
      shared->version = AGX_D3DKMT_PROCESS_VA_VERSION;
      shared->mapping_bytes = (uint32_t)bytes;
      shared->word_count = words;
      shared->va_base = base;
      shared->va_size = size;
   }

   bool valid = shared->magic == AGX_D3DKMT_PROCESS_VA_MAGIC &&
                shared->version == AGX_D3DKMT_PROCESS_VA_VERSION &&
                shared->mapping_bytes == bytes && shared->word_count == words &&
                shared->va_base == base && shared->va_size == size &&
                info.RegionSize >= bytes;
   process_va_unlock(process);
   if (!valid) {
      errno = EPROTO;
      goto fail;
   }

   ws->process_va = process;
   return true;

fail:
   process_va_close_mapping(process);
   return false;
}

/* Frontend-owned resource metadata is small (the D3D12 interop descriptor is
 * 264 bytes), but QueryResourceInfo reports an arbitrary kernel-owned size.
 * Keep a generous, explicit ceiling before allocating or opening buffers. */
#define AGX_D3DKMT_MAX_RUNTIME_DATA_SIZE (64u * 1024u)

static bool
d3dkmt_log_enabled(void)
{
   const char *value = getenv("NTAGX_ENUM_LOG");
   return value != NULL && value[0] != '\0' && value[0] != '0';
}

static void
d3dkmt_log(const char *format, ...)
{
   if (!d3dkmt_log_enabled())
      return;

   va_list args;
   va_start(args, format);
   fputs("NTAGX_D3DKMT ", stderr);
   vfprintf(stderr, format, args);
   fputc('\n', stderr);
   fflush(stderr);
   va_end(args);
}

/* -------------------------------------------------------------------------
 * Error mapping and waits.
 * ------------------------------------------------------------------------- */
int
agx_d3dkmt_fail(agx_kmt_status st)
{
   switch ((uint32_t)st) {
   case 0xC0000017u: /* STATUS_NO_MEMORY */
   case 0xC000009Au: /* STATUS_INSUFFICIENT_RESOURCES */
      errno = ENOMEM;
      break;
   case 0xC000000Du: /* STATUS_INVALID_PARAMETER */
   case 0xC0000008u: /* STATUS_INVALID_HANDLE */
      errno = EINVAL;
      break;
   case 0xC0000022u: /* STATUS_ACCESS_DENIED */
      errno = EACCES;
      break;
   case 0x00000102u: /* STATUS_TIMEOUT */
      errno = ETIME;
      break;
   case 0xC00000BBu: /* STATUS_NOT_SUPPORTED */
      errno = ENOTSUP;
      break;
   case 0xC01E0116u: /* STATUS_GRAPHICS_ALLOCATION_INVALID */
   case 0xC01E0110u: /* STATUS_GRAPHICS_NO_VIDEO_MEMORY */
      errno = ENOMEM;
      break;
   default:
      errno = EIO;
      break;
   }

   return -1;
}

int
agx_d3dkmt_poll_cpu_fences(const volatile uint64_t *const *current,
                           const uint64_t *wanted, uint32_t count,
                           bool wait_any, uint32_t timeout_ms,
                           uint32_t *first_signaled)
{
   if (timeout_ms == AGX_D3DKMT_INFINITE) {
      errno = EINVAL;
      return -1;
   }

   ULONGLONG start = GetTickCount64();
   for (;;) {
      enum agx_d3dkmt_cpu_wait_state state = agx_d3dkmt_cpu_wait_probe(
         current, wanted, count, wait_any, first_signaled);
      if (state == AGX_D3DKMT_CPU_WAIT_SATISFIED) {
         MemoryBarrier();
         return 0;
      }
      if (state == AGX_D3DKMT_CPU_WAIT_INVALID) {
         errno = EIO;
         return -1;
      }

      if (GetTickCount64() - start >= timeout_ms) {
         errno = ETIME;
         return -1;
      }

      if (!SwitchToThread())
         Sleep(1);
   }
}

int
agx_d3dkmt_wait_fence(struct agx_d3dkmt_winsys *ws, agx_kmt_handle fence,
                      const volatile uint64_t *fence_cpu, uint64_t value,
                      uint32_t timeout_ms)
{
   const agx_kmt_handle handles[1] = {fence};
   const uint64_t values[1] = {value};
   const volatile uint64_t *current[1] = {fence_cpu};

   enum agx_d3dkmt_cpu_wait_state state =
      agx_d3dkmt_cpu_wait_probe(current, values, 1, true, NULL);
   if (state == AGX_D3DKMT_CPU_WAIT_SATISFIED) {
      MemoryBarrier();
      return 0;
   }
   if (state == AGX_D3DKMT_CPU_WAIT_INVALID) {
      errno = EIO;
      return -1;
   }

   if (timeout_ms != AGX_D3DKMT_INFINITE)
      return agx_d3dkmt_poll_cpu_fences(current, values, 1, true, timeout_ms,
                                        NULL);

   struct agx_kmt_waitsyncfromcpu wait = {
      .hDevice = ws->device,
      .ObjectCount = 1,
      .ObjectHandleArray = handles,
      .FenceValueArray = values,
      .hAsyncEvent = NULL,
   };

   /* The NULL-event form is synchronous and per-call. The event form is not
    * used because dxgkrnl exposes no cancellation operation for a finite
    * caller-side timeout. */
   agx_kmt_status st =
      AGX_D3DKMT_CALL(ws, WaitForSynchronizationObjectFromCpu, &wait);
   if (!AGX_KMT_SUCCESS(st))
      return agx_d3dkmt_fail(st);

   MemoryBarrier();
   if (agx_d3dkmt_cpu_wait_probe(current, values, 1, true, NULL) ==
       AGX_D3DKMT_CPU_WAIT_SATISFIED)
      return 0;

   errno = EIO;
   return -1;
}

int
agx_d3dkmt_wait_paging(struct agx_d3dkmt_winsys *ws, uint64_t value)
{
   if (value == 0)
      return 0;

   /* Fast path: the paging fence has a CPU-visible value, so most waits cost
    * a load rather than a kernel transition. */
   if (ws->paging_fence_cpu != NULL && *ws->paging_fence_cpu >= value)
      return 0;

   return agx_d3dkmt_wait_fence(ws, ws->paging_fence, ws->paging_fence_cpu,
                                value,
                                AGX_D3DKMT_INFINITE);
}

/* -------------------------------------------------------------------------
 * Contexts.
 * ------------------------------------------------------------------------- */
static struct agx_d3dkmt_ctx *
ctx_get(struct agx_d3dkmt_winsys *ws, uint32_t queue_id)
{
   if (queue_id == 0 || queue_id > ws->ctx_capacity)
      return NULL;

   struct agx_d3dkmt_ctx *ctx = &ws->ctxs[queue_id - 1];
   return ctx->hContext ? ctx : NULL;
}

bool
agx_d3dkmt_queue_context_token(struct agx_device *dev, uint32_t queue_id,
                               uint32_t *token)
{
   struct agx_d3dkmt_winsys *ws;
   struct agx_d3dkmt_ctx *ctx;

   if (!dev || dev->ops.submit != agx_device_d3dkmt_ops.submit || !token)
      return false;
   ws = agx_d3dkmt_ws(dev);
   simple_mtx_lock(&ws->lock);
   ctx = ctx_get(ws, queue_id);
   if (ctx)
      *token = ctx->hContext;
   simple_mtx_unlock(&ws->lock);
   return ctx != NULL;
}

/* -------------------------------------------------------------------------
 * GEM_CREATE / GEM_CLOSE.
 * ------------------------------------------------------------------------- */
static void
d3dkmt_destroy_record(struct agx_d3dkmt_winsys *ws,
                      const struct agx_d3dkmt_allocation_record *record)
{
   if (record == NULL ||
       (record->allocation == 0 && record->resource == 0))
      return;

   const agx_kmt_handle list[1] = {record->allocation};
   struct agx_kmt_destroyallocation2 destroy = {
      .hDevice = ws->device,
      .hResource = record->resource,
      .phAllocationList = record->resource ? NULL : list,
      .AllocationCount = record->resource ? 0 : 1,
      .hPrivateRuntimeResourceHandle =
         (void *)record->runtime_resource,
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, DestroyAllocation2, &destroy);
   if (!AGX_KMT_SUCCESS(st))
      agx_d3dkmt_fail(st);
}

static bool
d3dkmt_make_resident(struct agx_d3dkmt_winsys *ws,
                     agx_kmt_handle allocation)
{
   if (!ws->policy.needs_residency)
      return true;

   const agx_kmt_handle list[1] = {allocation};
   struct agx_kmt_makeresident mr = {
      .hPagingQueue = ws->paging_queue,
      .NumAllocations = 1,
      .AllocationList = list,
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, MakeResident, &mr);
   if (!AGX_KMT_SUCCESS(st)) {
      agx_d3dkmt_fail(st);
      return false;
   }

   return agx_d3dkmt_wait_paging(ws, mr.PagingFenceValue) == 0;
}

static bool
d3dkmt_create_allocation(struct agx_d3dkmt_winsys *ws, uint64_t size,
                         enum agx_bo_flags flags,
                         enum agx_d3dkmt_external_memory_type external_type,
                         void *runtime_resource,
                         const void *runtime_data,
                         uint32_t runtime_data_size,
                         const void *resource_driver_data,
                         uint32_t resource_driver_data_size,
                         uint64_t standard_descriptor,
                         struct agx_d3dkmt_allocation_record *record)
{
   /*
    * CONTRACT.md SS12.1: the private-data size is pinned, and treated as a
    * memory-safety property rather than a convenience, because the D3D
    * runtime does not validate what the UMD claims against what the kernel
    * reads. Assert the width; never infer it.
    */
   struct agxw_alloc_private priv = {
      .version = AGXW_ALLOC_PRIVATE_VERSION,
      .struct_size = sizeof(priv),
      .size = size,
      .flags = agx_d3dkmt_agxw_alloc_flags((uint32_t)flags),
      .align_log2 = AGXW_PAGE_BITS,
      .reserved = standard_descriptor,
   };
   static_assert(sizeof(priv) == AGXW_ALLOC_PRIVATE_SIZE,
                 "allocation private data width");
   if (agxw_alloc_private_validate(&priv, sizeof(priv)) != 0) {
      errno = EINVAL;
      return false;
   }

   struct agx_kmt_allocationinfo2 info = {
      .pPrivateDriverData = &priv,
      .PrivateDriverDataSize = sizeof(priv),
   };

   /* Shared handles name a resource, not a bare allocation. The miniport's
    * current contract therefore requires the pinned resource blob whenever
    * CreateResource is set, even though the frontend has one allocation.
    * pPrivateRuntimeData is separate: dxgkrnl preserves it verbatim for the
    * frontend opening this handle in another process. */
   struct agxw_resource_private resource_priv = {
      .version = AGXW_RESOURCE_PRIVATE_VERSION,
      .struct_size = sizeof(resource_priv),
      .allocation_count = AGXW_MAX_ALLOCATIONS_PER_RESOURCE,
   };
   const bool external =
      external_type != AGX_D3DKMT_EXTERNAL_MEMORY_NONE;
   const bool runtime_owned = runtime_resource != NULL;
   const void *resource_blob =
      runtime_owned && resource_driver_data ? resource_driver_data
                                            : external ? &resource_priv : NULL;
   const uint32_t resource_blob_size =
      runtime_owned && resource_driver_data
         ? resource_driver_data_size
         : external ? (uint32_t)sizeof(resource_priv) : 0;
   struct agx_kmt_createallocation create = {
      .hDevice = ws->device,
      .pPrivateRuntimeData = runtime_data,
      .PrivateRuntimeDataSize = runtime_data_size,
      .pPrivateDriverData = resource_blob,
      .PrivateDriverDataSize = resource_blob_size,
      .NumAllocations = 1,
      .pAllocationInfo2 = &info,
      .Flags = runtime_owned
                  ? agx_d3dkmt_create_flags_runtime_resource((uint32_t)flags)
                  : agx_d3dkmt_create_flags_external((uint32_t)flags,
                                                     external_type),
      .hPrivateRuntimeResourceHandle = runtime_resource,
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, CreateAllocation2, &create);
   if (!AGX_KMT_SUCCESS(st)) {
      agx_d3dkmt_fail(st);
      return false;
   }

   *record = (struct agx_d3dkmt_allocation_record){
      .allocation = info.hAllocation,
      .resource = create.hResource,
      .global_share = create.hGlobalShare,
      .runtime_resource = (uintptr_t)runtime_resource,
      .external_type = external_type,
   };

   const bool expected_resource = external || runtime_owned;
   const bool expected_global =
      external_type == AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT;
   if (record->allocation == 0 || (!!record->resource != expected_resource) ||
       (!!record->global_share != expected_global)) {
      errno = EIO;
      d3dkmt_destroy_record(ws, record);
      memset(record, 0, sizeof(*record));
      return false;
   }

   /* Application heaps are controlled by D3D12 MakeResident/Evict.  Internal
    * renderer BOs and explicit external-memory BOs retain the winsys paging
    * path because no D3D runtime object exists to manage them.
    */
   if (!runtime_owned && !d3dkmt_make_resident(ws, record->allocation)) {
      d3dkmt_destroy_record(ws, record);
      memset(record, 0, sizeof(*record));
      return false;
   }

   return true;
}

static struct agx_bo *
d3dkmt_adopt_allocation(struct agx_device *dev, size_t size, size_t align,
                        enum agx_bo_flags flags,
                        const struct agx_d3dkmt_allocation_record *record,
                        bool allocate_and_bind)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_bo *bo;

   simple_mtx_lock(&ws->lock);
   uint32_t handle = agx_d3dkmt_pool_alloc_record(&ws->pool, record);
   simple_mtx_unlock(&ws->lock);

   if (handle == 0) {
      d3dkmt_destroy_record(ws, record);
      return NULL;
   }

   simple_mtx_lock(&dev->bo_map_lock);
   bo = agx_lookup_bo(dev, handle);
   agx_include_bo_handle(dev, handle);
   simple_mtx_unlock(&dev->bo_map_lock);

   assert(!memcmp(bo, &((struct agx_bo){}), sizeof(*bo)));
   bo->dev = dev;
   bo->size = size;
   bo->align = align;
   bo->flags = flags;
   bo->handle = handle;
   bo->uapi_handle = record->allocation;
   bo->prime_fd = -1;

   if (!allocate_and_bind)
      return bo;

   enum agx_va_flags va_flags = flags & AGX_BO_LOW_VA ? AGX_VA_USC : 0;
   bo->va = agx_va_alloc(dev, size, bo->align, va_flags, 0);
   if (!bo->va) {
      fprintf(stderr, "Failed to allocate BO VMA\n");
      agx_bo_free(dev, bo);
      return NULL;
   }

   uint32_t bind = DRM_ASAHI_BIND_READ;
   if (!(flags & AGX_BO_READONLY))
      bind |= DRM_ASAHI_BIND_WRITE;

   if (agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0, bind)) {
      agx_bo_free(dev, bo);
      return NULL;
   }

   return bo;
}

static void
d3dkmt_activate_bo(struct agx_device *dev, struct agx_bo *bo)
{
   assert(p_atomic_read(&bo->refcnt) == 0);
   p_atomic_set(&bo->refcnt, 1);
   agx_bo_read_set_changed(dev);
}

/*
 * Clone of agx_bo_alloc()/win_bo_alloc() with GEM_CREATE replaced by
 * D3DKMTCreateAllocation2. The handle split is the only structural change:
 * bo->handle stays a dense index into dev->bo_map (see agx_d3dkmt_core.h),
 * bo->uapi_handle carries the D3DKMT allocation handle that every kernel-
 * facing call site already passes.
 */
static struct agx_bo *
d3dkmt_bo_alloc(struct agx_device *dev, size_t size, size_t align,
                enum agx_bo_flags flags)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   /* executable implies low va */
   assert(!(flags & AGX_BO_EXEC) || (flags & AGX_BO_LOW_VA));
   flags = (enum agx_bo_flags)
      agx_d3dkmt_cpu_visible_flags((uint32_t)flags);

   struct agx_d3dkmt_allocation_record record;
   enum agx_d3dkmt_external_memory_type external_type =
      (flags & AGX_BO_SHAREABLE)
         ? AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32
         : AGX_D3DKMT_EXTERNAL_MEMORY_NONE;
   if (!d3dkmt_create_allocation(ws, size, flags, external_type, NULL, NULL,
                                 0, NULL, 0, 0, &record)) {
      fprintf(stderr, "D3DKMTCreateAllocation2 failed\n");
      return NULL;
   }

   struct agx_bo *bo =
      d3dkmt_adopt_allocation(dev, size, align, flags, &record, true);
   return bo;
}

struct agx_bo *
agx_d3dkmt_bo_create_runtime_resource(struct agx_device *dev, size_t size,
                                      unsigned align, enum agx_bo_flags flags,
                                      void *runtime_resource,
                                      const char *label)
{
   return agx_d3dkmt_bo_create_runtime_resource_with_driver_data(
      dev, size, align, flags, runtime_resource, NULL, 0, 0, label);
}

struct agx_bo *
agx_d3dkmt_bo_create_runtime_resource_with_driver_data(
   struct agx_device *dev, size_t size, unsigned align,
   enum agx_bo_flags flags, void *runtime_resource,
   const void *resource_driver_data, uint32_t resource_driver_data_size,
   uint64_t standard_descriptor,
   const char *label)
{
   struct agx_d3dkmt_winsys *ws;
   struct agx_d3dkmt_allocation_record record;
   struct agx_bo *bo;

   if (!dev || !dev->hdev || !runtime_resource || !size ||
       size > SIZE_MAX - (AIL_PAGESIZE - 1) ||
       (resource_driver_data_size && !resource_driver_data) ||
       (standard_descriptor &&
        agxw_standard_desc_unpack(standard_descriptor, NULL, NULL, NULL) != 0)) {
      errno = EINVAL;
      return NULL;
   }
   ws = agx_d3dkmt_ws(dev);
   if (!ws->runtime_owned) {
      errno = EINVAL;
      return NULL;
   }

   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);
   flags = (enum agx_bo_flags)
      agx_d3dkmt_cpu_visible_flags((uint32_t)flags);
   if (!d3dkmt_create_allocation(
          ws, size, flags, AGX_D3DKMT_EXTERNAL_MEMORY_NONE,
          runtime_resource, NULL, 0, resource_driver_data,
          resource_driver_data_size, standard_descriptor, &record))
      return NULL;

   bo = d3dkmt_adopt_allocation(dev, size, align, flags, &record, true);
   if (bo) {
      bo->label = label;
      d3dkmt_activate_bo(dev, bo);
   }
   return bo;
}

struct agx_bo *
agx_d3dkmt_bo_open_runtime_resource(
   struct agx_device *dev, size_t size, unsigned align,
   enum agx_bo_flags flags, void *runtime_resource,
   uint32_t allocation, uint32_t resource, const char *label)
{
   struct agx_d3dkmt_winsys *ws;
   struct agx_d3dkmt_allocation_record record;
   struct agx_bo *bo;

   if (!dev || !dev->hdev || !runtime_resource || !allocation || !resource ||
       !size || size > SIZE_MAX - (AIL_PAGESIZE - 1)) {
      errno = EINVAL;
      return NULL;
   }
   ws = agx_d3dkmt_ws(dev);
   if (!ws->runtime_owned) {
      errno = EINVAL;
      return NULL;
   }

   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);
   record = (struct agx_d3dkmt_allocation_record){
      .allocation = allocation,
      .resource = resource,
      .runtime_resource = (uintptr_t)runtime_resource,
   };
   bo = d3dkmt_adopt_allocation(dev, size, align, flags, &record, true);
   if (bo) {
      bo->label = label;
      d3dkmt_activate_bo(dev, bo);
   }
   return bo;
}

bool
agx_d3dkmt_bo_rotate_runtime_identities(struct agx_device *dev,
                                        struct agx_bo *const *bos,
                                        uint32_t count)
{
   struct agx_d3dkmt_winsys *ws;
   uint32_t stack_handles[8];
   uint32_t *dense_handles = stack_handles;
   bool result = false;

   if (!dev || dev->ops.submit != agx_device_d3dkmt_ops.submit ||
       (count && !bos))
      return false;
   if (count < 2)
      return true;
   if (count > ARRAY_SIZE(stack_handles)) {
      if ((size_t)count > SIZE_MAX / sizeof(*dense_handles))
         return false;
      dense_handles = malloc((size_t)count * sizeof(*dense_handles));
      if (!dense_handles)
         return false;
   }

   ws = agx_d3dkmt_ws(dev);
   simple_mtx_lock(&ws->lock);
   if (!ws->runtime_owned)
      goto out;

   for (uint32_t i = 0; i < count; ++i) {
      struct agx_d3dkmt_allocation_record record;
      struct agx_bo *bo = bos[i];
      if (!bo || bo->dev != dev || !bo->handle || !bo->uapi_handle ||
          !bo->va || bo->size != bos[0]->size ||
          bo->align != bos[0]->align || bo->flags != bos[0]->flags ||
          p_atomic_read(&bo->refcnt) <= 0 ||
          !agx_d3dkmt_pool_get_record(&ws->pool, bo->handle, &record) ||
          record.allocation != bo->uapi_handle ||
          !record.runtime_resource) {
         goto out;
      }
      for (uint32_t j = 0; j < i; ++j) {
         if (bos[j] == bo)
            goto out;
      }
      dense_handles[i] = bo->handle;
   }

   if (!agx_d3dkmt_pool_rotate_runtime_identities(
          &ws->pool, dense_handles, count))
      goto out;

   struct {
      uint32_t uapi_handle;
      struct agx_va *va;
      void *map;
      uint64_t writer;
      time_t last_used;
      uint64_t cache_generation;
      int prime_fd;
   } first = {
      .uapi_handle = bos[0]->uapi_handle,
      .va = bos[0]->va,
      .map = bos[0]->_map,
      .writer = bos[0]->writer,
      .last_used = bos[0]->last_used,
      .cache_generation = bos[0]->cache_generation,
      .prime_fd = bos[0]->prime_fd,
   };

   for (uint32_t i = 0; i + 1 < count; ++i) {
      bos[i]->uapi_handle = bos[i + 1]->uapi_handle;
      bos[i]->va = bos[i + 1]->va;
      bos[i]->_map = bos[i + 1]->_map;
      bos[i]->writer = bos[i + 1]->writer;
      bos[i]->last_used = bos[i + 1]->last_used;
      bos[i]->cache_generation = bos[i + 1]->cache_generation;
      bos[i]->prime_fd = bos[i + 1]->prime_fd;
   }
   bos[count - 1]->uapi_handle = first.uapi_handle;
   bos[count - 1]->va = first.va;
   bos[count - 1]->_map = first.map;
   bos[count - 1]->writer = first.writer;
   bos[count - 1]->last_used = first.last_used;
   bos[count - 1]->cache_generation = first.cache_generation;
   bos[count - 1]->prime_fd = first.prime_fd;
   agx_bo_read_set_changed(dev);
   result = true;

out:
   simple_mtx_unlock(&ws->lock);
   if (dense_handles != stack_handles)
      free(dense_handles);
   return result;
}

/*
 * GEM_CLOSE. Takes the DENSE handle, because agx_bo_free() memsets struct
 * agx_bo to zero before calling this (agx_device.c:100-105, deliberately, to
 * avoid a lookup race) -- so bo->uapi_handle is already gone and the pool is
 * the only remaining way to recover the D3DKMT handle.
 */
static void
d3dkmt_bo_free(struct agx_device *dev, uint32_t handle)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   simple_mtx_lock(&ws->lock);
   struct agx_d3dkmt_allocation_record record;
   bool found = agx_d3dkmt_pool_release_record(&ws->pool, handle, &record);
   simple_mtx_unlock(&ws->lock);

   if (!found)
      return;

   d3dkmt_destroy_record(ws, &record);
}

static bool
d3dkmt_external_query(struct agx_d3dkmt_winsys *ws, uintptr_t handle,
                      enum agx_d3dkmt_external_memory_type external_type,
                      uint32_t *runtime_data_size)
{
   uint32_t runtime_size = 0;
   uint32_t total_driver_size = 0;
   uint32_t resource_driver_size = 0;
   uint32_t allocation_count = 0;
   agx_kmt_status st;

   if (external_type == AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32) {
      if (handle == 0)
         return false;
      struct agx_kmt_queryresourceinfofromnthandle query = {
         .hDevice = ws->device,
         .hNtHandle = (void *)handle,
      };
      st = AGX_D3DKMT_CALL(ws, QueryResourceInfoFromNtHandle, &query);
      runtime_size = query.PrivateRuntimeDataSize;
      total_driver_size = query.TotalPrivateDriverDataSize;
      resource_driver_size = query.ResourcePrivateDriverDataSize;
      allocation_count = query.NumAllocations;
   } else if (external_type ==
              AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT) {
      if (handle == 0 || handle > UINT32_MAX)
         return false;
      struct agx_kmt_queryresourceinfo query = {
         .hDevice = ws->device,
         .hGlobalShare = (agx_kmt_handle)handle,
      };
      st = AGX_D3DKMT_CALL(ws, QueryResourceInfo, &query);
      runtime_size = query.PrivateRuntimeDataSize;
      total_driver_size = query.TotalPrivateDriverDataSize;
      resource_driver_size = query.ResourcePrivateDriverDataSize;
      allocation_count = query.NumAllocations;
   } else {
      errno = EINVAL;
      return false;
   }

   if (!AGX_KMT_SUCCESS(st)) {
      agx_d3dkmt_fail(st);
      return false;
   }

   /* This miniport's source-pinned contract is one allocation per resource,
    * one AGXW resource blob, and exactly one AGXW allocation blob. Runtime
    * data is frontend-owned and may be absent or carry a versioned object
    * descriptor. Refuse a foreign driver object before supplying buffers to
    * OpenResource. */
   if (allocation_count != AGXW_MAX_ALLOCATIONS_PER_RESOURCE ||
       runtime_size > AGX_D3DKMT_MAX_RUNTIME_DATA_SIZE ||
       resource_driver_size != AGXW_RESOURCE_PRIVATE_SIZE ||
       total_driver_size != AGXW_ALLOC_PRIVATE_SIZE) {
      errno = EINVAL;
      return false;
   }

   *runtime_data_size = runtime_size;
   return true;
}

static bool
d3dkmt_external_open(
   struct agx_d3dkmt_winsys *ws, uintptr_t handle,
   enum agx_d3dkmt_external_memory_type external_type,
   struct agx_d3dkmt_allocation_record *record,
   struct agxw_alloc_private *private_data, void *runtime_data,
   uint32_t *runtime_data_size)
{
   uint32_t queried_runtime_size = 0;
   if (!d3dkmt_external_query(ws, handle, external_type,
                              &queried_runtime_size))
      return false;

   const uint32_t caller_runtime_size =
      runtime_data_size ? *runtime_data_size : 0;
   if (runtime_data != NULL &&
       (runtime_data_size == NULL ||
        caller_runtime_size < queried_runtime_size)) {
      if (runtime_data_size)
         *runtime_data_size = queried_runtime_size;
      errno = ENOSPC;
      return false;
   }

   void *queried_runtime_data =
      queried_runtime_size ? malloc(queried_runtime_size) : NULL;
   if (queried_runtime_size && queried_runtime_data == NULL) {
      errno = ENOMEM;
      return false;
   }

   uint8_t total_driver_data[AGXW_ALLOC_PRIVATE_SIZE] = {0};
   struct agxw_resource_private resource_private = {0};
   struct agx_kmt_openallocationinfo2 info = {0};
   agx_kmt_handle resource = 0;
   agx_kmt_status st;

   if (external_type == AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32) {
      struct agx_kmt_openresourcefromnthandle open = {
         .hDevice = ws->device,
         .hNtHandle = (void *)handle,
         .NumAllocations = AGXW_MAX_ALLOCATIONS_PER_RESOURCE,
         .pOpenAllocationInfo2 = &info,
         .PrivateRuntimeDataSize = queried_runtime_size,
         .pPrivateRuntimeData = queried_runtime_data,
         .ResourcePrivateDriverDataSize = sizeof(resource_private),
         .pResourcePrivateDriverData = &resource_private,
         .TotalPrivateDriverDataBufferSize = sizeof(total_driver_data),
         .pTotalPrivateDriverDataBuffer = total_driver_data,
      };
      st = AGX_D3DKMT_CALL(ws, OpenResourceFromNtHandle, &open);
      resource = open.hResource;
      if (AGX_KMT_SUCCESS(st) &&
          (open.hKeyedMutex != 0 || open.hSyncObject != 0)) {
         errno = ENOTSUP;
         st = (agx_kmt_status)0xc00000bbu; /* STATUS_NOT_SUPPORTED */
      }
   } else {
      struct agx_kmt_openresource open = {
         .hDevice = ws->device,
         .hGlobalShare = (agx_kmt_handle)handle,
         .NumAllocations = AGXW_MAX_ALLOCATIONS_PER_RESOURCE,
         .pOpenAllocationInfo2 = &info,
         .pPrivateRuntimeData = queried_runtime_data,
         .PrivateRuntimeDataSize = queried_runtime_size,
         .pResourcePrivateDriverData = &resource_private,
         .ResourcePrivateDriverDataSize = sizeof(resource_private),
         .pTotalPrivateDriverDataBuffer = total_driver_data,
         .TotalPrivateDriverDataBufferSize = sizeof(total_driver_data),
      };
      st = AGX_D3DKMT_CALL(ws, OpenResource, &open);
      resource = open.hResource;
   }

   if (!AGX_KMT_SUCCESS(st)) {
      if (resource != 0) {
         const struct agx_d3dkmt_allocation_record opened = {
            .allocation = info.hAllocation,
            .resource = resource,
         };
         d3dkmt_destroy_record(ws, &opened);
      }
      agx_d3dkmt_fail(st);
      free(queried_runtime_data);
      return false;
   }

   const uintptr_t data_begin = (uintptr_t)total_driver_data;
   const uintptr_t data_end = data_begin + sizeof(total_driver_data);
   const uintptr_t private_begin = (uintptr_t)info.pPrivateDriverData;
   if (resource == 0 || info.hAllocation == 0 ||
       info.PrivateDriverDataSize != AGXW_ALLOC_PRIVATE_SIZE ||
       private_begin < data_begin || private_begin > data_end ||
       AGXW_ALLOC_PRIVATE_SIZE > data_end - private_begin ||
       agxw_resource_private_validate(&resource_private,
                                      sizeof(resource_private)) != 0 ||
       resource_private.allocation_count !=
          AGXW_MAX_ALLOCATIONS_PER_RESOURCE) {
      errno = EINVAL;
      const struct agx_d3dkmt_allocation_record opened = {
         .allocation = info.hAllocation,
         .resource = resource,
      };
      d3dkmt_destroy_record(ws, &opened);
      free(queried_runtime_data);
      return false;
   }

   memcpy(private_data, info.pPrivateDriverData, sizeof(*private_data));
   if (agxw_alloc_private_validate(private_data, sizeof(*private_data)) != 0 ||
       !(private_data->flags & AGXW_ALLOC_SHAREABLE) ||
       (private_data->flags & AGXW_ALLOC_VM_PRIVATE) ||
       private_data->size == 0 ||
       private_data->size > SIZE_MAX - (AIL_PAGESIZE - 1)) {
      errno = EINVAL;
      const struct agx_d3dkmt_allocation_record opened = {
         .allocation = info.hAllocation,
         .resource = resource,
      };
      d3dkmt_destroy_record(ws, &opened);
      free(queried_runtime_data);
      return false;
   }

   *record = (struct agx_d3dkmt_allocation_record){
      .allocation = info.hAllocation,
      .resource = resource,
      .global_share =
         external_type == AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT
            ? (agx_kmt_handle)handle
            : 0,
      .external_type = external_type,
   };

   if (!d3dkmt_make_resident(ws, record->allocation)) {
      d3dkmt_destroy_record(ws, record);
      memset(record, 0, sizeof(*record));
      free(queried_runtime_data);
      return false;
   }

   if (runtime_data != NULL && queried_runtime_size)
      memcpy(runtime_data, queried_runtime_data, queried_runtime_size);
   if (runtime_data_size)
      *runtime_data_size = queried_runtime_size;
   free(queried_runtime_data);
   return true;
}

static bool
d3dkmt_create_fixed_anchors(struct agx_device *dev)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_process_va_shared *shared = ws->process_va->shared;
   const struct {
      uint64_t address;
      uint64_t size;
      enum agx_bo_flags flags;
      uint32_t bind_flags;
   } desc[AGX_D3DKMT_FIXED_BO_COUNT] = {
      {AGX_ZERO_PAGE_ADDRESS, AGX_ZERO_PAGE_SIZE, 0, DRM_ASAHI_BIND_READ},
      {AGX_SCRATCH_PAGE_ADDRESS, AIL_PAGESIZE, 0,
       DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE},
      {dev->chip == AGX_CHIP_G17P ? AGX_G17P_RENDER_CONTEXT_BASE
                                  : LIBAGX_PRINTF_BUFFER_ADDRESS,
       dev->chip == AGX_CHIP_G17P ? AGX_G17P_RENDER_CONTEXT_SIZE
                                  : LIBAGX_PRINTF_BUFFER_SIZE,
       AGX_BO_WRITEBACK, DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE},
   };
   struct agx_d3dkmt_allocation_record records[AGX_D3DKMT_FIXED_BO_COUNT] = {0};
   uint32_t created = 0;
   uint32_t bound = 0;

   for (uint32_t i = 0; i < AGX_D3DKMT_FIXED_BO_COUNT; ++i) {
      enum agx_bo_flags flags =
         desc[i].flags | AGX_BO_SHARED | AGX_BO_SHAREABLE;
      if (!d3dkmt_create_allocation(
             ws, desc[i].size, flags,
             AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32, NULL, NULL, 0,
             NULL, 0, 0, &records[i]))
         goto fail;
      created++;

      struct agx_bo anchor = {.uapi_handle = records[i].allocation};
      if (agx_bo_bind(dev, &anchor, desc[i].address, desc[i].size, 0,
                      desc[i].bind_flags) != 0)
         goto fail;
      bound++;

      OBJECT_ATTRIBUTES attributes;
      InitializeObjectAttributes(&attributes, NULL, 0, NULL, NULL);
      void *nt_handle = NULL;
      agx_kmt_status st =
         AGX_D3DKMT_CALL(ws, ShareObjects, 1, &records[i].resource, &attributes,
                              AGX_KMT_SHARED_ALLOCATION_ALL_ACCESS, &nt_handle);
      if (!AGX_KMT_SUCCESS(st) || nt_handle == NULL) {
         if (!AGX_KMT_SUCCESS(st))
            agx_d3dkmt_fail(st);
         else
            errno = EIO;
         goto fail;
      }

      shared->fixed[i] = (struct agx_d3dkmt_shared_fixed_bo){
         .address = desc[i].address,
         .size = desc[i].size,
         .nt_handle = (uintptr_t)nt_handle,
         .allocation = records[i].allocation,
         .resource = records[i].resource,
         .flags = flags,
         .bind_flags = desc[i].bind_flags,
      };
   }

   shared->anchor_device = ws->device;
   return true;

fail:
   for (uint32_t i = 0; i < bound; ++i)
      agx_bo_bind(dev, NULL, desc[i].address, desc[i].size, 0,
                  DRM_ASAHI_BIND_UNBIND);
   for (uint32_t i = 0; i < created; ++i) {
      if (shared->fixed[i].nt_handle != 0)
         CloseHandle((HANDLE)(uintptr_t)shared->fixed[i].nt_handle);
      d3dkmt_destroy_record(ws, &records[i]);
      memset(&shared->fixed[i], 0, sizeof(shared->fixed[i]));
   }
   shared->anchor_device = 0;
   return false;
}

static int
d3dkmt_fixed_bo_index(const struct agx_device *dev, uint64_t address)
{
   if (address == AGX_ZERO_PAGE_ADDRESS)
      return 0;
   if (address == AGX_SCRATCH_PAGE_ADDRESS)
      return 1;
   if (dev->chip == AGX_CHIP_G17P &&
       address == AGX_G17P_RENDER_CONTEXT_BASE)
      return 2;
   if (dev->chip != AGX_CHIP_G17P &&
       address == LIBAGX_PRINTF_BUFFER_ADDRESS)
      return 2;
   return -1;
}

static struct agx_bo *
d3dkmt_bo_alloc_fixed(struct agx_device *dev, size_t size,
                      enum agx_bo_flags flags, uint64_t fixed_va,
                      uint32_t bind_flags, const char *label)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   /* The D3D runtime owns allocation/resource sharing policy.  Until the DDI
    * shared-resource callbacks are connected, keep fixed bootstrap BOs local
    * to this runtime device and bind them at the same renderer ABI addresses.
    * This path deliberately avoids ShareObjects and never fabricates an NT
    * handle for a runtime-created allocation. */
   if (ws->runtime_owned) {
      const size_t aligned_size = ALIGN_POT(size, AIL_PAGESIZE);
      struct agx_d3dkmt_allocation_record record;
      if (!d3dkmt_create_allocation(
             ws, aligned_size, flags, AGX_D3DKMT_EXTERNAL_MEMORY_NONE, NULL,
             NULL, 0, NULL, 0, 0, &record))
         return NULL;

      struct agx_bo *bo = d3dkmt_adopt_allocation(
         dev, aligned_size, AIL_PAGESIZE, flags, &record, false);
      if (!bo)
         return NULL;

      if (agx_bo_bind(dev, bo, fixed_va, aligned_size, 0, bind_flags)) {
         agx_bo_free(dev, bo);
         return NULL;
      }

      bo->label = label;
      d3dkmt_activate_bo(dev, bo);
      return bo;
   }

   struct agx_d3dkmt_process_va *process = ws->process_va;
   int index = d3dkmt_fixed_bo_index(dev, fixed_va);
   if (process == NULL || !process->attached || index < 0 ||
       !process_va_lock(process)) {
      errno = EINVAL;
      return NULL;
   }

   struct agx_d3dkmt_process_va_shared *shared = process->shared;
   if (shared->fixed[0].nt_handle == 0 && !d3dkmt_create_fixed_anchors(dev)) {
      process_va_unlock(process);
      return NULL;
   }

   if (shared->anchor_device != ws->device) {
      process_va_unlock(process);
      errno = EPROTO;
      return NULL;
   }

   struct agx_d3dkmt_shared_fixed_bo *fixed = &shared->fixed[index];
   enum agx_bo_flags shared_flags = flags | AGX_BO_SHARED | AGX_BO_SHAREABLE;
   if (fixed->nt_handle == 0 || fixed->address != fixed_va ||
       fixed->size != ALIGN_POT(size, AIL_PAGESIZE) ||
       fixed->flags != (uint32_t)shared_flags ||
       fixed->bind_flags != bind_flags) {
      process_va_unlock(process);
      errno = EPROTO;
      return NULL;
   }

   struct agx_d3dkmt_allocation_record record;
   struct agxw_alloc_private private_data;
   bool opened = d3dkmt_external_open(ws, (uintptr_t)fixed->nt_handle,
                                      AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32,
                                      &record, &private_data, NULL, NULL);
   if (!opened || private_data.size != fixed->size) {
      if (opened)
         d3dkmt_destroy_record(ws, &record);
      process_va_unlock(process);
      errno = EIO;
      return NULL;
   }

   struct agx_bo *bo = d3dkmt_adopt_allocation(dev, fixed->size, AIL_PAGESIZE,
                                               shared_flags, &record, false);
   process_va_unlock(process);
   if (bo == NULL)
      return NULL;

   bo->label = label;
   d3dkmt_activate_bo(dev, bo);
   return bo;
}

struct agx_bo *
agx_d3dkmt_bo_create_external(
   struct agx_device *dev, size_t size, unsigned align,
   enum agx_bo_flags flags,
   enum agx_d3dkmt_external_memory_type external_type, const char *label)
{
   return agx_d3dkmt_bo_create_external_with_runtime_data(
      dev, size, align, flags, external_type, NULL, 0, label);
}

struct agx_bo *
agx_d3dkmt_bo_create_external_with_runtime_data(
   struct agx_device *dev, size_t size, unsigned align,
   enum agx_bo_flags flags,
   enum agx_d3dkmt_external_memory_type external_type,
   const void *runtime_data, uint32_t runtime_data_size, const char *label)
{
   if (external_type == AGX_D3DKMT_EXTERNAL_MEMORY_NONE ||
       external_type > AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT ||
       size == 0 || size > SIZE_MAX - (AIL_PAGESIZE - 1) ||
       runtime_data_size > AGX_D3DKMT_MAX_RUNTIME_DATA_SIZE ||
       (runtime_data_size != 0 && runtime_data == NULL)) {
      errno = EINVAL;
      return NULL;
   }

   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);
   flags |= AGX_BO_SHARED | AGX_BO_SHAREABLE;
   flags = (enum agx_bo_flags)
      agx_d3dkmt_cpu_visible_flags((uint32_t)flags);

   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_allocation_record record;
   if (!d3dkmt_create_allocation(ws, size, flags, external_type, NULL,
                                 runtime_data, runtime_data_size, NULL, 0,
                                 0, &record))
      return NULL;

   struct agx_bo *bo =
      d3dkmt_adopt_allocation(dev, size, align, flags, &record, true);
   if (bo != NULL) {
      bo->label = label;
      d3dkmt_activate_bo(dev, bo);
   }
   return bo;
}

struct agx_bo *
agx_d3dkmt_bo_import_external(
   struct agx_device *dev, uintptr_t handle,
   enum agx_d3dkmt_external_memory_type external_type, const char *label)
{
   return agx_d3dkmt_bo_import_external_with_runtime_data(
      dev, handle, external_type, NULL, NULL, label);
}

struct agx_bo *
agx_d3dkmt_bo_import_external_with_runtime_data(
   struct agx_device *dev, uintptr_t handle,
   enum agx_d3dkmt_external_memory_type external_type, void *runtime_data,
   uint32_t *runtime_data_size, const char *label)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_allocation_record record;
   struct agxw_alloc_private private_data;
   if (!d3dkmt_external_open(ws, handle, external_type, &record, &private_data,
                             runtime_data, runtime_data_size))
      return NULL;

   enum agx_bo_flags flags = AGX_BO_SHARED | AGX_BO_SHAREABLE;
   if (private_data.flags & AGXW_ALLOC_WRITEBACK)
      flags |= AGX_BO_WRITEBACK;
   const size_t size = ALIGN_POT((size_t)private_data.size, AIL_PAGESIZE);

   struct agx_bo *bo =
      d3dkmt_adopt_allocation(dev, size, AIL_PAGESIZE, flags, &record, true);
   if (bo != NULL) {
      bo->label = label;
      d3dkmt_activate_bo(dev, bo);
   }
   return bo;
}

int
agx_d3dkmt_bo_export_external(
   struct agx_device *dev, struct agx_bo *bo,
   enum agx_d3dkmt_external_memory_type external_type, uintptr_t *handle)
{
   return agx_d3dkmt_bo_export_external_with_attributes(
      dev, bo, external_type, NULL, AGX_KMT_SHARED_ALLOCATION_ALL_ACCESS,
      handle);
}

int
agx_d3dkmt_bo_export_external_with_attributes(
   struct agx_device *dev, struct agx_bo *bo,
   enum agx_d3dkmt_external_memory_type external_type,
   const void *object_attributes, uint32_t access, uintptr_t *handle)
{
   if (bo == NULL || handle == NULL ||
       external_type == AGX_D3DKMT_EXTERNAL_MEMORY_NONE ||
       external_type > AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT) {
      errno = EINVAL;
      return -1;
   }

   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_allocation_record record;
   simple_mtx_lock(&ws->lock);
   bool found =
      agx_d3dkmt_pool_get_record(&ws->pool, bo->handle, &record);
   simple_mtx_unlock(&ws->lock);
   if (!found || record.allocation != bo->uapi_handle ||
       record.external_type != external_type || record.resource == 0) {
      errno = EINVAL;
      return -1;
   }

   if (external_type == AGX_D3DKMT_EXTERNAL_MEMORY_OPAQUE_WIN32_KMT) {
      if (object_attributes != NULL) {
         errno = EINVAL;
         return -1;
      }
      if (record.global_share == 0) {
         errno = EINVAL;
         return -1;
      }
      /* Non-owning by Vulkan definition: return the stable identifier itself,
       * not a duplicated/closable kernel handle. */
      *handle = (uintptr_t)record.global_share;
      return 0;
   }

   OBJECT_ATTRIBUTES default_attributes;
   if (object_attributes == NULL) {
      InitializeObjectAttributes(&default_attributes, NULL, 0, NULL, NULL);
      object_attributes = &default_attributes;
   }

   if (access == 0)
      access = AGX_KMT_SHARED_ALLOCATION_ALL_ACCESS;

   void *nt_handle = NULL;
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, ShareObjects,
      1, &record.resource, (void *)object_attributes, access, &nt_handle);
   if (!AGX_KMT_SUCCESS(st) || nt_handle == NULL) {
      if (AGX_KMT_SUCCESS(st))
         errno = EIO;
      else
         agx_d3dkmt_fail(st);
      return -1;
   }

   /* The caller owns this NT handle and closes it. The BO retains no copy. */
   *handle = (uintptr_t)nt_handle;
   return 0;
}

static bool
d3dkmt_bo_record(struct agx_device *dev, struct agx_bo *bo,
                 struct agx_d3dkmt_allocation_record *record)
{
   if (!dev || !bo || !record) {
      errno = EINVAL;
      return false;
   }

   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   simple_mtx_lock(&ws->lock);
   bool found = agx_d3dkmt_pool_get_record(&ws->pool, bo->handle, record);
   simple_mtx_unlock(&ws->lock);
   if (!found || record->allocation != bo->uapi_handle) {
      errno = EINVAL;
      return false;
   }
   return true;
}

bool
agx_d3dkmt_bo_runtime_identity(struct agx_device *dev, struct agx_bo *bo,
                               uint32_t *allocation, uint32_t *resource)
{
   struct agx_d3dkmt_allocation_record record;

   if (!allocation || !resource || !d3dkmt_bo_record(dev, bo, &record) ||
       !record.allocation || !record.resource || !record.runtime_resource ||
       record.external_type != AGX_D3DKMT_EXTERNAL_MEMORY_NONE) {
      errno = EINVAL;
      return false;
   }

   *allocation = record.allocation;
   *resource = record.resource;
   return true;
}

int
agx_d3dkmt_bo_make_resident(struct agx_device *dev, struct agx_bo *bo)
{
   struct agx_d3dkmt_allocation_record record;
   if (!d3dkmt_bo_record(dev, bo, &record))
      return -1;
   return d3dkmt_make_resident(agx_d3dkmt_ws(dev), record.allocation) ? 0 : -1;
}

int
agx_d3dkmt_bo_evict(struct agx_device *dev, struct agx_bo *bo)
{
   struct agx_d3dkmt_allocation_record record;
   if (!d3dkmt_bo_record(dev, bo, &record))
      return -1;

   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   if (!ws->policy.needs_residency)
      return 0;

   const agx_kmt_handle allocation = record.allocation;
   struct agx_kmt_evict evict = {
      .hDevice = ws->device,
      .NumAllocations = 1,
      .AllocationList = &allocation,
   };
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, Evict, &evict);
   return AGX_KMT_SUCCESS(st) ? 0 : agx_d3dkmt_fail(st);
}

int
agx_d3dkmt_open_nt_handle_from_name(struct agx_device *dev,
                                    const void *object_attributes,
                                    uint32_t access, uintptr_t *nt_handle)
{
   if (!dev || !object_attributes || !nt_handle) {
      errno = EINVAL;
      return -1;
   }

   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_kmt_openhandlefromname open = {
      .dwDesiredAccess = access,
      .pObjAttrib = (void *)object_attributes,
   };
   agx_kmt_status resource_status = AGX_D3DKMT_CALL(ws, OpenNtHandleFromName, &open);
   if (!AGX_KMT_SUCCESS(resource_status)) {
      open.hNtHandle = NULL;
      agx_kmt_status sync_status =
         AGX_D3DKMT_CALL(ws, OpenSyncObjectNtHandleFromName, &open);
      if (!AGX_KMT_SUCCESS(sync_status)) {
         /* The API has separate resource and sync-object name thunks but
          * ID3D12Device::OpenSharedHandleByName is type-neutral. Preserve an
          * access failure from either probe instead of hiding it behind the
          * other probe's expected type mismatch/not-found status. */
         const agx_kmt_status access_denied = (agx_kmt_status)0xC0000022u;
         return agx_d3dkmt_fail(resource_status == access_denied
                                   ? resource_status
                                   : sync_status);
      }
   }

   if (!open.hNtHandle) {
      errno = EIO;
      return -1;
   }
   *nt_handle = (uintptr_t)open.hNtHandle;
   return 0;
}

/* -------------------------------------------------------------------------
 * VM_BIND.
 * ------------------------------------------------------------------------- */
static int
d3dkmt_va_update_init(struct agx_d3dkmt_winsys *ws)
{
   if (ws->va_context != 0 && ws->va_fence != 0)
      return 0;
   if (ws->va_context != 0 || ws->va_fence != 0 ||
       ws->policy.usc_size == 0) {
      errno = EINVAL;
      return -1;
   }

   struct agx_kmt_createsyncobj2 fence = {
      .hDevice = ws->device,
      .Info = {
         .Type = AGX_KMT_SYNC_MONITORED_FENCE,
         /* UpdateGpuVirtualAddress completes this fence through the paging
          * scheduler.  No AGX command stream accesses the fence GPU VA. */
         .Flags = AGX_KMT_SYNC_NO_GPU_ACCESS,
         .MonitoredFence = {
            .InitialFenceValue = 0,
            .EngineAffinity = 1,
         },
      },
   };
   d3dkmt_log("va-interlock create-fence begin device=0x%08x",
              (unsigned)ws->device);
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, CreateSynchronizationObject2, &fence);
   d3dkmt_log("va-interlock create-fence end status=0x%08x fence=0x%08x",
              (unsigned)st, (unsigned)fence.hSyncObject);
   if (!AGX_KMT_SUCCESS(st))
      return agx_d3dkmt_fail(st);

   struct agxw_context_private priv = {
      .version = AGXW_CONTEXT_PRIVATE_VERSION,
      .struct_size = sizeof(priv),
      .priority = AGXW_PRIORITY_MEDIUM,
      .flags = AGXW_CONTEXT_PRIVATE_FLAG_VA_INTERLOCK,
      .usc_exec_base = ws->policy.usc_base,
   };
   if (agxw_context_private_validate(&priv, sizeof(priv), ws->policy.usc_base,
                                     ws->policy.usc_size) != 0) {
      struct agx_kmt_destroysyncobj d = {.hSyncObject = fence.hSyncObject};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &d);
      errno = EINVAL;
      return -1;
   }

   /* CreateContextVirtual selects the GPU-VA KMD context. Do not set
    * SynchronizationOnly: UpdateGpuVirtualAddress requires the same real
    * virtual-addressing context shape used by normal render queues. */
   struct agx_kmt_createcontextvirtual create = {
      .hDevice = ws->device,
      .NodeOrdinal = ws->policy.node_ordinal,
      .EngineAffinity = 1,
      .pPrivateDriverData = &priv,
      .PrivateDriverDataSize = sizeof(priv),
   };
   d3dkmt_log("va-interlock create-context begin device=0x%08x",
              (unsigned)ws->device);
   st = AGX_D3DKMT_CALL(ws, CreateContextVirtual, &create);
   d3dkmt_log("va-interlock create-context end status=0x%08x context=0x%08x",
              (unsigned)st, (unsigned)create.hContext);
   if (!AGX_KMT_SUCCESS(st)) {
      struct agx_kmt_destroysyncobj d = {.hSyncObject = fence.hSyncObject};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &d);
      return agx_d3dkmt_fail(st);
   }

   ws->va_context = create.hContext;
   ws->va_fence = fence.hSyncObject;
   ws->va_fence_cpu =
      (const volatile uint64_t *)fence.Info.MonitoredFence
         .FenceValueCPUVirtualAddress;
   ws->va_fence_next = 0;
   return 0;
}

static int
d3dkmt_bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
               uint32_t count)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_bind stack_binds[8];
   struct agx_kmt_updategpuva_op stack_ops[16];
   struct agx_d3dkmt_bind *binds = stack_binds;
   struct agx_kmt_updategpuva_op *uops = stack_ops;
   uint32_t nops = 0;
   int ret = -1;

   if (count == 0)
      return 0;

   if (count > ARRAY_SIZE(stack_binds)) {
      binds = malloc(count * sizeof(*binds));
      if (!binds) {
         errno = ENOMEM;
         return -1;
      }
   }

   /* Resolve dense GEM handles to D3DKMT allocation handles. */
   simple_mtx_lock(&ws->lock);
   for (uint32_t i = 0; i < count; ++i) {
      binds[i] = (struct agx_d3dkmt_bind){
         .flags = ops[i].flags,
         .kmt_handle = ops[i].handle,
         .offset = ops[i].offset,
         .range = ops[i].range,
         .addr = ops[i].addr,
      };
   }
   simple_mtx_unlock(&ws->lock);

   if (agx_d3dkmt_translate_binds(binds, count, &ws->policy, NULL, 0, &nops) !=
       AGX_D3DKMT_OK) {
      errno = EINVAL;
      goto out;
   }

   if (nops > ARRAY_SIZE(stack_ops)) {
      uops = malloc((size_t)nops * sizeof(*uops));
      if (!uops) {
         errno = ENOMEM;
         goto out;
      }
   }

   if (agx_d3dkmt_translate_binds(binds, count, &ws->policy, uops, nops,
                                  &nops) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      goto out;
   }

   /*
    * The one real semantic change in this op. DRM_IOCTL_ASAHI_VM_BIND returns
    * with the binding in effect; D3DKMTUpdateGpuVirtualAddress queues page-
    * table work that completes on a monitored fence. agx_bo_alloc() maps and
    * then immediately writes through the mapping, so the winsys must restore
    * the synchronous contract rather than push asynchrony up into Honeykrisp.
    */
   simple_mtx_lock(&ws->lock);
   if (ws->device_lost || ws->va_context == 0 || ws->va_fence == 0 ||
       ws->va_fence_next == UINT64_MAX) {
      simple_mtx_unlock(&ws->lock);
      errno = ws->device_lost ? EIO : EINVAL;
      goto out;
   }
   uint64_t fence_value = ws->va_fence_next;
   uint64_t completion_value = fence_value + 1;
   struct agx_kmt_updategpuva update = {
      .hDevice = ws->device,
      .hContext = ws->va_context,
      .hFenceObject = ws->va_fence,
      .NumOperations = nops,
      .Operations = uops,
      .FenceValue = fence_value,
      .Flags = 1u, /* D3DKMT_UPDATEGPUVIRTUALADDRESS::DoNotWait */
   };
   d3dkmt_log("update-gpuva begin context=0x%08x fence=0x%08x value=%llu "
              "operations=%u flags=0x%x",
              (unsigned)update.hContext, (unsigned)update.hFenceObject,
              (unsigned long long)update.FenceValue, update.NumOperations,
              update.Flags);
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, UpdateGpuVirtualAddress, &update);
   d3dkmt_log("update-gpuva end status=0x%08x completion=%llu",
              (unsigned)st, (unsigned long long)completion_value);
   if (AGX_KMT_SUCCESS(st))
      ws->va_fence_next = completion_value;
   simple_mtx_unlock(&ws->lock);

   if (!AGX_KMT_SUCCESS(st)) {
      fprintf(stderr, "D3DKMTUpdateGpuVirtualAddress failed\n");
      ret = agx_d3dkmt_fail(st);
      goto out;
   }

   if (ws->va_fence_cpu == NULL || *ws->va_fence_cpu < completion_value) {
      d3dkmt_log("update-gpuva wait begin fence=0x%08x value=%llu",
                 (unsigned)ws->va_fence,
                 (unsigned long long)completion_value);
      ret = agx_d3dkmt_wait_fence(ws, ws->va_fence, ws->va_fence_cpu,
                                  completion_value, AGX_D3DKMT_INFINITE);
      d3dkmt_log("update-gpuva wait end result=%d errno=%d cpu-value=%llu",
                 ret, errno,
                 (unsigned long long)(ws->va_fence_cpu
                                         ? *ws->va_fence_cpu
                                         : 0));
   } else {
      ret = 0;
   }
   if (ret != 0)
      agx_d3dkmt_mark_device_lost(ws, NULL);

out:
   if (ret == 0)
      agx_bo_read_set_changed(dev);
   if (binds != stack_binds)
      free(binds);
   if (uops != stack_ops)
      free(uops);
   return ret;
}

/*
 * Queue-relative sparse binding for D3D12.  Unlike the generic DRM-shaped
 * bo_bind hook, this deliberately does not wait on the CPU.  Microsoft's
 * tiled-resource contract gives UpdateGpuVirtualAddress a rendering context
 * and monitored fence: VidMm waits for FenceValue on the context's dedicated
 * paging context, applies the mappings, then signals FenceValue + 1.  The
 * caller reserves that next point in its queue timeline, so later rendering
 * waits in the scheduler instead of stalling the DDI thread.
 */
static int
d3dkmt_bo_bind_timeline(struct agx_device *dev, uint32_t queue_id,
                        uint32_t syncobj, uint64_t wait_value,
                        struct drm_asahi_gem_bind_op *ops, uint32_t count)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_bind stack_binds[8];
   struct agx_kmt_updategpuva_op stack_ops[16];
   struct agx_d3dkmt_bind *binds = stack_binds;
   struct agx_kmt_updategpuva_op *uops = stack_ops;
   struct agx_d3dkmt_ctx *ctx = NULL;
   uint32_t nops = 0;
   bool context_claimed = false;
   int ret = -1;

   if (!queue_id || !syncobj || wait_value == UINT64_MAX ||
       (count && !ops)) {
      errno = EINVAL;
      return -1;
   }
   if (!count)
      return 0;

   if (count > ARRAY_SIZE(stack_binds)) {
      binds = malloc((size_t)count * sizeof(*binds));
      if (!binds) {
         errno = ENOMEM;
         return -1;
      }
   }

   simple_mtx_lock(&ws->lock);
   for (uint32_t i = 0; i < count; ++i) {
      binds[i] = (struct agx_d3dkmt_bind){
         .flags = ops[i].flags,
         .kmt_handle = ops[i].handle,
         .offset = ops[i].offset,
         .range = ops[i].range,
         .addr = ops[i].addr,
      };
   }
   simple_mtx_unlock(&ws->lock);

   if (agx_d3dkmt_translate_binds(binds, count, &ws->policy, NULL, 0,
                                  &nops) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      goto out;
   }
   if (nops > ARRAY_SIZE(stack_ops)) {
      uops = malloc((size_t)nops * sizeof(*uops));
      if (!uops) {
         errno = ENOMEM;
         goto out;
      }
   }
   if (agx_d3dkmt_translate_binds(binds, count, &ws->policy, uops, nops,
                                  &nops) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      goto out;
   }

   simple_mtx_lock(&ws->lock);
   ctx = ctx_get(ws, queue_id);
   struct agx_d3dkmt_sync *fence =
      agx_d3dkmt_sync_table_get(&ws->syncs, syncobj);
   if (ws->device_lost || !ctx || ctx->submit_failed ||
       ctx->completion_unknown || ctx->submit_refs != 0 || !fence ||
       !fence->kmt) {
      const bool lost = ws->device_lost || (ctx && ctx->submit_failed);
      const bool busy = ctx && ctx->submit_refs != 0;
      simple_mtx_unlock(&ws->lock);
      if (lost)
         errno = EIO;
      else if (busy)
         errno = EBUSY;
      else
         errno = EINVAL;
      goto out;
   }
   ctx->submit_refs = 1;
   ws->active_submits++;
   context_claimed = true;

   struct agx_kmt_updategpuva update = {
      .hDevice = ws->device,
      .hContext = ctx->hContext,
      .hFenceObject = fence->kmt,
      .NumOperations = nops,
      .Operations = uops,
      .FenceValue = wait_value,
      /* DoNotWait must remain clear: VidMm waits for this queue point before
       * changing its page tables, then signals wait_value + 1. */
      .Flags = 0,
   };
   agx_kmt_status status =
      AGX_D3DKMT_CALL(ws, UpdateGpuVirtualAddress, &update);
   if (AGX_KMT_SUCCESS(status)) {
      const uint64_t completion_value = wait_value + 1;
      if (completion_value > fence->last_request)
         fence->last_request = completion_value;
      ret = 0;
   }
   simple_mtx_unlock(&ws->lock);

   if (!AGX_KMT_SUCCESS(status)) {
      ret = agx_d3dkmt_fail(status);
      agx_d3dkmt_mark_device_lost(ws, ctx);
   } else {
      agx_bo_read_set_changed(dev);
   }

out:
   if (context_claimed) {
      simple_mtx_lock(&ws->lock);
      assert(ctx != NULL && ctx->submit_refs == 1 &&
             ws->active_submits > 0);
      ctx->submit_refs = 0;
      ws->active_submits--;
      simple_mtx_unlock(&ws->lock);
   }
   if (uops != stack_ops)
      free(uops);
   if (binds != stack_binds)
      free(binds);
   return ret;
}

/* -------------------------------------------------------------------------
 * GEM_MMAP / GEM_MUNMAP.
 *
 * Linux does GEM_MMAP_OFFSET plus mmap(2); the AGXKMD port invented a
 * GEM_MMAP ioctl. WDDM already has the operation: D3DKMTLock2 returns a CPU
 * mapping of an allocation. It does NOT accept a placement hint, which is why
 * VK_EXT_map_memory_placed stays disabled on Windows (already the case,
 * patch 0007) -- agx_bo_map_placed()'s fixed_addr is unusable here and is
 * asserted away rather than silently ignored.
 * ------------------------------------------------------------------------- */
static void
d3dkmt_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
               UNUSED size_t length)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   assert(bo->_map == NULL && "not double mapped");
   assert(fixed_addr == NULL &&
          "placed maps need VK_EXT_map_memory_placed, disabled on Windows");
   (void)fixed_addr;

   struct agx_kmt_lock2 lock = {
      .hDevice = ws->device,
      .hAllocation = bo->uapi_handle,
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, Lock2, &lock);
   if (!AGX_KMT_SUCCESS(st)) {
      bo->_map = NULL;
      agx_d3dkmt_fail(st);
      fprintf(stderr, "D3DKMTLock2 failed: handle=%u\n", bo->uapi_handle);
      return;
   }

   bo->_map = lock.pData;
   bo->_map_size = bo->size;
   agx_bo_read_set_changed(dev);
}

static void
d3dkmt_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   struct agx_kmt_unlock2 unlock = {
      .hDevice = ws->device,
      .hAllocation = bo->uapi_handle,
   };

   AGX_D3DKMT_CALL(ws, Unlock2, &unlock);
   bo->_map = NULL;
   agx_bo_read_set_changed(dev);
}

static int
d3dkmt_bo_sync_for_device(struct agx_device *dev, struct agx_bo *bo,
                          size_t offset, size_t size)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   if (!bo || bo->dev != dev || !bo->uapi_handle || offset > bo->size ||
       size > bo->size - offset) {
      errno = EINVAL;
      return -1;
   }
   if (!size)
      return 0;

   const struct agx_kmt_invalidatecache invalidate = {
      .hDevice = ws->device,
      .hAllocation = bo->uapi_handle,
      .Offset = offset,
      .Length = size,
   };
   agx_kmt_status status =
      AGX_D3DKMT_CALL(ws, InvalidateCache, &invalidate);
   if (!AGX_KMT_SUCCESS(status)) {
      agx_d3dkmt_mark_device_lost(ws, NULL);
      return agx_d3dkmt_fail(status);
   }
   return 0;
}

/* -------------------------------------------------------------------------
 * GET_PARAMS.
 *
 * KMTQAITYPE_UMDRIVERPRIVATE is the WDDM-native miniport->UMD capability
 * channel (it is what DxgkDdiQueryAdapterInfo(DXGKQAITYPE_UMDRIVERPRIVATE)
 * exists to serve, and what every vendor UMD uses for exactly this). It is
 * queried once at open; this op is a copy out of that cached blob.
 * ------------------------------------------------------------------------- */
static int
d3dkmt_escape(struct agx_d3dkmt_winsys *ws, uint32_t op, void *payload,
              uint32_t payload_size)
{
   /*
    * CONTRACT.md SS8: exactly two escapes exist, both pure device-property
    * reads with no WDDM equivalent -- GET_PARAMS and GET_TIME. Nothing that
    * allocates, binds, submits or synchronises goes through one, because an
    * escape for any of those would rebuild the private-side-channel split
    * this whole port exists to remove.
    */
   const uint32_t total = (uint32_t)sizeof(struct agxw_escape_header) +
                          payload_size;
   uint8_t *blob = malloc(total);
   if (blob == NULL) {
      errno = ENOMEM;
      return -1;
   }

   struct agxw_escape_header hdr = {
      .magic = AGXW_ESCAPE_MAGIC,
      .version = AGXW_ESCAPE_VERSION,
      .op = op,
      .payload_size = payload_size,
   };
   memcpy(blob, &hdr, sizeof(hdr));
   memset(blob + sizeof(hdr), 0, payload_size);

   struct agx_kmt_escape escape = {
      .hAdapter = ws->adapter,
      .pPrivateDriverData = blob,
      .PrivateDriverDataSize = total,
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, Escape, &escape);
   if (!AGX_KMT_SUCCESS(st)) {
      free(blob);
      return agx_d3dkmt_fail(st);
   }

   memcpy(&hdr, blob, sizeof(hdr));
   if (hdr.status != 0) {
      free(blob);
      errno = -hdr.status;
      return -1;
   }

   memcpy(payload, blob + sizeof(hdr), payload_size);
   free(blob);
   return 0;
}

/*
 * GET_PARAMS. One of the two escapes -- the AGX global parameter block has no
 * WDDM equivalent at all, and KMTQAITYPE_UMDRIVERPRIVATE is spoken for by the
 * adapter identity blob (CONTRACT.md SS2.4, SS8).
 *
 * Fetched once at open, because agx_open_device() calls this before anything
 * else and the values never change; this op is a copy out of that cache.
 */
static ssize_t
d3dkmt_get_params(struct agx_device *dev, void *buf, size_t size)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   memset(buf, 0, size);

   if (ws->params == NULL || ws->params_size == 0)
      return -EINVAL;

   memcpy(buf, ws->params, MIN2(size, (size_t)ws->params_size));
   return (ssize_t)ws->params_size;
}

/* -------------------------------------------------------------------------
 * GEM_BIND_OBJECT (timestamp objects).
 *
 * AGX registers a BO with the firmware's timestamp-object table and gets back
 * a firmware-scoped handle that goes into drm_asahi_cmd_render::ts_vtx /
 * ts_frag.
 *
 * CONTRACT.md SS3.5 puts this on AGXW_ALLOC_TIMESTAMP: the KMD does the
 * context-0 bind at ALLOCATION time and returns object_handle in the output
 * private data. That is the right shape, and it is not an escape -- but it
 * does not fit the current call flow, and this is a contract gap to close
 * rather than something the UMD can paper over:
 *
 *   1. struct agxw_alloc_private has no object_handle field. It is
 *      {version, struct_size, size, flags, align_log2, reserved}, so there is
 *      nowhere for SS3.5's promised output to go.
 *   2. hk_query_pool.c:287 calls agx_bind_timestamps() on a BO that
 *      agx_bo_create() has ALREADY created, so the winsys cannot know at
 *      create time that an allocation will be a timestamp buffer. Making it
 *      knowable means a new agx_bo_flags bit plumbed through
 *      hk_create_query_pool -- small, but a Honeykrisp change, not a winsys
 *      one.
 *
 * Until both are settled this op is an identity: the "object handle" IS the
 * allocation handle, which is a value the miniport already has and can map to
 * a firmware slot lazily when it parses ts_vtx/ts_frag out of the command
 * stream. That is still not an escape, and it keeps timestamp queries on the
 * same allocator as everything else.
 * ------------------------------------------------------------------------- */
static int
d3dkmt_bo_bind_object(struct agx_device *dev,
                      struct drm_asahi_gem_bind_object *bind)
{
   (void)dev;

   if (bind->op != DRM_ASAHI_BIND_OBJECT_OP_BIND) {
      errno = EINVAL;
      return -1;
   }

   if (bind->handle == 0) {
      errno = EINVAL;
      return -1;
   }

   bind->object_handle = bind->handle;
   return 0;
}

static int
d3dkmt_bo_unbind_object(struct agx_device *dev, uint32_t object_handle)
{
   (void)dev;
   (void)object_handle;
   return 0;
}

/* -------------------------------------------------------------------------
 * SUBMIT.
 * ------------------------------------------------------------------------- */
static struct agx_bo *
staging_for(struct agx_device *dev, struct agx_d3dkmt_ctx *ctx, uint32_t size)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   unsigned slot = ctx->staging_slot;
   struct agx_bo *bo = ctx->staging[slot];

   if (bo != NULL && bo->size < size) {
      /* Wait for the slot to retire before recycling its memory. */
      if (agx_d3dkmt_wait_fence(ws, ctx->fence, ctx->fence_cpu,
                                ctx->staging_retire[slot],
                                AGX_D3DKMT_INFINITE))
         return NULL;
      agx_bo_unreference(dev, bo);
      bo = ctx->staging[slot] = NULL;
   }

   if (bo == NULL) {
      size_t alloc = ALIGN_POT(MAX2((size_t)size, (size_t)AIL_PAGESIZE),
                               AIL_PAGESIZE);
      bo = agx_bo_create(dev, alloc, 0, AGX_BO_WRITEBACK, "D3DKMT cmdbuf");
      if (bo == NULL)
         return NULL;
      ctx->staging[slot] = bo;
      ctx->staging_retire[slot] = 0;
   } else if (ctx->staging_retire[slot] != 0) {
      if (agx_d3dkmt_wait_fence(ws, ctx->fence, ctx->fence_cpu,
                                ctx->staging_retire[slot],
                                AGX_D3DKMT_INFINITE))
         return NULL;
   }

   return bo;
}

/* Publish every CPU-authored range the command stream may read. A frontend
 * can provide a proven-complete bounded vector; otherwise whole
 * mapped/shareable BOs form a conservative superset independent of shader
 * reflection and cover command/control streams, executables, descriptors,
 * vertex/index/uniform/texture/indirect inputs and cross-API shared storage.
 *
 * That conservative walk is a correctness fallback, not submission work: the
 * live vector changes only when BO visibility or placement changes. Cache its
 * canonical result by generation so stable workloads take one short mutex,
 * one generation comparison and one bounded memcpy rather than walking every
 * Mesa BO on every queue submission. */
static int
d3dkmt_add_conservative_bo_range(
   struct agx_d3dkmt_winsys *ws, struct agx_bo *bo, uint64_t fixed_va,
   struct agxw_submit_read_range *ranges, uint32_t *count)
{
   if (bo == NULL || p_atomic_read(&bo->refcnt) <= 0 || !bo->size ||
       (!bo->_map &&
        !(bo->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))))
      return 0;

   const uint64_t gpu_va = bo->va != NULL ? bo->va->addr : fixed_va;
   if (gpu_va == 0)
      return 0;

   struct agxw_submit_read_range range;
   if (agx_d3dkmt_bo_read_range(
          &ws->policy, (bo->flags & AGX_BO_LOW_VA) != 0,
          gpu_va, bo->size, &range) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      return -1;
   }
   if (agx_d3dkmt_read_range_add(
          ranges, count, AGXW_MAX_READ_RANGES_PER_SUBMIT,
          range.gpu_va, range.length) != AGX_D3DKMT_OK) {
      errno = ENOSPC;
      return -1;
   }
   return 0;
}

static int
d3dkmt_scan_conservative_read_set(
   struct agx_device *dev, struct agxw_submit_read_range *ranges,
   uint32_t *range_count)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   uint32_t count = 0;

   simple_mtx_lock(&dev->bo_map_lock);
   for (uint32_t handle = 0; handle < dev->max_handle; ++handle) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (bo->va != NULL && d3dkmt_add_conservative_bo_range(
                               ws, bo, 0, ranges, &count)) {
         simple_mtx_unlock(&dev->bo_map_lock);
         return -1;
      }
   }

   /* WDDM process-root bootstrap allocations are mapped at renderer ABI
    * constants and intentionally do not own agx_va records: freeing one Mesa
    * wrapper must not release the process-wide reservation. Add their actual
    * fixed GPU addresses explicitly when CPU-visible/shareable, covering the
    * scratch initialization and printf/control storage the generic bo->va
    * walk cannot see. */
   const struct {
      struct agx_bo *bo;
      uint64_t gpu_va;
   } fixed[] = {
      {dev->zero_bo, AGX_ZERO_PAGE_ADDRESS},
      {dev->scratch_bo, AGX_SCRATCH_PAGE_ADDRESS},
      {dev->printf.bo, dev->printf_buffer_address},
      {dev->g17p_render_context_bo, AGX_G17P_RENDER_CONTEXT_BASE},
   };
   for (uint32_t i = 0; i < ARRAY_SIZE(fixed); ++i) {
      /* Allocator-owned BOs were included by the ordinary walk above.  Only
       * process-root wrappers lack agx_va and need their fixed VA supplied. */
      if (fixed[i].bo != NULL && fixed[i].bo->va == NULL &&
          d3dkmt_add_conservative_bo_range(
             ws, fixed[i].bo, fixed[i].gpu_va, ranges, &count)) {
         simple_mtx_unlock(&dev->bo_map_lock);
         return -1;
      }
   }
   simple_mtx_unlock(&dev->bo_map_lock);
   *range_count = count;
   return 0;
}

static int
d3dkmt_get_conservative_read_set(
   struct agx_device *dev, struct agxw_submit_read_range *ranges,
   uint32_t *range_count)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   int status;

   simple_mtx_lock(&ws->read_set_lock);
   const uint64_t current = p_atomic_read(&dev->bo_read_set_generation);
   status = agx_d3dkmt_read_set_cache_copy(
      &ws->read_set_cache, current, ranges,
      AGXW_MAX_READ_RANGES_PER_SUBMIT, range_count);
   if (status == AGX_D3DKMT_OK) {
      simple_mtx_unlock(&ws->read_set_lock);
      return 0;
   }
   if (status != AGX_D3DKMT_ERR_STALE) {
      simple_mtx_unlock(&ws->read_set_lock);
      errno = status == AGX_D3DKMT_ERR_NOSPACE ? ENOSPC : EINVAL;
      return -1;
   }

   /* Retry publication across incidental BO churn. If the process is
    * continuously creating/mapping resources, the final scan is still the
    * original conservative submit path and is safe to consume; it simply is
    * not retained as a cache entry. BOs referenced by this submission remain
    * live for its duration and therefore appear in every scan. */
   for (uint32_t attempt = 0; attempt < 3; ++attempt) {
      const uint64_t before = p_atomic_read(&dev->bo_read_set_generation);
      if (d3dkmt_scan_conservative_read_set(dev, ranges, range_count)) {
         simple_mtx_unlock(&ws->read_set_lock);
         return -1;
      }
      const uint64_t after = p_atomic_read(&dev->bo_read_set_generation);
      status = agx_d3dkmt_read_set_cache_publish(
         &ws->read_set_cache, before, after, ranges, *range_count);
      if (status == AGX_D3DKMT_OK) {
         simple_mtx_unlock(&ws->read_set_lock);
         return 0;
      }
      if (status != AGX_D3DKMT_ERR_STALE) {
         simple_mtx_unlock(&ws->read_set_lock);
         errno = EINVAL;
         return -1;
      }
   }

   simple_mtx_unlock(&ws->read_set_lock);
   return 0;
}

static int
d3dkmt_build_read_set(struct agx_device *dev, struct agx_bo *staging,
                      const struct agx_submit_virt *virt,
                      struct agxw_submit_read_range *ranges,
                      uint32_t *range_count)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   uint32_t count = 0;
   struct agxw_submit_read_range staging_range;

   if (virt != NULL &&
       (virt->flags & ~AGX_SUBMIT_VIRT_VALID_FLAGS) != 0) {
      errno = EINVAL;
      return -1;
   }

   if (virt != NULL &&
       (virt->flags & AGX_SUBMIT_VIRT_READ_SET_COMPLETE) != 0) {
      int exact_ret = agx_d3dkmt_read_set_add_exact(
         ranges, &count, AGXW_MAX_READ_RANGES_PER_SUBMIT,
         virt->read_ranges, virt->read_range_count);
      if (exact_ret != AGX_D3DKMT_OK) {
         errno = exact_ret == AGX_D3DKMT_ERR_NOSPACE ? ENOSPC : EINVAL;
         return -1;
      }
   } else {
      if (d3dkmt_get_conservative_read_set(dev, ranges, &count))
         return -1;
   }

   /* The staging BO is compulsory even when a frontend publishes an exact
    * set. On the conservative path it normally coalesces with the cached
    * vector; adding it again also makes the invariant robust to future
    * staging lifetime changes. */
   if (agx_d3dkmt_bo_read_range(
          &ws->policy, (staging->flags & AGX_BO_LOW_VA) != 0,
          staging->va->addr, staging->size,
          &staging_range) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      return -1;
   }
   int add_ret = agx_d3dkmt_read_range_add(
      ranges, &count, AGXW_MAX_READ_RANGES_PER_SUBMIT,
      staging_range.gpu_va, staging_range.length);
   if (add_ret != AGX_D3DKMT_OK) {
      errno = add_ret == AGX_D3DKMT_ERR_NOSPACE ? ENOSPC : EINVAL;
      return -1;
   }

   *range_count = count;
   return 0;
}

/*
 * DRM_IOCTL_ASAHI_SUBMIT carries in-syncs, the command stream and out-syncs in
 * ONE ioctl. Under the contract that becomes:
 *
 *   1. D3DKMTWaitForSynchronizationObjectFromGpu on the context, for the
 *      in-syncs. NOT the submit private data's in_sync_count, which
 *      CONTRACT.md SS7.3 requires to be 0 in contract version 1 -- AppleAgxGpu
 *      rejects a non-zero one (submit.c:629-630) and the WDDM front end
 *      inherits that gap. The contract notes this is the idiomatic shape
 *      regardless, and stays the recommendation after the gap closes.
 *   2. D3DKMTSubmitCommand, with Commands pointing at an AGXW_DMA_HEADER
 *      followed by the verbatim AGX command stream. No allocation list, no
 *      patch locations: final GPU VAs are already in the stream, exactly as on
 *      Linux (CONTRACT.md SS7.2 item 3).
 *   3. D3DKMTSignalSynchronizationObjectFromGpu, on the same context, for
 *      every out-sync and the staging-retire fence. This queues an ordered
 *      dxgkrnl software signal packet behind the submitted DMA. D3DKMT sync
 *      handles are dxgkrnl handles and MUST NOT appear in KMD private data.
 *
 * A D3DKMT context is a strictly serial queue, so (1)-(3) are equivalent
 * to the atomic ioctl PROVIDED nothing fails in between -- a failure after the
 * wait would leave a dangling dependency on the context. Everything under
 * UMD control (context lookup, staging allocation, sync resolution, blob
 * sizing) is therefore done BEFORE the first D3DKMT call. A thunk failure
 * after sync values are reserved poisons the context fail-closed.
 */
static int
d3dkmt_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
              struct agx_submit_virt *virt)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   const struct drm_asahi_sync *syncs =
      (const struct drm_asahi_sync *)(uintptr_t)submit->syncs;
   agx_kmt_handle waits[AGXW_MAX_SYNCS_PER_SUBMIT];
   uint64_t wait_values[AGXW_MAX_SYNCS_PER_SUBMIT];
   agx_kmt_handle signals[AGXW_MAX_SYNCS_PER_SUBMIT];
   uint64_t signal_values[AGXW_MAX_SYNCS_PER_SUBMIT];
   union {
      struct agxw_submit_private_v4 alignment;
      uint8_t bytes[sizeof(struct agxw_submit_private_v4) +
                    2 * AGXW_MAX_READ_RANGES_PER_SUBMIT *
                       sizeof(struct agxw_submit_read_range) +
                    AGXW_MAX_FENCE_DEPENDENCIES_PER_SUBMIT *
                       sizeof(struct agxw_submit_fence_dependency)];
   } private_storage;
   struct agxw_submit_private *priv =
      (struct agxw_submit_private *)private_storage.bytes;
   uint32_t priv_size = 0;
   int ret = -1;

   const uint32_t n_in = submit->in_sync_count;
   const uint32_t n_out = submit->out_sync_count;
   const uint32_t fence_dependency_count =
      virt ? virt->fence_dependency_count : 0;
   const bool use_fence_dependencies = fence_dependency_count != 0;
   uint32_t private_flags = 0;

   /* +1 for the winsys's own staging-retire fence. */
   if ((uint64_t)n_in + n_out + 1 > AGXW_MAX_SYNCS_PER_SUBMIT ||
       submit->cmdbuf_size > AGXW_MAX_CMDBUF_BYTES ||
       fence_dependency_count > AGXW_MAX_FENCE_DEPENDENCIES_PER_SUBMIT ||
       (fence_dependency_count &&
        (!virt || !virt->fence_dependencies))) {
      errno = EINVAL;
      return -1;
   }

   static_assert(AGX_SUBMIT_WRITE_SET_COMPLETE ==
                    AGX_D3DKMT_SUBMIT_WRITE_SET_COMPLETE,
                 "portable submit-marker mirror");
   if (agx_d3dkmt_submit_private_flags(submit->flags, &private_flags) !=
       AGX_D3DKMT_OK) {
      errno = EINVAL;
      return -1;
   }

   simple_mtx_lock(&ws->lock);
   struct agx_d3dkmt_ctx *ctx = ctx_get(ws, submit->queue_id);
   if (ctx == NULL) {
      simple_mtx_unlock(&ws->lock);
      errno = EINVAL;
      return -1;
   }
   if (ws->device_lost || ctx->submit_failed || ctx->submit_refs != 0) {
      simple_mtx_unlock(&ws->lock);
      errno = (ws->device_lost || ctx->submit_failed) ? EIO : EBUSY;
      return -1;
   }
   ctx->submit_refs = 1;
   ws->active_submits++;
   const unsigned staging_slot = ctx->staging_slot;
   simple_mtx_unlock(&ws->lock);

   /* The DMA buffer: header + verbatim AGX command stream, in a BO this
    * winsys owns and has already mapped to GPU VA. */
   struct agx_bo *staging =
      staging_for(dev, ctx,
                  (uint32_t)sizeof(struct agxw_dma_header) +
                     submit->cmdbuf_size);
   if (staging == NULL) {
      errno = ENOMEM;
      goto out;
   }

   uint8_t *map = agx_bo_map(staging);
   if (map == NULL) {
      errno = ENOMEM;
      goto out;
   }

   struct agxw_dma_header dma = {
      .magic = AGXW_DMA_MAGIC,
      .header_size = sizeof(dma),
      .cmdbuf_size = submit->cmdbuf_size,
   };
   memcpy(map, &dma, sizeof(dma));
   memcpy(map + sizeof(dma), (const void *)(uintptr_t)submit->cmdbuf,
          submit->cmdbuf_size);

   const uint32_t dma_bytes =
      (uint32_t)sizeof(struct agxw_dma_header) + submit->cmdbuf_size;
   if (agx_bo_sync_for_device(staging, 0, dma_bytes))
      goto out;

   const size_t priv_header_size =
      use_fence_dependencies ? sizeof(struct agxw_submit_private_v4)
                             : sizeof(struct agxw_submit_private_v5);
   const size_t priv_capacity =
      priv_header_size +
      (use_fence_dependencies ? 1u : 2u) *
         AGXW_MAX_READ_RANGES_PER_SUBMIT *
         sizeof(struct agxw_submit_read_range) +
      (size_t)fence_dependency_count *
         sizeof(struct agxw_submit_fence_dependency);
   memset(priv, 0, priv_capacity);
   struct agxw_submit_read_range *read_ranges =
      (struct agxw_submit_read_range *)((uint8_t *)priv + priv_header_size);
   uint32_t read_range_count = 0;
   if (d3dkmt_build_read_set(dev, staging, virt, read_ranges,
                             &read_range_count))
      goto out;

   static_assert(AGX_SUBMIT_FENCE_WAIT == AGXW_SUBMIT_FENCE_WAIT,
                 "placement-fence wait flag mirror");
   static_assert(AGX_SUBMIT_FENCE_SIGNAL == AGXW_SUBMIT_FENCE_SIGNAL,
                 "placement-fence signal flag mirror");
   static_assert(AGX_SUBMIT_FENCE_BOTTOM_OF_PIPE ==
                    AGXW_SUBMIT_FENCE_BOTTOM_OF_PIPE,
                 "placement-fence scope flag mirror");
   if (agx_d3dkmt_submit_private_pack(
          private_flags, read_ranges, read_range_count,
          use_fence_dependencies ? virt->fence_dependencies : NULL,
          fence_dependency_count, priv, (uint32_t)priv_capacity,
          &priv_size) != AGX_D3DKMT_OK) {
      errno = EINVAL;
      goto out;
   }

   /* Resolve every syncobj handle before the first D3DKMT call. */
   uint32_t n_sig = 0;
   uint64_t retire;

   simple_mtx_lock(&ws->lock);
   for (uint32_t i = 0; i < n_in; ++i) {
      struct agx_d3dkmt_sync *s =
         agx_d3dkmt_sync_table_get(&ws->syncs, syncs[i].handle);
      if (s == NULL) {
         simple_mtx_unlock(&ws->lock);
         errno = EINVAL;
         goto out;
      }
      waits[i] = s->kmt;
      wait_values[i] = syncs[i].sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ
                          ? syncs[i].timeline_value
                          : agx_d3dkmt_binary_wait_value(s);
   }

   for (uint32_t i = 0; i < n_out; ++i) {
      const struct drm_asahi_sync *src = &syncs[n_in + i];
      struct agx_d3dkmt_sync *e =
         agx_d3dkmt_sync_table_get(&ws->syncs, src->handle);
      if (e == NULL) {
         simple_mtx_unlock(&ws->lock);
         errno = EINVAL;
         goto out;
      }

      uint64_t v = src->sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ
                      ? src->timeline_value
                      : agx_d3dkmt_binary_signal_value(e, NULL);

      signals[n_sig] = e->kmt;
      signal_values[n_sig] = v;
      if (v > e->last_request)
         e->last_request = v;
      n_sig++;
   }

   retire = ++ctx->fence_next;
   signals[n_sig] = ctx->fence;
   signal_values[n_sig] = retire;
   n_sig++;
   simple_mtx_unlock(&ws->lock);

   if (n_in > 0) {
      struct agx_kmt_waitsyncfromgpu wait = {
         .hContext = ctx->hContext,
         .ObjectCount = n_in,
         .ObjectHandleArray = waits,
         .MonitoredFenceValueArray = wait_values,
      };
      agx_kmt_status st = AGX_D3DKMT_CALL(ws, WaitForSynchronizationObjectFromGpu, &wait);
      if (!AGX_KMT_SUCCESS(st)) {
         ret = agx_d3dkmt_fail(st);
         agx_d3dkmt_mark_device_lost(ws, ctx);
         goto out;
      }
   }

   struct agx_kmt_submitcommand cmd = {
      .Commands = staging->va->addr,
      .CommandLength =
         (uint32_t)sizeof(struct agxw_dma_header) + submit->cmdbuf_size,
      .pPrivateDriverData = priv,
      .PrivateDriverDataSize = priv_size,
      .BroadcastContextCount = 1,
      .BroadcastContext = {ctx->hContext},
   };

   agx_kmt_status st = AGX_D3DKMT_CALL(ws, SubmitCommand, &cmd);
   if (!AGX_KMT_SUCCESS(st)) {
      ret = agx_d3dkmt_fail(st);
      agx_d3dkmt_mark_device_lost(ws, ctx);
      goto out;
   }

   struct agx_kmt_signalsyncfromgpu signal = {
      .hContext = ctx->hContext,
      .ObjectCount = n_sig,
      .ObjectHandleArray = signals,
      .MonitoredFenceValueArray = signal_values,
   };
   st = AGX_D3DKMT_CALL(ws, SignalSynchronizationObjectFromGpu, &signal);
   if (!AGX_KMT_SUCCESS(st)) {
      /* The DMA is already in flight but its completion signals were not
       * queued. Poison the context so staging can never be reused unsafely. */
      ret = agx_d3dkmt_fail(st);
      agx_d3dkmt_mark_completion_unknown(ws, ctx);
      goto out;
   }

   simple_mtx_lock(&ws->lock);
   ctx->staging_retire[staging_slot] = retire;
   ctx->staging_slot = (staging_slot + 1) % AGX_D3DKMT_STAGING_RING;
   simple_mtx_unlock(&ws->lock);
   ret = 0;

out:
   simple_mtx_lock(&ws->lock);
   assert(ctx->submit_refs == 1 && ws->active_submits > 0);
   ctx->submit_refs = 0;
   ws->active_submits--;
   simple_mtx_unlock(&ws->lock);
   return ret;
}

/* -------------------------------------------------------------------------
 * simple_ioctl: VM_CREATE, QUEUE_CREATE, QUEUE_DESTROY, GET_TIME.
 * ------------------------------------------------------------------------- */
static int
d3dkmt_reserve(struct agx_d3dkmt_winsys *ws, uint64_t base, uint64_t size)
{
   struct agx_kmt_reservegpuva reserve = {
      .hPagingQueue = ws->paging_queue,
      .BaseAddress = base,
      .MinimumAddress = base,
      .MaximumAddress = base + size,
      .Size = size,
      .ReservationType = AGX_KMT_GPUVA_RESERVE_NO_ACCESS,
   };

   d3dkmt_log("reserve-gpuva begin queue=0x%08x base=0x%llx size=0x%llx",
              (unsigned)ws->paging_queue, (unsigned long long)base,
              (unsigned long long)size);
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, ReserveGpuVirtualAddress, &reserve);
   d3dkmt_log("reserve-gpuva end status=0x%08x address=0x%llx fence=%llu",
              (unsigned)st, (unsigned long long)reserve.VirtualAddress,
              (unsigned long long)reserve.PagingFenceValue);
   if (!AGX_KMT_SUCCESS(st)) {
      fprintf(stderr, "D3DKMTReserveGpuVirtualAddress(0x%llx, 0x%llx) failed\n",
              (unsigned long long)base, (unsigned long long)size);
      return agx_d3dkmt_fail(st);
   }

   /*
    * CONTRACT.md SS4.3 item 2 decides that a non-zero BaseAddress is
    * MANDATORY, not advisory -- but flags it as the one place it wants a
    * hardware datum, because whether VidMm passes BaseAddress through rather
    * than overriding it is unproven. This check is the seam: if VidMm
    * relocates, exactly one call site fails loudly here instead of every GPU
    * address in the driver being silently wrong. Honeykrisp's VA layout is
    * not relocatable -- shader_base and the zero page at 2^32 are fixed ABI,
    * while legacy AGX2 additionally bakes its printf buffer at 2^36 into
    * precompiled shaders -- so a driver that
    * cannot pin its own addresses cannot run at all, and saying so on the
    * first reservation is the cheapest possible discovery.
    */
   if (reserve.VirtualAddress != base) {
      fprintf(stderr,
              "D3DKMTReserveGpuVirtualAddress ignored the base hint "
              "(wanted 0x%llx, got 0x%llx); Honeykrisp's VA layout is fixed\n",
              (unsigned long long)base,
              (unsigned long long)reserve.VirtualAddress);
      errno = ENOTSUP;
      return -1;
   }

   d3dkmt_log("reserve-gpuva wait begin fence=%llu",
              (unsigned long long)reserve.PagingFenceValue);
   int ret = agx_d3dkmt_wait_paging(ws, reserve.PagingFenceValue);
   d3dkmt_log("reserve-gpuva wait end result=%d errno=%d cpu-value=%llu", ret,
              errno,
              (unsigned long long)(ws->paging_fence_cpu
                                      ? *ws->paging_fence_cpu
                                      : 0));
   return ret;
}

static int
d3dkmt_vm_create(struct agx_device *dev, struct drm_asahi_vm_create *vm)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   /*
    * There is no VM object to create: CONTRACT.md SS5.1 makes the AGX VM the
    * D3DKMT device, which agx_d3dkmt_open_device() already made. What DOES
    * have to happen here, and can only happen here, is reserving the user VA
    * window: D3DKMTMapGpuVirtualAddress can only map inside a reservation,
    * and vm_create is the moment Honeykrisp has decided where its heaps go.
    *
    * vm->kernel_start/kernel_end are INPUTS on DRM -- userspace picks where
    * the kernel's slice sits. Under WDDM the miniport owns its own kernel VA
    * (it lives in TTBR1 and is not part of the WDDM VA space at all,
    * CONTRACT.md SS4.2), so they are advisory here and only the window below
    * kernel_start is claimed.
    */
   uint64_t base = dev->params.vm_start;
   uint64_t end = vm->kernel_start;

   if (end <= base) {
      errno = EINVAL;
      return -1;
   }

   /*
    * CONTRACT.md SS4.2: a single 16 KiB kernel/firmware page sits in the
    * MIDDLE of the user window. Reserving across it would ask VidMm for a
    * range that includes an address it has declared internal, so the window
    * is reserved as the one or two sub-ranges that avoid it.
    */
   struct agx_d3dkmt_va_range ranges[2];
   uint32_t n = agx_d3dkmt_va_split_carveout(base, end - base, ranges);
   if (n == 0) {
      errno = EINVAL;
      return -1;
   }

   if (!process_va_open(ws, base, end - base) ||
       !process_va_lock(ws->process_va))
      return -1;

   struct agx_d3dkmt_process_va_shared *shared = ws->process_va->shared;
   if (shared->reservation_ready) {
      if (shared->reservation_count != n ||
          memcmp(shared->reservations, ranges, n * sizeof(ranges[0])) != 0) {
         process_va_unlock(ws->process_va);
         errno = EPROTO;
         return -1;
      }
   } else {
      for (uint32_t i = 0; i < n; ++i) {
         if (d3dkmt_reserve(ws, ranges[i].base, ranges[i].size)) {
            /* Roll back anything already taken so a retry is clean. */
            for (uint32_t j = 0; j < i; ++j) {
               struct agx_kmt_freegpuva f = {
                  .hAdapter = ws->adapter,
                  .BaseAddress = ranges[j].base,
                  .Size = ranges[j].size,
               };
               AGX_D3DKMT_CALL(ws, FreeGpuVirtualAddress, &f);
            }
            process_va_unlock(ws->process_va);
            return -1;
         }
      }

      memcpy(shared->reservations, ranges, n * sizeof(ranges[0]));
      shared->reservation_count = n;
      shared->reservation_ready = 1;
   }

   if (shared->clients == UINT32_MAX) {
      process_va_unlock(ws->process_va);
      errno = EOVERFLOW;
      return -1;
   }
   shared->clients++;
   ws->process_va->attached = true;
   process_va_unlock(ws->process_va);

   /* The process broker owns the reservation.  A device close must not free
    * ranges still used by another UMD copy. */
   ws->va_count = 0;

   /*
    * CONTRACT.md SS3.3: AGX_BO_EXEC is a MAPPING protection, not an
    * allocation flag, and a bind op carries no execute bit -- so the winsys
    * derives it from the address. agx_open_device() has already chosen
    * shader_base and the 4 GiB USC heap above it (agx_device.c:596-612)
    * before it calls VM_CREATE, so this is the first and only place the
    * window is knowable.
    */
   ws->policy.usc_base = dev->shader_base;
   ws->policy.usc_size = 0x100000000ull;

   /* UpdateGpuVirtualAddress is sequenced against a real rendering context
    * and its monitored fence. Establish that interlock before
    * agx_open_device() performs its first zero-page bind. */
   if (d3dkmt_va_update_init(ws) != 0) {
      if (process_va_lock(ws->process_va)) {
         assert(shared->clients > 0);
         shared->clients--;
         ws->process_va->attached = false;
         if (shared->clients == 0) {
            for (uint32_t i = 0; i < shared->reservation_count; ++i) {
               struct agx_kmt_freegpuva f = {
                  .hAdapter = ws->adapter,
                  .BaseAddress = shared->reservations[i].base,
                  .Size = shared->reservations[i].size,
               };
               AGX_D3DKMT_CALL(ws, FreeGpuVirtualAddress, &f);
            }
            shared->reservation_ready = 0;
            shared->reservation_count = 0;
         }
         process_va_unlock(ws->process_va);
      }
      return -1;
   }

   /* One VM per device; the id is never interpreted by the kernel here. */
   vm->vm_id = 1;
   return 0;
}

static bool
d3dkmt_process_va_init(struct agx_device *dev, uint64_t main_base,
                       uint64_t main_size, uint64_t usc_base, uint64_t usc_size)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_process_va *process = ws->process_va;
   if (process == NULL || !process->attached || !process_va_lock(process))
      return false;

   struct agx_d3dkmt_process_va_shared *shared = process->shared;
   if (!shared->layout_ready) {
      shared->main_base = main_base;
      shared->main_size = main_size;
      shared->usc_base = usc_base;
      shared->usc_size = usc_size;
      memset(shared->bitmap, 0,
             (size_t)shared->word_count * sizeof(shared->bitmap[0]));
      shared->layout_ready = 1;
   }

   bool valid = shared->main_base == main_base &&
                shared->main_size == main_size &&
                shared->usc_base == usc_base && shared->usc_size == usc_size;
   process_va_unlock(process);
   if (!valid)
      errno = EPROTO;
   return valid;
}

static uint64_t
d3dkmt_process_va_alloc(struct agx_device *dev, uint64_t size_B,
                        uint64_t align_B, enum agx_va_flags flags,
                        uint64_t fixed_va)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_process_va *process = ws->process_va;
   if (process == NULL || !process->attached || !process_va_lock(process))
      return 0;

   struct agx_d3dkmt_process_va_shared *shared = process->shared;
   uint64_t allowed_base =
      (flags & AGX_VA_USC) ? shared->usc_base : shared->main_base;
   uint64_t allowed_size =
      (flags & AGX_VA_USC) ? shared->usc_size : shared->main_size;
   uint64_t addr = shared->layout_ready
                      ? agx_d3dkmt_va_bitmap_alloc(
                           shared->bitmap, shared->word_count, shared->va_base,
                           shared->va_size, allowed_base, allowed_size, size_B,
                           align_B, fixed_va)
                      : 0;
   process_va_unlock(process);
   if (addr == 0)
      errno = ENOSPC;
   return addr;
}

static void
d3dkmt_process_va_free(struct agx_device *dev, uint64_t addr, uint64_t size_B,
                       enum agx_va_flags flags)
{
   (void)flags;
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   struct agx_d3dkmt_process_va *process = ws->process_va;
   if (process == NULL || !process->attached || !process_va_lock(process))
      return;

   struct agx_d3dkmt_process_va_shared *shared = process->shared;
   bool freed =
      agx_d3dkmt_va_bitmap_free(shared->bitmap, shared->word_count,
                                shared->va_base, shared->va_size, addr, size_B);
   process_va_unlock(process);
   if (!freed)
      fprintf(stderr, "Invalid process GPU VA free at 0x%llx\n",
              (unsigned long long)addr);
}

static int
d3dkmt_queue_create(struct agx_device *dev, struct drm_asahi_queue_create *q)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   int ret = -1;

   simple_mtx_lock(&ws->lock);

   uint32_t slot = 0;
   while (slot < ws->ctx_capacity && ws->ctxs[slot].hContext != 0)
      slot++;

   if (slot == ws->ctx_capacity) {
      /* A realloc would invalidate a ctx pointer held by submit(). */
      if (ws->active_submits != 0) {
         errno = EBUSY;
         goto out;
      }
      uint32_t cap = ws->ctx_capacity ? ws->ctx_capacity * 2 : 4;
      struct agx_d3dkmt_ctx *ctxs =
         realloc(ws->ctxs, (size_t)cap * sizeof(*ctxs));
      if (ctxs == NULL) {
         errno = ENOMEM;
         goto out;
      }
      memset(ctxs + ws->ctx_capacity, 0,
             (size_t)(cap - ws->ctx_capacity) * sizeof(*ctxs));
      ws->ctxs = ctxs;
      ws->ctx_capacity = cap;
   }

   struct agx_d3dkmt_ctx *ctx = &ws->ctxs[slot];
   memset(ctx, 0, sizeof(*ctx));

   /* Per-context completion fence, used only for staging-ring recycling. */
   struct agx_kmt_createsyncobj2 fence = {
      .hDevice = ws->device,
      .Info = {
         .Type = AGX_KMT_SYNC_MONITORED_FENCE,
         .Flags = AGX_KMT_SYNC_NO_GPU_ACCESS,
      },
   };
   agx_kmt_status st = AGX_D3DKMT_CALL(ws, CreateSynchronizationObject2, &fence);
   if (!AGX_KMT_SUCCESS(st)) {
      ret = agx_d3dkmt_fail(st);
      goto out;
   }

   struct agxw_context_private priv = {
      .version = AGXW_CONTEXT_PRIVATE_VERSION,
      .struct_size = sizeof(priv),
      .priority = q->priority,
      .usc_exec_base = q->usc_exec_base,
   };

   /* Pre-validate with the miniport's own validator rather than a private
    * copy of the rules, so a rejection here means the same thing it would
    * mean in the kernel (CONTRACT.md SS5.2). */
   if (agxw_context_private_validate(&priv, sizeof(priv), ws->policy.usc_base,
                                     ws->policy.usc_size) != 0) {
      struct agx_kmt_destroysyncobj d = {.hSyncObject = fence.hSyncObject};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &d);
      errno = EINVAL;
      goto out;
   }

   struct agx_kmt_createcontextvirtual create = {
      .hDevice = ws->device,
      .NodeOrdinal = ws->policy.node_ordinal,
      .EngineAffinity = 1,
      .pPrivateDriverData = &priv,
      .PrivateDriverDataSize = sizeof(priv),
   };

   st = AGX_D3DKMT_CALL(ws, CreateContextVirtual, &create);
   if (!AGX_KMT_SUCCESS(st)) {
      struct agx_kmt_destroysyncobj d = {.hSyncObject = fence.hSyncObject};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &d);
      ret = agx_d3dkmt_fail(st);
      goto out;
   }

   ctx->hContext = create.hContext;
   ctx->fence = fence.hSyncObject;
   ctx->fence_cpu =
      (const volatile uint64_t *)fence.Info.MonitoredFence
         .FenceValueCPUVirtualAddress;

   /* queue_id is a dense slot index, not the D3DKMT context handle: it is
    * stored in drm_asahi_submit::queue_id and looked up on every submit. */
   q->queue_id = slot + 1;
   ret = 0;

out:
   simple_mtx_unlock(&ws->lock);
   return ret;
}

static int
d3dkmt_queue_destroy(struct agx_device *dev, struct drm_asahi_queue_destroy *q)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   simple_mtx_lock(&ws->lock);
   struct agx_d3dkmt_ctx *ctx = ctx_get(ws, q->queue_id);
   if (ctx == NULL) {
      simple_mtx_unlock(&ws->lock);
      errno = EINVAL;
      return -1;
   }
   if (ctx->submit_refs != 0 || ctx->completion_unknown) {
      simple_mtx_unlock(&ws->lock);
      errno = EBUSY;
      return -1;
   }

   struct agx_d3dkmt_ctx local = *ctx;
   memset(ctx, 0, sizeof(*ctx));
   simple_mtx_unlock(&ws->lock);

   struct agx_kmt_destroycontext destroy = {.hContext = local.hContext};
   AGX_D3DKMT_CALL(ws, DestroyContext, &destroy);

   for (unsigned i = 0; i < AGX_D3DKMT_STAGING_RING; ++i) {
      if (local.staging[i] != NULL)
         agx_bo_unreference(dev, local.staging[i]);
   }

   struct agx_kmt_destroysyncobj d = {.hSyncObject = local.fence};
   AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &d);
   return 0;
}

/*
 * GET_TIME. The second and last escape.
 *
 * D3DKMTQueryClockCalibration exists and is the obvious WDDM-native candidate,
 * but it reports an engine tick whose relationship to AGX's
 * command_timestamp_frequency_hz is unspecified, and Honeykrisp converts this
 * value with exactly that frequency (agx_gpu_timestamp_to_ns, agx_device.h).
 * The contract routes it through the escape so the KMD answers from the same
 * timebase source that serves the ioctl today, which is the property that
 * actually matters (CONTRACT.md SS8).
 */
static int
d3dkmt_get_time(struct agx_device *dev, struct drm_asahi_get_time *t)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);

   return d3dkmt_escape(ws, AGXW_ESC_GET_TIME, t, AGXW_GET_TIME_SIZE);
}

static int
d3dkmt_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req)
{
   switch (cmd) {
   case DRM_IOCTL_ASAHI_VM_CREATE:
      return d3dkmt_vm_create(dev, req);
   case DRM_IOCTL_ASAHI_QUEUE_CREATE:
      return d3dkmt_queue_create(dev, req);
   case DRM_IOCTL_ASAHI_QUEUE_DESTROY:
      return d3dkmt_queue_destroy(dev, req);
   case DRM_IOCTL_ASAHI_GET_TIME:
      return d3dkmt_get_time(dev, req);
   default:
      errno = EINVAL;
      return -1;
   }
}

const agx_device_ops_t agx_device_d3dkmt_ops = {
   .bo_alloc = d3dkmt_bo_alloc,
   .bo_bind = d3dkmt_bo_bind,
   .bo_bind_timeline = d3dkmt_bo_bind_timeline,
   .bo_mmap = d3dkmt_bo_mmap,
   .get_params = d3dkmt_get_params,
   .submit = d3dkmt_submit,
   .bo_bind_object = d3dkmt_bo_bind_object,
   .bo_unbind_object = d3dkmt_bo_unbind_object,
   .simple_ioctl = d3dkmt_simple_ioctl,
   .bo_munmap = d3dkmt_bo_munmap,
   .bo_sync_for_device = d3dkmt_bo_sync_for_device,
   .bo_free = d3dkmt_bo_free,
   .va_init = d3dkmt_process_va_init,
   .va_alloc = d3dkmt_process_va_alloc,
   .va_free = d3dkmt_process_va_free,
   .bo_alloc_fixed = d3dkmt_bo_alloc_fixed,
};

/* -------------------------------------------------------------------------
 * Adapter enumeration and device open.
 * ------------------------------------------------------------------------- */
/*
 * Is this D3DKMT adapter ours?
 *
 * CONTRACT.md SS2.3 is explicit that VendorId/DeviceId are NOT reliable here:
 * the adapter is ACPI-enumerated (ACPI\NTAS0023), there is no PCI config
 * space, and whether dxgkrnl even calls DxgkDdiGetVendorAndDevice for a
 * non-PCI adapter is unproven. The identity blob is the only supported
 * discriminator, and getting it wrong means claiming another vendor's GPU.
 */
static bool
query_identity(const struct agx_d3dkmt_fns *fns, agx_kmt_handle adapter,
               struct agx_d3dkmt_policy *policy,
               struct agxw_adapter_identity *out)
{
   struct agxw_adapter_identity blob;

   memset(&blob, 0, sizeof(blob));

   struct agx_kmt_queryadapterinfo query = {
      .hAdapter = adapter,
      .Type = AGX_KMT_QAITYPE_UMDRIVERPRIVATE,
      .pPrivateDriverData = &blob,
      .PrivateDriverDataSize = sizeof(blob),
   };

   agx_kmt_status status = fns->QueryAdapterInfo(&query);
   d3dkmt_log("identity adapter=0x%08x status=0x%08x magic=0x%08x "
              "version=%u size=%u contract=%u params=%u submit-max=%u",
              (unsigned)adapter, (unsigned)status, blob.magic, blob.version,
              blob.struct_size, blob.contract_version, blob.params_blob_size,
              blob.submit_private_max_version);
   if (!AGX_KMT_SUCCESS(status))
      return false;

   return agx_d3dkmt_parse_identity(&blob, sizeof(blob), policy, out) ==
          AGX_D3DKMT_OK;
}

static bool
enumerate_adapters_matching(const struct agxw_adapter_identity *expected,
                            struct agx_kmt_luid *out, uint32_t capacity,
                            uint32_t *count)
{
   struct agx_d3dkmt_fns fns;
   struct agx_kmt_adapterinfo *infos = NULL;
   uint32_t found = 0;

   *count = 0;

   if (!agx_d3dkmt_load(&fns)) {
      d3dkmt_log("enumerate load-thunks failed");
      return false;
   }

   struct agx_kmt_enumadapters2 e = {0};
   agx_kmt_status status = fns.EnumAdapters2(&e);
   d3dkmt_log("enumerate count status=0x%08x adapters=%u", (unsigned)status,
              e.NumAdapters);
   if (!AGX_KMT_SUCCESS(status))
      return false;
   if (e.NumAdapters == 0)
      return true;

   infos = calloc(e.NumAdapters, sizeof(*infos));
   if (infos == NULL)
      return false;

   e.pAdapters = infos;
   status = fns.EnumAdapters2(&e);
   d3dkmt_log("enumerate fill status=0x%08x adapters=%u", (unsigned)status,
              e.NumAdapters);
   if (!AGX_KMT_SUCCESS(status)) {
      free(infos);
      return false;
   }

   for (uint32_t i = 0; i < e.NumAdapters; ++i) {
      struct agx_d3dkmt_policy policy;
      struct agxw_adapter_identity id;

      d3dkmt_log("enumerate candidate=%u adapter=0x%08x luid=%08x:%08x", i,
                 (unsigned)infos[i].hAdapter,
                 (unsigned)infos[i].AdapterLuid.HighPart,
                 (unsigned)infos[i].AdapterLuid.LowPart);

      /* A foreign adapter's UMDRIVERPRIVATE blob fails the magic/version/
       * contract-version check and is skipped. */
      const bool valid =
         query_identity(&fns, infos[i].hAdapter, &policy, &id);
      struct agx_kmt_closeadapter close = {
         .hAdapter = infos[i].hAdapter,
      };
      const agx_kmt_status close_status = fns.CloseAdapter(&close);
      if (!AGX_KMT_SUCCESS(close_status))
         d3dkmt_log("enumerate close adapter=0x%08x status=0x%08x",
                    (unsigned)infos[i].hAdapter, (unsigned)close_status);

      if (valid && (!expected || !memcmp(&id, expected, sizeof(id)))) {
         if (found < capacity)
            out[found] = infos[i].AdapterLuid;
         found++;
      }
   }

   free(infos);
   *count = found;
   d3dkmt_log("enumerate accepted=%u", found);
   return true;
}

bool
agx_d3dkmt_enumerate_adapters(struct agx_kmt_luid *out, uint32_t capacity,
                              uint32_t *count)
{
   if (!count || (capacity && !out))
      return false;
   return enumerate_adapters_matching(NULL, out, capacity, count);
}

bool
agx_d3dkmt_find_adapter_luid(const struct agxw_adapter_identity *identity,
                             struct agx_kmt_luid *luid)
{
   uint32_t count = 0;
   if (!identity || !luid ||
       !enumerate_adapters_matching(identity, luid, 1, &count))
      return false;
   return count == 1;
}

bool
agx_d3dkmt_open_device(struct agx_device *dev, const struct agx_kmt_luid *luid)
{
   struct agx_d3dkmt_winsys *ws = calloc(1, sizeof(*ws));
   if (ws == NULL)
      return false;

   if (!agx_d3dkmt_load(&ws->fns))
      goto fail_alloc;

   ws->policy = agx_d3dkmt_policy_default();
   ws->luid = *luid;

   struct agx_kmt_openadapterfromluid open = {.AdapterLuid = *luid};
   d3dkmt_log("open-adapter begin luid=%08x:%08x", (unsigned)luid->HighPart,
              (unsigned)luid->LowPart);
   agx_kmt_status status = ws->fns.OpenAdapterFromLuid(&open);
   d3dkmt_log("open-adapter end status=0x%08x adapter=0x%08x",
              (unsigned)status, (unsigned)open.hAdapter);
   if (!AGX_KMT_SUCCESS(status))
      goto fail_alloc;
   ws->adapter = open.hAdapter;

   struct agxw_adapter_identity identity;
   if (!query_identity(&ws->fns, ws->adapter, &ws->policy, &identity))
      goto fail_adapter;

   /* GET_PARAMS is an escape (CONTRACT.md SS8) and its payload never
    * changes, so it is fetched once here and agx_open_device()'s
    * ops.get_params() copies out of this. */
   ws->params_size = identity.params_blob_size;
   ws->params = malloc(ws->params_size);
   if (ws->params == NULL)
      goto fail_adapter;
   d3dkmt_log("get-params begin size=%u", ws->params_size);
   int params_status =
      d3dkmt_escape(ws, AGXW_ESC_GET_PARAMS, ws->params, ws->params_size);
   d3dkmt_log("get-params end result=%d errno=%d", params_status, errno);
   if (params_status != 0)
      goto fail_params;

   if (!process_device_open(ws))
      goto fail_params;

   struct agx_kmt_createpagingqueue pq = {.hDevice = ws->device};
   d3dkmt_log("create-paging-queue begin device=0x%08x",
              (unsigned)ws->device);
   status = AGX_D3DKMT_CALL(ws, CreatePagingQueue, &pq);
   d3dkmt_log("create-paging-queue end status=0x%08x queue=0x%08x "
              "fence=0x%08x cpu=%p",
              (unsigned)status, (unsigned)pq.hPagingQueue,
              (unsigned)pq.hSyncObject, pq.FenceValueCPUVirtualAddress);
   if (!AGX_KMT_SUCCESS(status))
      goto fail_process_device;
   ws->paging_queue = pq.hPagingQueue;
   ws->paging_fence = pq.hSyncObject;
   ws->paging_fence_cpu =
      (const volatile uint64_t *)pq.FenceValueCPUVirtualAddress;

   simple_mtx_init(&ws->lock, mtx_plain);
   simple_mtx_init(&ws->read_set_lock, mtx_plain);
   agx_d3dkmt_read_set_cache_init(&ws->read_set_cache);
   agx_d3dkmt_pool_init(&ws->pool);
   agx_d3dkmt_sync_table_init(&ws->syncs);

   dev->hdev = ws;
   dev->fd = -1;
   dev->is_virtio = false;
   dev->write_set_contract = true;
   p_atomic_set(&dev->bo_read_set_generation, 1);
   dev->ops = agx_device_d3dkmt_ops;
   return true; /* caller then runs agx_open_device() */

fail_process_device:
   process_device_release(ws);
   free(ws->params);
   free(ws);
   return false;
fail_params:
   free(ws->params);
fail_adapter:;
   struct agx_kmt_closeadapter ca = {.hAdapter = ws->adapter};
   ws->fns.CloseAdapter(&ca);
fail_alloc:
   free(ws);
   return false;
}

bool
agx_d3dkmt_open_runtime_device(
   struct agx_device *dev,
   const struct agx_d3dkmt_runtime_device *runtime_device)
{
   struct agx_d3dkmt_winsys *ws;
   struct agx_kmt_createpagingqueue paging = {0};
   agx_kmt_status status;

   if (!dev || !runtime_device || !runtime_device->transport.userdata ||
       !agx_d3dkmt_runtime_ops_complete(runtime_device->transport.ops) ||
       runtime_device->params_size != AGXW_PARAMS_GLOBAL_SIZE)
      return false;

   ws = calloc(1, sizeof(*ws));
   if (!ws)
      return false;

   ws->runtime = runtime_device->transport;
   ws->runtime_owned = true;
   ws->policy = agx_d3dkmt_policy_default();
   ws->luid = runtime_device->process_key;

   /* These are source-side sentinels only. Runtime transport methods ignore
    * them and use their callback userdata; all returned allocation/sync/
    * paging handles remain real 32-bit D3DKMT handles. Context HANDLEs are
    * represented by bridge-owned tokens. */
   ws->adapter = 1;
   ws->device = 1;
   ws->params_size = runtime_device->params_size;
   ws->params = malloc(ws->params_size);
   if (!ws->params)
      goto fail;

   if (d3dkmt_escape(ws, AGXW_ESC_GET_PARAMS, ws->params, ws->params_size))
      goto fail;

   status = AGX_D3DKMT_CALL(ws, CreatePagingQueue, &paging);
   if (!AGX_KMT_SUCCESS(status) || !paging.hPagingQueue ||
       !paging.hSyncObject || !paging.FenceValueCPUVirtualAddress) {
      if (AGX_KMT_SUCCESS(status) && paging.hPagingQueue) {
         struct agx_kmt_destroypagingqueue destroy = {
            .hPagingQueue = paging.hPagingQueue,
         };
         (void)AGX_D3DKMT_CALL(ws, DestroyPagingQueue, &destroy);
      }
      goto fail;
   }

   ws->paging_queue = paging.hPagingQueue;
   ws->paging_fence = paging.hSyncObject;
   ws->paging_fence_cpu =
      (const volatile uint64_t *)paging.FenceValueCPUVirtualAddress;
   simple_mtx_init(&ws->lock, mtx_plain);
   simple_mtx_init(&ws->read_set_lock, mtx_plain);
   agx_d3dkmt_read_set_cache_init(&ws->read_set_cache);
   agx_d3dkmt_pool_init(&ws->pool);
   agx_d3dkmt_sync_table_init(&ws->syncs);

   dev->hdev = ws;
   dev->fd = -1;
   dev->is_virtio = false;
   dev->write_set_contract = true;
   p_atomic_set(&dev->bo_read_set_generation, 1);
   dev->ops = agx_device_d3dkmt_ops;
   return true;

fail:
   free(ws->params);
   free(ws);
   return false;
}

static void
process_va_destroy_anchor_record(struct agx_d3dkmt_winsys *ws,
                                 agx_kmt_handle device,
                                 const struct agx_d3dkmt_shared_fixed_bo *fixed)
{
   if (fixed->allocation == 0 && fixed->resource == 0)
      return;

   const agx_kmt_handle allocation = fixed->allocation;
   struct agx_kmt_destroyallocation2 destroy = {
      .hDevice = device,
      .hResource = fixed->resource,
      .phAllocationList = fixed->resource ? NULL : &allocation,
      .AllocationCount = fixed->resource ? 0 : 1,
   };
   agx_kmt_status status = AGX_D3DKMT_CALL(ws, DestroyAllocation2, &destroy);
   if (!AGX_KMT_SUCCESS(status))
      agx_d3dkmt_fail(status);
}

static void
process_va_detach(struct agx_d3dkmt_winsys *ws)
{
   struct agx_d3dkmt_process_va *process = ws->process_va;
   if (process == NULL)
      return;

   if (process->attached && process_va_lock(process)) {
      struct agx_d3dkmt_process_va_shared *shared = process->shared;
      if (shared->clients > 0)
         shared->clients--;

      agx_kmt_handle anchor_device = shared->anchor_device;
      if (shared->clients == 0) {
         for (uint32_t i = 0; i < AGX_D3DKMT_FIXED_BO_COUNT; ++i) {
            if (shared->fixed[i].nt_handle != 0)
               CloseHandle((HANDLE)(uintptr_t)shared->fixed[i].nt_handle);
            process_va_destroy_anchor_record(ws, anchor_device,
                                             &shared->fixed[i]);
            memset(&shared->fixed[i], 0, sizeof(shared->fixed[i]));
         }

         for (uint32_t i = 0; i < shared->reservation_count; ++i) {
            struct agx_kmt_freegpuva free_va = {
               .hAdapter = ws->adapter,
               .BaseAddress = shared->reservations[i].base,
               .Size = shared->reservations[i].size,
            };
            AGX_D3DKMT_CALL(ws, FreeGpuVirtualAddress, &free_va);
         }

         shared->anchor_device = 0;
         shared->reservation_ready = 0;
         shared->reservation_count = 0;
         shared->layout_ready = 0;
         memset(shared->reservations, 0, sizeof(shared->reservations));
         memset(shared->bitmap, 0,
                (size_t)shared->word_count * sizeof(shared->bitmap[0]));

      }

      process->attached = false;
      process_va_unlock(process);
   }

   process_va_close_mapping(process);
   ws->process_va = NULL;
}

void
agx_d3dkmt_close_device(struct agx_device *dev)
{
   struct agx_d3dkmt_winsys *ws = agx_d3dkmt_ws(dev);
   if (ws == NULL)
      return;

   /* Adapter teardown is the forced-retirement boundary for a transfer
    * context or render context whose completion became unknowable. */
   if (ws->xfer_ctx != 0) {
      struct agx_kmt_destroycontext dc = {.hContext = ws->xfer_ctx};
      AGX_D3DKMT_CALL(ws, DestroyContext, &dc);
      ws->xfer_ctx = 0;
   }

   for (uint32_t i = 0; i < ws->ctx_capacity; ++i) {
      if (ws->ctxs[i].hContext == 0)
         continue;

      struct agx_kmt_destroycontext dc = {.hContext = ws->ctxs[i].hContext};
      AGX_D3DKMT_CALL(ws, DestroyContext, &dc);

      for (unsigned slot = 0; slot < AGX_D3DKMT_STAGING_RING; ++slot) {
         if (ws->ctxs[i].staging[slot] != NULL)
            agx_bo_unreference(dev, ws->ctxs[i].staging[slot]);
      }

      struct agx_kmt_destroysyncobj ds = {.hSyncObject = ws->ctxs[i].fence};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &ds);
   }
   free(ws->ctxs);

   if (ws->va_context != 0) {
      struct agx_kmt_destroycontext dc = {.hContext = ws->va_context};
      AGX_D3DKMT_CALL(ws, DestroyContext, &dc);
      ws->va_context = 0;
   }
   if (ws->va_fence != 0) {
      struct agx_kmt_destroysyncobj ds = {.hSyncObject = ws->va_fence};
      AGX_D3DKMT_CALL(ws, DestroySynchronizationObject, &ds);
      ws->va_fence = 0;
   }

   for (uint32_t i = 0; i < ws->va_count; ++i) {
      struct agx_kmt_freegpuva fva = {
         .hAdapter = ws->adapter,
         .BaseAddress = ws->va[i].base,
         .Size = ws->va[i].size,
      };
      AGX_D3DKMT_CALL(ws, FreeGpuVirtualAddress, &fva);
   }

   struct agx_kmt_destroypagingqueue dpq = {.hPagingQueue = ws->paging_queue};
   AGX_D3DKMT_CALL(ws, DestroyPagingQueue, &dpq);

   process_va_detach(ws);
   process_device_release(ws);

   agx_d3dkmt_pool_finish(&ws->pool);
   agx_d3dkmt_sync_table_finish(&ws->syncs);
   simple_mtx_destroy(&ws->read_set_lock);
   simple_mtx_destroy(&ws->lock);
   free(ws->params);
   free(ws);

   dev->hdev = NULL;
}

struct agx_kmt_luid
agx_d3dkmt_device_luid(const struct agx_device *dev)
{
   const struct agx_d3dkmt_winsys *ws =
      (const struct agx_d3dkmt_winsys *)dev->hdev;

   return ws ? ws->luid : (struct agx_kmt_luid){0, 0};
}

agx_kmt_handle
agx_d3dkmt_device_paging_queue(const struct agx_device *dev)
{
   const struct agx_d3dkmt_winsys *ws;

   if (!dev || dev->ops.submit != agx_device_d3dkmt_ops.submit)
      return 0;
   ws = (const struct agx_d3dkmt_winsys *)dev->hdev;
   return ws ? ws->paging_queue : 0;
}

bool
agx_d3dkmt_device_wait_paging(struct agx_device *dev, uint64_t value)
{
   struct agx_d3dkmt_winsys *ws;

   if (!dev || dev->ops.submit != agx_device_d3dkmt_ops.submit)
      return false;
   ws = agx_d3dkmt_ws(dev);
   return ws && ws->paging_queue && agx_d3dkmt_wait_paging(ws, value) == 0;
}
