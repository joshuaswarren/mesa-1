/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * libdrm API subset used by agx_device.c / u_sync_provider.c, implemented in
 * d12_wine_drm.c over the Wine unixlib. Same names and return conventions as
 * libdrm (0 or -errno for the syncobj/prime helpers, 0/-1 for drmIoctl).
 */
#ifndef D12_WINE_XF86DRM_H
#define D12_WINE_XF86DRM_H

#include <stdint.h>
#include <stddef.h>
#include "drm-uapi/drm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _drmVersion {
   int version_major;
   int version_minor;
   int version_patchlevel;
   int name_len;
   char *name;
   int date_len;
   char *date;
   int desc_len;
   char *desc;
} drmVersion, *drmVersionPtr;

/* drm.h maps DRM_CLOEXEC/DRM_RDWR to the open(2) flags; mingw has no
 * O_CLOEXEC, so give it the Linux value (asm-generic fcntl.h). */
#ifndef O_CLOEXEC
#define O_CLOEXEC 02000000
#endif

#define DRM_NODE_PRIMARY 0
#define DRM_NODE_CONTROL 1
#define DRM_NODE_RENDER  2

int drmIoctl(int fd, unsigned request, void *arg);
drmVersionPtr drmGetVersion(int fd);
void drmFreeVersion(drmVersionPtr v);
int drmGetCap(int fd, uint64_t capability, uint64_t *value);

int drmPrimeHandleToFD(int fd, uint32_t handle, uint32_t flags, int *prime_fd);
int drmPrimeFDToHandle(int fd, int prime_fd, uint32_t *handle);

int drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle);
int drmSyncobjDestroy(int fd, uint32_t handle);
int drmSyncobjHandleToFD(int fd, uint32_t handle, int *obj_fd);
int drmSyncobjFDToHandle(int fd, int obj_fd, uint32_t *handle);
int drmSyncobjImportSyncFile(int fd, uint32_t handle, int sync_file_fd);
int drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd);
int drmSyncobjWait(int fd, uint32_t *handles, unsigned num_handles,
                   int64_t timeout_nsec, unsigned flags,
                   uint32_t *first_signaled);
int drmSyncobjReset(int fd, const uint32_t *handles, uint32_t handle_count);
int drmSyncobjSignal(int fd, const uint32_t *handles, uint32_t handle_count);
int drmSyncobjTimelineSignal(int fd, const uint32_t *handles,
                             uint64_t *points, uint32_t handle_count);
int drmSyncobjTimelineWait(int fd, uint32_t *handles, uint64_t *points,
                           unsigned num_handles, int64_t timeout_nsec,
                           unsigned flags, uint32_t *first_signaled);
int drmSyncobjQuery2(int fd, uint32_t *handles, uint64_t *points,
                     uint32_t handle_count, uint32_t flags);
int drmSyncobjTransfer(int fd, uint32_t dst_handle, uint64_t dst_point,
                       uint32_t src_handle, uint64_t src_point,
                       uint32_t flags);

#ifdef __cplusplus
}
#endif
#endif
