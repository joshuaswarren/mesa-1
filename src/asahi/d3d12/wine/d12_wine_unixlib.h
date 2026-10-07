/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Shared ABI between the ARM64EC PE d3d12.dll and its Wine unixlib d3d12.so.
 * Both sides are 64-bit little-endian in one process, so pointers are passed
 * as-is. Only fixed-width types below; never `long` (LLP64 vs LP64).
 */

#ifndef D12_WINE_UNIXLIB_H
#define D12_WINE_UNIXLIB_H

#include <stdint.h>

enum d12_wine_call {
   D12_WINE_CALL_OPEN = 0,
   D12_WINE_CALL_CLOSE,
   D12_WINE_CALL_IOCTL,
   D12_WINE_CALL_MMAP,
   D12_WINE_CALL_MUNMAP,
   D12_WINE_CALL_SYNCOBJ_WAIT,
   D12_WINE_CALL_IS_RENDER_NODE,
   D12_WINE_CALL_CLOCK,
   D12_WINE_CALL_DUPFD_CLOEXEC,
   D12_WINE_CALL_DMABUF_SIZE,
   D12_WINE_CALL_RESOURCE_FD,
   D12_WINE_CALL_COUNT
};

/* Linux mmap/mprotect constants the PE side must speak. */
#define D12_WINE_PROT_READ   0x1
#define D12_WINE_PROT_WRITE  0x2
#define D12_WINE_MAP_SHARED  0x01
#define D12_WINE_MAP_FIXED   0x10

struct d12_wine_open_args {
   int32_t fd;       /* out: render node fd, or -1 */
   int32_t err;      /* out: errno when fd < 0 */
   char name[32];    /* out: DRM driver name */
};

struct d12_wine_close_args {
   int32_t fd;
   int32_t pad;
};

struct d12_wine_ioctl_args {
   int32_t fd;
   uint32_t request;
   uint64_t arg;     /* pointer to the ioctl struct, same process */
   int32_t ret;      /* out: ioctl() return (0 or -1) */
   int32_t err;      /* out: errno */
};

struct d12_wine_mmap_args {
   uint64_t addr;    /* in: fixed address or 0; out: mapping or 0 */
   uint64_t length;
   int32_t prot;
   int32_t flags;
   int32_t fd;
   int32_t err;      /* out */
   uint64_t offset;
};

struct d12_wine_munmap_args {
   uint64_t addr;
   uint64_t length;
   int32_t ret;
   int32_t err;
};

/* Timeline (points != 0) or binary syncobj wait with a RELATIVE timeout; the
 * unix side re-bases it on CLOCK_MONOTONIC. rel_timeout_ns < 0 waits forever. */
struct d12_wine_syncobj_wait_args {
   int32_t fd;
   uint32_t count;
   uint64_t handles; /* const uint32_t * */
   uint64_t points;  /* const uint64_t * or 0 */
   int64_t rel_timeout_ns;
   uint32_t flags;
   uint32_t first_signaled; /* out */
   int32_t ret;      /* out: 0 or -errno */
   int32_t pad;
};

struct d12_wine_is_render_node_args {
   int32_t fd;
   int32_t result;   /* out: 1 if a character device render node named asahi */
};

struct d12_wine_clock_args {
   int64_t monotonic_ns; /* out */
};

struct d12_wine_dupfd_args {
   int32_t fd;
   int32_t new_fd;
   int32_t err;
   int32_t pad;
};

struct d12_wine_dmabuf_size_args {
   int32_t fd;
   int32_t err;
   int64_t size;
};

#define D12_WINE_CLOCK_ABI_QUERY INT64_MIN
#define D12_WINE_CLOCK_ABI_VERSION (-INT64_C(0x12120003))
/* The size-preserving CLOCK handshake detects a mismatched old call table. */
struct d12_wine_resource_fd_args {
   uint64_t handle;
   int32_t fd;
   int32_t err;
};

#endif
