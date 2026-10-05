/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Process-memory and kernel entry points for the Wine PE build. Replaces
 * util/os_mman.h for agx_device.c (AGX_WINE_UNIXLIB) and exposes the unixlib
 * plumbing to the D3D12 platform backend.
 */
#ifndef AGX_WINE_UNIXLIB_H
#define AGX_WINE_UNIXLIB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_READ   0x1
#define PROT_WRITE  0x2
#define MAP_SHARED  0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED   0x10
#define MAP_FAILED  ((void *)(intptr_t)-1)

void *os_mmap(void *addr, size_t length, int prot, int flags, int fd,
              int64_t offset);
int os_munmap(void *addr, size_t length);

/* Loads the unixlib (once). False when d3d12.so is missing or this is not a
 * Wine builtin module; every DRM call then fails with ENOSYS. */
bool d12_wine_unixlib_init(void);
/* Open the asahi render node through the unixlib; -1 on failure. */
int d12_wine_open_render_node(void);
void d12_wine_close_fd(int fd);
bool d12_wine_fd_is_asahi_render_node(int fd);

#ifdef __cplusplus
}
#endif
#endif
