/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Minimal <linux/types.h> for UAPI headers compiled into the Wine PE build.
 * Identical typedefs to drm.h's non-Linux branch (legal C11 redefinitions).
 */
#ifndef D12_WINE_LINUX_TYPES_H
#define D12_WINE_LINUX_TYPES_H
#include <stdint.h>
#include <stddef.h>
typedef int8_t   __s8;
typedef uint8_t  __u8;
typedef int16_t  __s16;
typedef uint16_t __u16;
typedef int32_t  __s32;
typedef uint32_t __u32;
typedef int64_t  __s64;
typedef uint64_t __u64;
typedef size_t   __kernel_size_t;
#endif
