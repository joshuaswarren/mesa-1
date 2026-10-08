/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>

/* Little-endian field access for command-stream and descriptor words. */

static inline uint16_t
agx_load_le16(const uint8_t *src)
{
   return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

static inline uint32_t
agx_load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static inline uint64_t
agx_load_le64(const uint8_t *src)
{
   return (uint64_t)agx_load_le32(src) |
          ((uint64_t)agx_load_le32(src + 4) << 32);
}

static inline void
agx_store_le16(uint8_t *dst, uint16_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
}

static inline void
agx_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static inline void
agx_store_le64(uint8_t *dst, uint64_t value)
{
   agx_store_le32(dst, (uint32_t)value);
   agx_store_le32(dst + 4, (uint32_t)(value >> 32));
}
