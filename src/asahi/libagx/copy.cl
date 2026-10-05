/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "present_blt.h"
#include "resolve.h"

KERNEL(32)
libagx_fill(global uint32_t *address, uint32_t value)
{
   assert((((uintptr_t)address) & 0x3) == 0);

   address[cl_global_id.x] = value;
}

KERNEL(32)
libagx_copy_uint4(global uint4 *dest, global uint4 *src)
{
   assert((((uintptr_t)dest) & 0xf) == 0);
   assert((((uintptr_t)src) & 0xf) == 0);

   dest[cl_global_id.x] = src[cl_global_id.x];
}

KERNEL(16)
libagx_copy_uchar(global uint8_t *dest, global uint8_t *src)
{
   dest[cl_global_id.x] = src[cl_global_id.x];
}

/*
 * Copy a rectangle from an AIL GPU-tiled image into a linear buffer. This is
 * the device-side form of layout/tiling.cc::memcpy_small: tiles are in raster
 * order, while elements inside each power-of-two tile use Morton order.
 *
 * Keeping this raw (rather than formatted texture loads) preserves exact bits
 * for depth/stencil and typeless copies, as D3D12 CopyTextureRegion requires.
 */
static uint
libagx_space_bits(uint x)
{
   uint accum = 0;
   for (uint i = 0; i < 14; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

static uint
libagx_space_bits_7(uint x)
{
   uint accum = 0;
   for (uint i = 0; i < 7; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

KERNEL(32)
libagx_detile(global uint8_t *dest, global uint8_t *src,
              uint dest_pitch_B, uint width_el, uint height_el,
              uint src_x_el, uint src_y_el, uint stride_el,
              uint tile_width_el, uint tile_height_el, uint blocksize_B)
{
   uint id = cl_global_id.x;
   if (id >= width_el * height_el)
      return;

   uint x = src_x_el + (id % width_el);
   uint y = src_y_el + (id / width_el);
   uint tiles_per_row = (stride_el + tile_width_el - 1) / tile_width_el;
   uint tile_idx = (y / tile_height_el) * tiles_per_row +
                   (x / tile_width_el);
   uint tile_area_el = tile_width_el * tile_height_el;
   uint morton_el = libagx_space_bits(x & (tile_width_el - 1)) |
                    (libagx_space_bits(y & (tile_height_el - 1)) << 1);
   uint src_offset_B = (tile_idx * tile_area_el + morton_el) * blocksize_B;
   uint dst_offset_B = (id / width_el) * dest_pitch_B +
                       (id % width_el) * blocksize_B;

   for (uint b = 0; b < blocksize_B; ++b)
      dest[dst_offset_B + b] = src[src_offset_B + b];
}

/*
 * Raw D3D12 texture copies operate on format elements, not interpreted
 * colours.  An element is a pixel for ordinary formats and a whole block for
 * BC formats.  Keeping both sides in element coordinates also implements the
 * specified BC<->RGBA32 copy compatibility without format conversion: one
 * 16-byte BC block is copied to or from one 16-byte RGBA32 pixel.
 *
 * CPU command recording validates every rectangle against the logical image
 * or placed footprint before this kernel is dispatched.  The coordinate
 * guard is a second line of defence against an over-wide launch.
 */
static uint
libagx_copy_image_offset(uint x, uint y, uint pitch_B, uint stride_el,
                         uint tile_width_el, uint tile_height_el,
                         uint blocksize_B, uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tiles_per_row = (stride_el + tile_width_el - 1) / tile_width_el;
   uint tile_idx = (y / tile_height_el) * tiles_per_row +
                   (x / tile_width_el);
   uint tile_area_el = tile_width_el * tile_height_el;
   uint morton_el = libagx_space_bits(x & (tile_width_el - 1)) |
                    (libagx_space_bits(y & (tile_height_el - 1)) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

/* Reduce power-of-two tile geometry on the CPU. Small tiles can use the
 * seven-bit encoder, but legacy AIL mip tails may use larger square tiles.
 * Keep their high coordinate bits just as the generic helper does. */
static uint
libagx_copy_image_offset_pow2(uint x, uint y, uint pitch_B,
                              uint tiles_per_row, uint tile_width_log2,
                              uint tile_height_log2, uint blocksize_B,
                              uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tile_width_el = 1u << tile_width_log2;
   uint tile_height_el = 1u << tile_height_log2;
   uint tile_idx = (y >> tile_height_log2) * tiles_per_row +
                   (x >> tile_width_log2);
   uint tile_area_el = 1u << (tile_width_log2 + tile_height_log2);
   uint tile_x = x & (tile_width_el - 1);
   uint tile_y = y & (tile_height_el - 1);
   uint morton_el = tile_width_log2 <= 7 && tile_height_log2 <= 7
      ? libagx_space_bits_7(tile_x) | (libagx_space_bits_7(tile_y) << 1)
      : libagx_space_bits(tile_x) | (libagx_space_bits(tile_y) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

KERNEL(32)
libagx_copy_image(global uint8_t *dest, global uint8_t *src,
                  uint width_el, uint height_el,
                  uint dest_x_el, uint dest_y_el,
                  uint src_x_el, uint src_y_el,
                  uint dest_pitch_B, uint src_pitch_B,
                  uint dest_tiles_per_row, uint src_tiles_per_row,
                  uint dest_tile_width_log2, uint dest_tile_height_log2,
                  uint src_tile_width_log2, uint src_tile_height_log2,
                  uint blocksize_B, uint dest_tiled, uint src_tiled)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= width_el || y >= height_el)
      return;

   uint dest_offset_B = libagx_copy_image_offset_pow2(
      dest_x_el + x, dest_y_el + y, dest_pitch_B, dest_tiles_per_row,
      dest_tile_width_log2, dest_tile_height_log2, blocksize_B, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset_pow2(
      src_x_el + x, src_y_el + y, src_pitch_B, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, blocksize_B, src_tiled);

   for (uint b = 0; b < blocksize_B; ++b)
      dest[dest_offset_B + b] = src[src_offset_B + b];
}

/* Round an unsigned four-sample average like G17P's linear sampler followed
 * by a normalized colour write. Exact output ties move toward code 128. This
 * avoids the still-unqualified G17P float-to-integer convert encoding. */
static uint8_t
libagx_average4_unorm8(uint a, uint b, uint c, uint d)
{
   uint sum = a + b + c + d;
   uint quotient = sum >> 2;
   uint remainder = sum & 3u;
   return (uint8_t)(quotient +
                    (remainder > 2u ||
                     (remainder == 2u && quotient < 128u)));
}

/*
 * Exact full-level 2x downsample for ordinary four-channel UNORM8 images.
 * RGBA and BGRA are both channel-order agnostic here: each stored byte is
 * filtered independently and written back in the same order.
 */
KERNEL(32)
libagx_mipmap_unorm8x4(global uint8_t *dest, global uint8_t *src,
                       uint dest_width_px, uint dest_height_px,
                       uint src_scale_x, uint src_scale_y,
                       uint dest_tiles_per_row, uint src_tiles_per_row,
                       uint dest_tile_width_log2, uint dest_tile_height_log2,
                       uint src_tile_width_log2, uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint dest_offset = libagx_copy_image_offset_pow2(
      x, y, 0, dest_tiles_per_row, dest_tile_width_log2,
      dest_tile_height_log2, 4, 1);
   uint src_x = x * src_scale_x;
   uint src_y = y * src_scale_y;
   uint src00 = libagx_copy_image_offset_pow2(
      src_x, src_y, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src10 = libagx_copy_image_offset_pow2(
      src_x + src_scale_x - 1, src_y, 0, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, 4, 1);
   uint src01 = libagx_copy_image_offset_pow2(
      src_x, src_y + src_scale_y - 1, 0, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, 4, 1);
   uint src11 = libagx_copy_image_offset_pow2(
      src_x + src_scale_x - 1, src_y + src_scale_y - 1, 0,
      src_tiles_per_row, src_tile_width_log2, src_tile_height_log2, 4, 1);

   for (uint channel = 0; channel < 4; ++channel) {
      dest[dest_offset + channel] = libagx_average4_unorm8(
         src[src00 + channel], src[src10 + channel],
         src[src01 + channel], src[src11 + channel]);
   }
}

/*
 * Exact odd-size full-level downsample for four-channel UNORM8 images.  The
 * CPU builds one packed Q8 coordinate entry per destination X and Y, avoiding
 * every per-pixel divide and all 64-bit arithmetic here.  G17P's sampler
 * rounds the four-tap result half-up to four fractional colour bits; its
 * normalized write rounds exact ties toward the midpoint code 128.
 */
KERNEL(32)
libagx_mipmap_unorm8x4_odd(global uint8_t *dest, global uint8_t *src,
                           global uint *axis,
                           uint dest_width_px, uint dest_height_px,
                           uint dest_tiles_per_row, uint src_tiles_per_row,
                           uint dest_tile_width_log2,
                           uint dest_tile_height_log2,
                           uint src_tile_width_log2,
                           uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint x_entry = axis[x];
   uint y_entry = axis[dest_width_px + y];
   uint src_x0 = x_entry & 0xffffu;
   uint src_y0 = y_entry & 0xffffu;
   uint wx = (x_entry >> 16) & 0xffu;
   uint wy = (y_entry >> 16) & 0xffu;
   uint src_x1 = src_x0 + (wx != 0u);
   uint src_y1 = src_y0 + (wy != 0u);

   uint dest_offset = libagx_copy_image_offset_pow2(
      x, y, 0, dest_tiles_per_row, dest_tile_width_log2,
      dest_tile_height_log2, 4, 1);
   uint src00 = libagx_copy_image_offset_pow2(
      src_x0, src_y0, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src10 = libagx_copy_image_offset_pow2(
      src_x1, src_y0, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src01 = libagx_copy_image_offset_pow2(
      src_x0, src_y1, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src11 = libagx_copy_image_offset_pow2(
      src_x1, src_y1, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint ix = 256u - wx;
   uint iy = 256u - wy;

   for (uint channel = 0; channel < 4; ++channel) {
      uint value = src[src00 + channel] * ix * iy +
                   src[src10 + channel] * wx * iy +
                   src[src01 + channel] * ix * wy +
                   src[src11 + channel] * wx * wy;
      uint fixed4 = (value + 2048u) >> 12;
      uint out = fixed4 >> 4;
      uint remainder = fixed4 & 15u;
      out += remainder > 8u || (remainder == 8u && out < 128u);
      dest[dest_offset + channel] = (uint8_t)out;
   }
}

/*
 * Resolve an unorm8x4 multisample surface into a single-sample surface.
 * AIL tiles one logical pixel as a block containing every sample. Averaging
 * bytes is format-order agnostic for the four 8-bit UNORM channel formats
 * this entry point accepts on the CPU side.
 */
KERNEL(32)
libagx_resolve_unorm8x4(global uint8_t *dest, global uint8_t *src,
                        uint width_px, uint height_px,
                        uint dest_x_px, uint dest_y_px,
                        uint src_x_px, uint src_y_px,
                        uint dest_pitch_B, uint dest_stride_el,
                        uint src_stride_el,
                        uint dest_tile_width_el, uint dest_tile_height_el,
                        uint src_tile_width_el, uint src_tile_height_el,
                        uint sample_count, uint dest_tiled, uint src_tiled,
                        uint resolve_mode)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);

   for (uint b = 0; b < 4; ++b) {
      uint sum = 0;
      uint extrema = resolve_mode == 1 ? 255 : 0;
      for (uint s = 0; s < sample_count; ++s) {
         uint src_offset_B = libagx_copy_image_offset(
            src_x_px + x, src_y_px + y, 0, src_stride_el,
            src_tile_width_el, src_tile_height_el,
            4 * sample_count, src_tiled);
         uint value = src[src_offset_B + s * 4 + b];
         sum += value;
         extrema = resolve_mode == 1 ? min(extrema, value)
                                      : max(extrema, value);
      }

      dest[dest_offset_B + b] = resolve_mode
                                   ? extrema
                                   : (sum + sample_count / 2) / sample_count;
   }
}

static float
libagx_srgb_to_linear(float value)
{
   return value <= 0.04045f
             ? value / 12.92f
             : native_powr((value + 0.055f) / 1.055f, 2.4f);
}

static float
libagx_linear_to_srgb(float value)
{
   return value <= 0.0031308f
             ? value * 12.92f
             : 1.055f * native_powr(value, 1.0f / 2.4f) - 0.055f;
}

static uint
libagx_present_load_u16(global uint8_t *address)
{
   return (uint)address[0] | ((uint)address[1] << 8);
}

static uint
libagx_present_load_u32(global uint8_t *address)
{
   return (uint)address[0] | ((uint)address[1] << 8) |
          ((uint)address[2] << 16) | ((uint)address[3] << 24);
}

static void
libagx_present_store_u16(global uint8_t *address, uint value)
{
   address[0] = (uint8_t)value;
   address[1] = (uint8_t)(value >> 8);
}

static void
libagx_present_store_u32(global uint8_t *address, uint value)
{
   address[0] = (uint8_t)value;
   address[1] = (uint8_t)(value >> 8);
   address[2] = (uint8_t)(value >> 16);
   address[3] = (uint8_t)(value >> 24);
}

/* Explicit IEEE-754 binary16 conversion keeps the raw-memory kernel
 * independent of optional OpenCL half-storage extensions. */
static float
libagx_present_half_to_float(uint value)
{
   uint sign = (value & 0x8000u) << 16;
   uint exponent = (value >> 10) & 0x1fu;
   uint mantissa = value & 0x3ffu;
   uint bits;

   if (exponent == 0) {
      if (mantissa == 0) {
         bits = sign;
      } else {
         int unbiased = -14;
         while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --unbiased;
         }
         mantissa &= 0x3ffu;
         bits = sign | (uint)(unbiased + 127) << 23 | mantissa << 13;
      }
   } else if (exponent == 0x1fu) {
      bits = sign | 0x7f800000u | mantissa << 13;
      if (mantissa)
         bits |= 1u;
   } else {
      bits = sign | (exponent + 112u) << 23 | mantissa << 13;
   }
   return as_float(bits);
}

static uint
libagx_present_float_to_half(float value)
{
   uint bits = as_uint(value);
   uint sign = (bits >> 16) & 0x8000u;
   uint exponent = (bits >> 23) & 0xffu;
   uint mantissa = bits & 0x7fffffu;

   if (exponent == 0xffu) {
      if (!mantissa)
         return sign | 0x7c00u;
      return sign | 0x7c00u | max(1u, mantissa >> 13);
   }
   if (exponent > 142u)
      return sign | 0x7c00u;
   if (exponent < 113u) {
      if (exponent < 103u)
         return sign;
      mantissa |= 0x800000u;
      uint shift = 125u - exponent;
      uint rounded =
         (mantissa + ((1u << (shift - 1u)) - 1u) +
          ((mantissa >> shift) & 1u)) >> shift;
      return sign | rounded;
   }

   uint half_exponent = exponent - 112u;
   mantissa += 0xfffu + ((mantissa >> 13) & 1u);
   if (mantissa & 0x800000u) {
      mantissa = 0;
      ++half_exponent;
      if (half_exponent >= 0x1fu)
         return sign | 0x7c00u;
   }
   return sign | half_exponent << 10 | mantissa >> 13;
}

static float4
libagx_present_load_pixel(global uint8_t *address, uint format)
{
   if (format == LIBAGX_PRESENT_FORMAT_RGBA8 ||
       format == LIBAGX_PRESENT_FORMAT_RGBA8_SRGB) {
      return (float4)((float)address[0], (float)address[1],
                      (float)address[2], (float)address[3]) /
             255.0f;
   }
   if (format == LIBAGX_PRESENT_FORMAT_BGRA8 ||
       format == LIBAGX_PRESENT_FORMAT_BGRA8_SRGB ||
       format == LIBAGX_PRESENT_FORMAT_BGRX8) {
      return (float4)((float)address[2], (float)address[1],
                      (float)address[0],
                      format == LIBAGX_PRESENT_FORMAT_BGRX8
                         ? 255.0f
                         : (float)address[3]) /
             255.0f;
   }

   uint packed = libagx_present_load_u16(address);
   if (format == LIBAGX_PRESENT_FORMAT_BGR565) {
      return (float4)((float)((packed >> 11) & 0x1fu) / 31.0f,
                      (float)((packed >> 5) & 0x3fu) / 63.0f,
                      (float)(packed & 0x1fu) / 31.0f, 1.0f);
   }
   if (format == LIBAGX_PRESENT_FORMAT_BGRA5551) {
      return (float4)((float)((packed >> 10) & 0x1fu) / 31.0f,
                      (float)((packed >> 5) & 0x1fu) / 31.0f,
                      (float)(packed & 0x1fu) / 31.0f,
                      (float)((packed >> 15) & 1u));
   }
   if (format == LIBAGX_PRESENT_FORMAT_RGB10A2) {
      packed = libagx_present_load_u32(address);
      return (float4)((float)(packed & 0x3ffu) / 1023.0f,
                      (float)((packed >> 10) & 0x3ffu) / 1023.0f,
                      (float)((packed >> 20) & 0x3ffu) / 1023.0f,
                      (float)((packed >> 30) & 3u) / 3.0f);
   }

   return (float4)(
      libagx_present_half_to_float(libagx_present_load_u16(address + 0)),
      libagx_present_half_to_float(libagx_present_load_u16(address + 2)),
      libagx_present_half_to_float(libagx_present_load_u16(address + 4)),
      libagx_present_half_to_float(libagx_present_load_u16(address + 6)));
}

static uint
libagx_present_quantize(float value, uint maximum)
{
   return convert_uint_sat_rte(clamp(value, 0.0f, 1.0f) *
                               (float)maximum);
}

static void
libagx_present_store_pixel(global uint8_t *address, uint format,
                           float4 value)
{
   if (format == LIBAGX_PRESENT_FORMAT_RGBA8 ||
       format == LIBAGX_PRESENT_FORMAT_RGBA8_SRGB) {
      address[0] = (uint8_t)libagx_present_quantize(value.x, 255u);
      address[1] = (uint8_t)libagx_present_quantize(value.y, 255u);
      address[2] = (uint8_t)libagx_present_quantize(value.z, 255u);
      address[3] = (uint8_t)libagx_present_quantize(value.w, 255u);
      return;
   }
   if (format == LIBAGX_PRESENT_FORMAT_BGRA8 ||
       format == LIBAGX_PRESENT_FORMAT_BGRA8_SRGB ||
       format == LIBAGX_PRESENT_FORMAT_BGRX8) {
      address[0] = (uint8_t)libagx_present_quantize(value.z, 255u);
      address[1] = (uint8_t)libagx_present_quantize(value.y, 255u);
      address[2] = (uint8_t)libagx_present_quantize(value.x, 255u);
      address[3] = format == LIBAGX_PRESENT_FORMAT_BGRX8
                      ? 255
                      : (uint8_t)libagx_present_quantize(value.w, 255u);
      return;
   }
   if (format == LIBAGX_PRESENT_FORMAT_BGR565) {
      uint packed = libagx_present_quantize(value.z, 31u) |
                    libagx_present_quantize(value.y, 63u) << 5 |
                    libagx_present_quantize(value.x, 31u) << 11;
      libagx_present_store_u16(address, packed);
      return;
   }
   if (format == LIBAGX_PRESENT_FORMAT_BGRA5551) {
      uint packed = libagx_present_quantize(value.z, 31u) |
                    libagx_present_quantize(value.y, 31u) << 5 |
                    libagx_present_quantize(value.x, 31u) << 10 |
                    libagx_present_quantize(value.w, 1u) << 15;
      libagx_present_store_u16(address, packed);
      return;
   }
   if (format == LIBAGX_PRESENT_FORMAT_RGB10A2) {
      uint packed = libagx_present_quantize(value.x, 1023u) |
                    libagx_present_quantize(value.y, 1023u) << 10 |
                    libagx_present_quantize(value.z, 1023u) << 20 |
                    libagx_present_quantize(value.w, 3u) << 30;
      libagx_present_store_u32(address, packed);
      return;
   }

   libagx_present_store_u16(address + 0,
                            libagx_present_float_to_half(value.x));
   libagx_present_store_u16(address + 2,
                            libagx_present_float_to_half(value.y));
   libagx_present_store_u16(address + 4,
                            libagx_present_float_to_half(value.z));
   libagx_present_store_u16(address + 6,
                            libagx_present_float_to_half(value.w));
}

static uint
libagx_resolve_channel_count(uint format)
{
   if (format == LIBAGX_RESOLVE_FORMAT_UNORM8_1 ||
       format == LIBAGX_RESOLVE_FORMAT_SNORM8_1 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT8_1 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT8_1 ||
       format == LIBAGX_RESOLVE_FORMAT_UNORM16_1 ||
       format == LIBAGX_RESOLVE_FORMAT_SNORM16_1 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT16_1 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT16_1 ||
       format == LIBAGX_RESOLVE_FORMAT_FLOAT16_1 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT32_1 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT32_1 ||
       format == LIBAGX_RESOLVE_FORMAT_FLOAT32_1)
      return 1;

   if (format == LIBAGX_RESOLVE_FORMAT_UNORM8_2 ||
       format == LIBAGX_RESOLVE_FORMAT_SNORM8_2 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT8_2 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT8_2 ||
       format == LIBAGX_RESOLVE_FORMAT_UNORM16_2 ||
       format == LIBAGX_RESOLVE_FORMAT_SNORM16_2 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT16_2 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT16_2 ||
       format == LIBAGX_RESOLVE_FORMAT_FLOAT16_2 ||
       format == LIBAGX_RESOLVE_FORMAT_UINT32_2 ||
       format == LIBAGX_RESOLVE_FORMAT_SINT32_2 ||
       format == LIBAGX_RESOLVE_FORMAT_FLOAT32_2)
      return 2;

   return 4;
}

static uint
libagx_resolve_scalar_size(uint format)
{
   if (format >= LIBAGX_RESOLVE_FORMAT_UNORM8_1 &&
       format <= LIBAGX_RESOLVE_FORMAT_SINT8_4)
      return 1;
   if (format >= LIBAGX_RESOLVE_FORMAT_UNORM16_1 &&
       format <= LIBAGX_RESOLVE_FORMAT_FLOAT16_4)
      return 2;
   return 4;
}

static uint
libagx_resolve_load_scalar(global uint8_t *address, uint size_B)
{
   if (size_B == 1)
      return address[0];
   if (size_B == 2)
      return libagx_present_load_u16(address);
   return libagx_present_load_u32(address);
}

static void
libagx_resolve_store_scalar(global uint8_t *address, uint size_B,
                            uint value)
{
   if (size_B == 1)
      address[0] = (uint8_t)value;
   else if (size_B == 2)
      libagx_present_store_u16(address, value);
   else
      libagx_present_store_u32(address, value);
}

static int
libagx_resolve_sign_extend(uint value, uint bits)
{
   uint sign = 1u << (bits - 1u);
   return (int)((value ^ sign) - sign);
}

/* R11G11B10 uses the positive-only five-bit-exponent float encoding. */
static float
libagx_resolve_ufloat_to_float(uint value, uint mantissa_bits)
{
   uint exponent = value >> mantissa_bits;
   uint mantissa = value & ((1u << mantissa_bits) - 1u);
   uint bits;

   if (exponent == 0) {
      if (mantissa == 0) {
         bits = 0;
      } else {
         int unbiased = -14;
         uint hidden = 1u << mantissa_bits;
         while ((mantissa & hidden) == 0) {
            mantissa <<= 1;
            --unbiased;
         }
         mantissa &= hidden - 1u;
         bits = (uint)(unbiased + 127) << 23 |
                mantissa << (23u - mantissa_bits);
      }
   } else if (exponent == 0x1fu) {
      bits = 0x7f800000u | mantissa << (23u - mantissa_bits);
      if (mantissa)
         bits |= 1u;
   } else {
      bits = (exponent + 112u) << 23 |
             mantissa << (23u - mantissa_bits);
   }
   return as_float(bits);
}

static uint
libagx_resolve_float_to_ufloat(float value, uint mantissa_bits)
{
   uint bits = as_uint(value);
   uint exponent = (bits >> 23) & 0xffu;
   uint mantissa = bits & 0x7fffffu;
   uint mantissa_shift = 23u - mantissa_bits;

   if (bits & 0x80000000u)
      return 0;
   if (exponent == 0xffu) {
      if (!mantissa)
         return 0x1fu << mantissa_bits;
      return 0x1fu << mantissa_bits |
             max(1u, mantissa >> mantissa_shift);
   }
   if (exponent > 142u)
      return 0x1fu << mantissa_bits;
   if (exponent < 113u) {
      if (exponent < 113u - mantissa_bits)
         return 0;
      mantissa |= 0x800000u;
      uint shift = 135u - mantissa_bits - exponent;
      return (mantissa + ((1u << (shift - 1u)) - 1u) +
              ((mantissa >> shift) & 1u)) >> shift;
   }

   uint packed_exponent = exponent - 112u;
   mantissa += (1u << (mantissa_shift - 1u)) - 1u +
               ((mantissa >> mantissa_shift) & 1u);
   if (mantissa & 0x800000u) {
      mantissa = 0;
      ++packed_exponent;
      if (packed_exponent >= 0x1fu)
         return 0x1fu << mantissa_bits;
   }
   return packed_exponent << mantissa_bits |
          mantissa >> mantissa_shift;
}

static uint
libagx_resolve_present_format(uint format)
{
   if (format == LIBAGX_RESOLVE_FORMAT_RGBA8_SRGB)
      return LIBAGX_PRESENT_FORMAT_RGBA8_SRGB;
   if (format == LIBAGX_RESOLVE_FORMAT_BGRA8_UNORM)
      return LIBAGX_PRESENT_FORMAT_BGRA8;
   if (format == LIBAGX_RESOLVE_FORMAT_BGRA8_SRGB)
      return LIBAGX_PRESENT_FORMAT_BGRA8;
   if (format == LIBAGX_RESOLVE_FORMAT_BGRX8_UNORM)
      return LIBAGX_PRESENT_FORMAT_BGRX8;
   if (format == LIBAGX_RESOLVE_FORMAT_BGR565_UNORM)
      return LIBAGX_PRESENT_FORMAT_BGR565;
   if (format == LIBAGX_RESOLVE_FORMAT_BGRA5551_UNORM)
      return LIBAGX_PRESENT_FORMAT_BGRA5551;
   if (format == LIBAGX_RESOLVE_FORMAT_RGB10A2_UNORM)
      return LIBAGX_PRESENT_FORMAT_RGB10A2;
   return LIBAGX_PRESENT_FORMAT_INVALID;
}

static float4
libagx_resolve_load_float(global uint8_t *address, uint format)
{
   uint present_format = libagx_resolve_present_format(format);
   if (present_format) {
      float4 value = libagx_present_load_pixel(address, present_format);
      if (format == LIBAGX_RESOLVE_FORMAT_RGBA8_SRGB ||
          format == LIBAGX_RESOLVE_FORMAT_BGRA8_SRGB) {
         value.x = libagx_srgb_to_linear(value.x);
         value.y = libagx_srgb_to_linear(value.y);
         value.z = libagx_srgb_to_linear(value.z);
      }
      return value;
   }

   if (format == LIBAGX_RESOLVE_FORMAT_R11G11B10_FLOAT) {
      uint packed = libagx_present_load_u32(address);
      return (float4)(
         libagx_resolve_ufloat_to_float(packed & 0x7ffu, 6),
         libagx_resolve_ufloat_to_float((packed >> 11) & 0x7ffu, 6),
         libagx_resolve_ufloat_to_float((packed >> 22) & 0x3ffu, 5), 1.0f);
   }

   uint channels = libagx_resolve_channel_count(format);
   uint scalar_size_B = libagx_resolve_scalar_size(format);
   float4 result = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
   for (uint c = 0; c < channels; ++c) {
      uint raw = libagx_resolve_load_scalar(address + c * scalar_size_B,
                                            scalar_size_B);
      float value;
      if (format >= LIBAGX_RESOLVE_FORMAT_UNORM8_1 &&
          format <= LIBAGX_RESOLVE_FORMAT_UNORM8_4) {
         value = (float)raw / 255.0f;
      } else if (format >= LIBAGX_RESOLVE_FORMAT_SNORM8_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_SNORM8_4) {
         value = max(-1.0f, (float)libagx_resolve_sign_extend(raw, 8) /
                             127.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_UNORM16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_UNORM16_4) {
         value = (float)raw / 65535.0f;
      } else if (format >= LIBAGX_RESOLVE_FORMAT_SNORM16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_SNORM16_4) {
         value = max(-1.0f, (float)libagx_resolve_sign_extend(raw, 16) /
                             32767.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_FLOAT16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_FLOAT16_4) {
         value = libagx_present_half_to_float(raw);
      } else {
         value = as_float(raw);
      }
      result[c] = value;
   }
   return result;
}

static void
libagx_resolve_store_float(global uint8_t *address, uint format,
                           float4 value)
{
   uint present_format = libagx_resolve_present_format(format);
   if (present_format) {
      if (format == LIBAGX_RESOLVE_FORMAT_RGBA8_SRGB ||
          format == LIBAGX_RESOLVE_FORMAT_BGRA8_SRGB) {
         value.x = libagx_linear_to_srgb(clamp(value.x, 0.0f, 1.0f));
         value.y = libagx_linear_to_srgb(clamp(value.y, 0.0f, 1.0f));
         value.z = libagx_linear_to_srgb(clamp(value.z, 0.0f, 1.0f));
      }
      libagx_present_store_pixel(address, present_format, value);
      return;
   }

   if (format == LIBAGX_RESOLVE_FORMAT_R11G11B10_FLOAT) {
      uint packed = libagx_resolve_float_to_ufloat(value.x, 6) |
                    libagx_resolve_float_to_ufloat(value.y, 6) << 11 |
                    libagx_resolve_float_to_ufloat(value.z, 5) << 22;
      libagx_present_store_u32(address, packed);
      return;
   }

   uint channels = libagx_resolve_channel_count(format);
   uint scalar_size_B = libagx_resolve_scalar_size(format);
   for (uint c = 0; c < channels; ++c) {
      uint raw;
      if (format >= LIBAGX_RESOLVE_FORMAT_UNORM8_1 &&
          format <= LIBAGX_RESOLVE_FORMAT_UNORM8_4) {
         raw = convert_uint_sat_rte(clamp(value[c], 0.0f, 1.0f) * 255.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_SNORM8_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_SNORM8_4) {
         raw = (uint)convert_int_rte(clamp(value[c], -1.0f, 1.0f) * 127.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_UNORM16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_UNORM16_4) {
         raw = convert_uint_sat_rte(clamp(value[c], 0.0f, 1.0f) * 65535.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_SNORM16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_SNORM16_4) {
         raw = (uint)convert_int_rte(clamp(value[c], -1.0f, 1.0f) *
                                     32767.0f);
      } else if (format >= LIBAGX_RESOLVE_FORMAT_FLOAT16_1 &&
                 format <= LIBAGX_RESOLVE_FORMAT_FLOAT16_4) {
         raw = libagx_present_float_to_half(value[c]);
      } else {
         raw = as_uint(value[c]);
      }
      libagx_resolve_store_scalar(address + c * scalar_size_B,
                                  scalar_size_B, raw);
   }
}

static uint4
libagx_resolve_load_uint(global uint8_t *address, uint format)
{
   if (format == LIBAGX_RESOLVE_FORMAT_RGB10A2_UINT) {
      uint packed = libagx_present_load_u32(address);
      return (uint4)(packed & 0x3ffu, (packed >> 10) & 0x3ffu,
                     (packed >> 20) & 0x3ffu, packed >> 30);
   }

   uint channels = libagx_resolve_channel_count(format);
   uint scalar_size_B = libagx_resolve_scalar_size(format);
   uint4 result = (uint4)(0u);
   for (uint c = 0; c < channels; ++c)
      result[c] = libagx_resolve_load_scalar(address + c * scalar_size_B,
                                             scalar_size_B);
   return result;
}

static void
libagx_resolve_store_uint(global uint8_t *address, uint format, uint4 value)
{
   if (format == LIBAGX_RESOLVE_FORMAT_RGB10A2_UINT) {
      libagx_present_store_u32(address, (value.x & 0x3ffu) |
                               (value.y & 0x3ffu) << 10 |
                               (value.z & 0x3ffu) << 20 |
                               (value.w & 3u) << 30);
      return;
   }

   uint channels = libagx_resolve_channel_count(format);
   uint scalar_size_B = libagx_resolve_scalar_size(format);
   for (uint c = 0; c < channels; ++c)
      libagx_resolve_store_scalar(address + c * scalar_size_B,
                                  scalar_size_B, value[c]);
}

static int4
libagx_resolve_load_sint(global uint8_t *address, uint format)
{
   uint channels = libagx_resolve_channel_count(format);
   uint scalar_size_B = libagx_resolve_scalar_size(format);
   uint scalar_bits = scalar_size_B * 8u;
   int4 result = (int4)(0);
   for (uint c = 0; c < channels; ++c) {
      uint raw = libagx_resolve_load_scalar(address + c * scalar_size_B,
                                            scalar_size_B);
      result[c] = libagx_resolve_sign_extend(raw, scalar_bits);
   }
   return result;
}

static void
libagx_resolve_store_sint(global uint8_t *address, uint format, int4 value)
{
   libagx_resolve_store_uint(address, format, as_uint4(value));
}

/* Typed D3D12 MIN/MAX/AVERAGE resolve for every native AGX render-target
 * scalar/vector format.  The CPU table rejects integer AVERAGE before this
 * shader is selected.  One dispatch covers linear or AIL-tiled destinations
 * and the interleaved per-pixel multisample layout without transient views. */
KERNEL(32)
libagx_resolve_typed(global uint8_t *dest, global uint8_t *src,
                     uint width_px, uint height_px,
                     uint dest_x_px, uint dest_y_px,
                     uint src_x_px, uint src_y_px,
                     uint dest_pitch_B, uint dest_stride_el,
                     uint src_stride_el,
                     uint dest_tile_width_el, uint dest_tile_height_el,
                     uint src_tile_width_el, uint src_tile_height_el,
                     uint sample_count, uint dest_tiled, uint src_tiled,
                     uint pixel_size_B, uint format, uint resolve_mode,
                     uint value_class)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, pixel_size_B, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      pixel_size_B * sample_count, src_tiled);
   uint choose_max = resolve_mode == LIBAGX_RESOLVE_MODE_MAX;

   if (value_class == 1) {
      uint4 result = libagx_resolve_load_uint(src + src_offset_B, format);
      for (uint sample = 1; sample < sample_count; ++sample) {
         uint4 next = libagx_resolve_load_uint(
            src + src_offset_B + sample * pixel_size_B, format);
         result = choose_max ? max(result, next) : min(result, next);
      }
      libagx_resolve_store_uint(dest + dest_offset_B, format, result);
   } else if (value_class == 2) {
      int4 result = libagx_resolve_load_sint(src + src_offset_B, format);
      for (uint sample = 1; sample < sample_count; ++sample) {
         int4 next = libagx_resolve_load_sint(
            src + src_offset_B + sample * pixel_size_B, format);
         result = choose_max ? max(result, next) : min(result, next);
      }
      libagx_resolve_store_sint(dest + dest_offset_B, format, result);
   } else {
      float4 result = libagx_resolve_load_float(src + src_offset_B, format);
      if (resolve_mode == LIBAGX_RESOLVE_MODE_AVERAGE) {
         for (uint sample = 1; sample < sample_count; ++sample)
            result += libagx_resolve_load_float(
               src + src_offset_B + sample * pixel_size_B, format);
         result /= (float)sample_count;
      } else {
         for (uint sample = 1; sample < sample_count; ++sample) {
            float4 next = libagx_resolve_load_float(
               src + src_offset_B + sample * pixel_size_B, format);
            result = choose_max ? fmax(result, next) : fmin(result, next);
         }
      }
      libagx_resolve_store_float(dest + dest_offset_B, format, result);
   }
}

static float4
libagx_present_sample(global uint8_t *src, uint x, uint y,
                      uint src_pitch_B, uint src_stride_el,
                      uint src_tile_width_el, uint src_tile_height_el,
                      uint src_pixel_size_B, uint sample_count,
                      uint src_tiled, uint src_format, uint decode_gamma)
{
   uint offset = libagx_copy_image_offset(
      x, y, src_pitch_B, src_stride_el, src_tile_width_el,
      src_tile_height_el, src_pixel_size_B * sample_count, src_tiled);
   float4 sum = (float4)(0.0f);
   for (uint sample = 0; sample < sample_count; ++sample) {
      float4 value = libagx_present_load_pixel(
         src + offset + sample * src_pixel_size_B, src_format);
      if (decode_gamma) {
         value.x = libagx_srgb_to_linear(clamp(value.x, 0.0f, 1.0f));
         value.y = libagx_srgb_to_linear(clamp(value.y, 0.0f, 1.0f));
         value.z = libagx_srgb_to_linear(clamp(value.z, 0.0f, 1.0f));
      }
      sum += value;
   }
   return sum / (float)sample_count;
}

/*
 * DXGI presentation blit. Resolve, bilinear scale, counter-clockwise
 * rotation and display-format conversion happen in one dispatch. This is
 * intentionally raw-memory based so it works for both linear and AIL tiled
 * allocations without building transient texture/sampler descriptors.
 */
KERNEL(32)
libagx_present_blt(global uint8_t *dest, global uint8_t *src,
                   uint dest_width_px, uint dest_height_px,
                   uint dest_x_px, uint dest_y_px,
                   uint src_left_px, uint src_top_px,
                   uint src_right_px, uint src_bottom_px,
                   uint dest_pitch_B, uint src_pitch_B,
                   uint dest_stride_el, uint src_stride_el,
                   uint dest_tile_width_el, uint dest_tile_height_el,
                   uint src_tile_width_el, uint src_tile_height_el,
                   uint dest_pixel_size_B, uint src_pixel_size_B,
                   uint source_sample_count, uint dest_tiled,
                   uint src_tiled, uint dest_format, uint src_format,
                   uint rotation)
{
   uint id = cl_global_id.x;
   if (id >= dest_width_px * dest_height_px)
      return;

   uint x = id % dest_width_px;
   uint y = id / dest_width_px;
   float u = ((float)x + 0.5f) / (float)dest_width_px;
   float v = ((float)y + 0.5f) / (float)dest_height_px;
   float source_u = u;
   float source_v = v;

   /* Rotate source contents counter-clockwise, per BltDXGI. These are the
    * inverse maps from destination space back into the source rectangle. */
   if (rotation == LIBAGX_PRESENT_ROTATION_CCW_90) {
      source_u = 1.0f - v;
      source_v = u;
   } else if (rotation == LIBAGX_PRESENT_ROTATION_CCW_180) {
      source_u = 1.0f - u;
      source_v = 1.0f - v;
   } else if (rotation == LIBAGX_PRESENT_ROTATION_CCW_270) {
      source_u = v;
      source_v = 1.0f - u;
   }

   float source_x = (float)src_left_px +
                    source_u * (float)(src_right_px - src_left_px) - 0.5f;
   float source_y = (float)src_top_px +
                    source_v * (float)(src_bottom_px - src_top_px) - 0.5f;
   source_x = clamp(source_x, (float)src_left_px,
                    (float)(src_right_px - 1u));
   source_y = clamp(source_y, (float)src_top_px,
                    (float)(src_bottom_px - 1u));
   uint x0 = (uint)floor(source_x);
   uint y0 = (uint)floor(source_y);
   uint x1 = min(x0 + 1u, src_right_px - 1u);
   uint y1 = min(y0 + 1u, src_bottom_px - 1u);
   float x_weight = source_x - (float)x0;
   float y_weight = source_y - (float)y0;

   uint decode_gamma = LIBAGX_PRESENT_FORMAT_IS_SRGB(src_format);
   uint encode_gamma = LIBAGX_PRESENT_FORMAT_IS_SRGB(dest_format);

   float4 c00 = libagx_present_sample(
      src, x0, y0, src_pitch_B, src_stride_el, src_tile_width_el,
      src_tile_height_el, src_pixel_size_B, source_sample_count, src_tiled,
      src_format, decode_gamma);
   float4 c10 = libagx_present_sample(
      src, x1, y0, src_pitch_B, src_stride_el, src_tile_width_el,
      src_tile_height_el, src_pixel_size_B, source_sample_count, src_tiled,
      src_format, decode_gamma);
   float4 c01 = libagx_present_sample(
      src, x0, y1, src_pitch_B, src_stride_el, src_tile_width_el,
      src_tile_height_el, src_pixel_size_B, source_sample_count, src_tiled,
      src_format, decode_gamma);
   float4 c11 = libagx_present_sample(
      src, x1, y1, src_pitch_B, src_stride_el, src_tile_width_el,
      src_tile_height_el, src_pixel_size_B, source_sample_count, src_tiled,
      src_format, decode_gamma);
   float4 top = c00 + (c10 - c00) * x_weight;
   float4 bottom = c01 + (c11 - c01) * x_weight;
   float4 value = top + (bottom - top) * y_weight;

   if (encode_gamma) {
      value.x = libagx_linear_to_srgb(clamp(value.x, 0.0f, 1.0f));
      value.y = libagx_linear_to_srgb(clamp(value.y, 0.0f, 1.0f));
      value.z = libagx_linear_to_srgb(clamp(value.z, 0.0f, 1.0f));
   }

   uint dest_offset = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, dest_pixel_size_B,
      dest_tiled);
   libagx_present_store_pixel(dest + dest_offset, dest_format, value);
}

/* Strongly-typed RGBA8 resolves convert the source storage encoding to
 * linear before averaging and encode according to the resolve format. */
KERNEL(32)
libagx_resolve_unorm8x4_typed(global uint8_t *dest, global uint8_t *src,
                              uint width_px, uint height_px,
                              uint dest_x_px, uint dest_y_px,
                              uint src_x_px, uint src_y_px,
                              uint dest_pitch_B, uint dest_stride_el,
                              uint src_stride_el,
                              uint dest_tile_width_el,
                              uint dest_tile_height_el,
                              uint src_tile_width_el,
                              uint src_tile_height_el,
                              uint sample_count, uint dest_tiled,
                              uint src_tiled, uint src_srgb, uint dest_srgb)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      4 * sample_count, src_tiled);

   for (uint c = 0; c < 4; ++c) {
      float sum = 0.0f;
      for (uint s = 0; s < sample_count; ++s) {
         float value = (float)src[src_offset_B + s * 4 + c] / 255.0f;
         if (src_srgb && c < 3)
            value = libagx_srgb_to_linear(value);
         sum += value;
      }

      float value = sum / (float)sample_count;
      if (dest_srgb && c < 3)
         value = libagx_linear_to_srgb(value);
      dest[dest_offset_B + c] = convert_uchar_sat_rte(value * 255.0f);
   }
}

/* Resolve the R32G32B32A32_FLOAT shape used by the render-target sample-count
 * probes. Each AIL pixel block contains sample_count consecutive float4s. */
KERNEL(32)
libagx_resolve_float32x4(global float *dest, global float *src,
                         uint width_px, uint height_px,
                         uint dest_x_px, uint dest_y_px,
                         uint src_x_px, uint src_y_px,
                         uint dest_pitch_B, uint dest_stride_el,
                         uint src_stride_el,
                         uint dest_tile_width_el, uint dest_tile_height_el,
                         uint src_tile_width_el, uint src_tile_height_el,
                         uint sample_count, uint dest_tiled, uint src_tiled)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 16, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      16 * sample_count, src_tiled);

   for (uint c = 0; c < 4; ++c) {
      float sum = 0.0f;
      for (uint s = 0; s < sample_count; ++s)
         sum += src[(src_offset_B + s * 16) / 4 + c];

      dest[dest_offset_B / 4 + c] = sum / (float)sample_count;
   }
}

/* Resolve a multisampled 32-bit float depth plane with D3D12's MIN mode. */
KERNEL(32)
libagx_resolve_float32_min(global float *dest, global float *src,
                           uint width_px, uint height_px,
                           uint dest_x_px, uint dest_y_px,
                           uint src_x_px, uint src_y_px,
                           uint dest_pitch_B, uint dest_stride_el,
                           uint src_stride_el,
                           uint dest_tile_width_el, uint dest_tile_height_el,
                           uint src_tile_width_el, uint src_tile_height_el,
                           uint sample_count, uint dest_tiled, uint src_tiled)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      4 * sample_count, src_tiled);

   float value = src[src_offset_B / 4];
   for (uint s = 1; s < sample_count; ++s)
      value = fmin(value, src[src_offset_B / 4 + s]);

   dest[dest_offset_B / 4] = value;
}

/* Resolve one 32-bit float channel by averaging all samples. This is the
 * scalar counterpart of libagx_resolve_float32x4 and also covers a D32_FLOAT
 * source resolved through the compatible R32_FLOAT view required by D3D12. */
KERNEL(32)
libagx_resolve_float32_average(global float *dest, global float *src,
                               uint width_px, uint height_px,
                               uint dest_x_px, uint dest_y_px,
                               uint src_x_px, uint src_y_px,
                               uint dest_pitch_B, uint dest_stride_el,
                               uint src_stride_el,
                               uint dest_tile_width_el,
                               uint dest_tile_height_el,
                               uint src_tile_width_el,
                               uint src_tile_height_el,
                               uint sample_count, uint dest_tiled,
                               uint src_tiled)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      4 * sample_count, src_tiled);

   float sum = 0.0f;
   for (uint s = 0; s < sample_count; ++s)
      sum += src[src_offset_B / 4 + s];

   dest[dest_offset_B / 4] = sum / (float)sample_count;
}

/* Resolve the scalar MIN/MAX modes used by 32-bit float/integer colour and
 * depth planes, plus the separate 8-bit stencil plane. Operation values are
 * private to d3d12agx: 0=float, 1=uint, 2=sint; choose_max selects the
 * extremum. Keeping the scalar size explicit makes every address bounded by
 * the same checked surface geometry used by the copy kernels. */
KERNEL(32)
libagx_resolve_scalar_extrema(global uint8_t *dest, global uint8_t *src,
                              uint width_px, uint height_px,
                              uint dest_x_px, uint dest_y_px,
                              uint src_x_px, uint src_y_px,
                              uint dest_pitch_B, uint dest_stride_el,
                              uint src_stride_el,
                              uint dest_tile_width_el,
                              uint dest_tile_height_el,
                              uint src_tile_width_el,
                              uint src_tile_height_el,
                              uint sample_count, uint dest_tiled,
                              uint src_tiled, uint scalar_size_B,
                              uint operation, uint choose_max)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, scalar_size_B, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset(
      src_x_px + x, src_y_px + y, 0, src_stride_el,
      src_tile_width_el, src_tile_height_el,
      scalar_size_B * sample_count, src_tiled);

   if (scalar_size_B == 1) {
      uint8_t value = src[src_offset_B];
      for (uint s = 1; s < sample_count; ++s) {
         uint8_t next = src[src_offset_B + s];
         value = choose_max ? max(value, next) : min(value, next);
      }
      dest[dest_offset_B] = value;
   } else if (operation == 0) {
      global float *src_f = (global float *)(src + src_offset_B);
      global float *dest_f = (global float *)(dest + dest_offset_B);
      float value = src_f[0];
      for (uint s = 1; s < sample_count; ++s)
         value = choose_max ? fmax(value, src_f[s]) : fmin(value, src_f[s]);
      dest_f[0] = value;
   } else if (operation == 1) {
      global uint *src_u = (global uint *)(src + src_offset_B);
      global uint *dest_u = (global uint *)(dest + dest_offset_B);
      uint value = src_u[0];
      for (uint s = 1; s < sample_count; ++s)
         value = choose_max ? max(value, src_u[s]) : min(value, src_u[s]);
      dest_u[0] = value;
   } else {
      global int *src_i = (global int *)(src + src_offset_B);
      global int *dest_i = (global int *)(dest + dest_offset_B);
      int value = src_i[0];
      for (uint s = 1; s < sample_count; ++s)
         value = choose_max ? max(value, src_i[s]) : min(value, src_i[s]);
      dest_i[0] = value;
   }
}

KERNEL(32)
libagx_fill_image(global uint8_t *dest, global uint8_t *value,
                  uint width_el, uint height_el,
                  uint dest_x_el, uint dest_y_el, uint dest_pitch_B,
                  uint dest_stride_el, uint dest_tile_width_el,
                  uint dest_tile_height_el, uint blocksize_B, uint dest_tiled)
{
   uint id = cl_global_id.x;
   if (id >= width_el * height_el)
      return;

   uint x = id % width_el;
   uint y = id / width_el;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_el + x, dest_y_el + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, blocksize_B, dest_tiled);

   for (uint b = 0; b < blocksize_B; ++b)
      dest[dest_offset_B + b] = value[b];
}

KERNEL(32)
libagx_fill_uint4(global uint4 *address, uint layer_stride_uint4, uint a,
                  uint b, uint c, uint d)
{
   uint offs = cl_global_id.x + cl_global_id.y * layer_stride_uint4;
   address[offs] = (uint4)(a, b, c, d);
}
