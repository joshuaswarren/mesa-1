/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Root signature blobs: serialize a D3D12_ROOT_SIGNATURE_DESC into the
 * DXBC/RTS0 container D3D12 defines, and parse one back.
 *
 * This matters beyond format bookkeeping. A root signature is the only
 * thing that says which shader register a root parameter feeds:
 * SetGraphicsRootDescriptorTable names a root PARAMETER index, while the
 * shader asks for a REGISTER, and nothing else connects the two. Until it
 * was parsed, the driver assumed parameter index == binding slot, which is
 * true only for the shapes our own tests happened to use.
 *
 * The container layout was verified byte for byte against blobs produced
 * by dxc (SDK 26100) with -T rootsig_1_0, for a root-descriptor-only
 * signature, a descriptor-table signature, and a mixed one using every
 * parameter type, a non-zero register space and a non-zero base register.
 * Those blobs are archived alongside the tests.
 */

#include "d12_private.h"

#include <stdlib.h>
#include <string.h>

/* 'DXBC' and 'RTS0' as little-endian 32-bit tags. */
#define D12_FOURCC(a, b, c, d)                                               \
   ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) |           \
    ((uint32_t)(d) << 24))

#define D12_TAG_DXBC D12_FOURCC('D', 'X', 'B', 'C')
#define D12_TAG_RTS0 D12_FOURCC('R', 'T', 'S', '0')
#define D12_TAG_RDAT D12_FOURCC('R', 'D', 'A', 'T')

/* Sizes of the serialized records, in bytes. */
#define D12_RS_HEADER_SIZE 24 /* version, nparam, poff, nsamp, soff, flags */
#define D12_RS_PARAM_SIZE 12  /* type, visibility, payload offset */
#define D12_RS_RANGE_SIZE 20
#define D12_RS_TABLE_SIZE 8    /* nranges, ranges offset */
#define D12_RS_DESCRIPTOR_SIZE 8  /* register, space */
#define D12_RS_CONSTANTS_SIZE 12  /* register, space, num values */
#define D12_RS_SAMPLER_SIZE 52
#define D12_RS_RANGE1_SIZE 24
#define D12_RS_DESCRIPTOR1_SIZE 12
#define D12_RS_SAMPLER1_SIZE 56

struct d12_writer {
   uint8_t *data;
   size_t size, used;
   bool failed;
};

static void
d12_copy_sampler_1_2_to_1_0(D3D12_STATIC_SAMPLER_DESC *dst,
                             const D3D12_STATIC_SAMPLER_DESC1 *src);

static void
w_reserve(struct d12_writer *w, size_t n)
{
   if (w->failed)
      return;

   if (w->used + n > w->size) {
      size_t want = MAX2(w->size * 2, w->used + n + 256);
      uint8_t *grown = realloc(w->data, want);
      if (!grown) {
         w->failed = true;
         return;
      }
      w->data = grown;
      w->size = want;
   }
}

static void
w_u32(struct d12_writer *w, uint32_t v)
{
   w_reserve(w, 4);
   if (w->failed)
      return;

   /* Explicit little-endian: the container is a file format, not a struct. */
   w->data[w->used++] = (uint8_t)(v);
   w->data[w->used++] = (uint8_t)(v >> 8);
   w->data[w->used++] = (uint8_t)(v >> 16);
   w->data[w->used++] = (uint8_t)(v >> 24);
}

static void
w_u32_at(struct d12_writer *w, size_t offset, uint32_t v)
{
   if (w->failed || offset + 4 > w->used)
      return;

   w->data[offset] = (uint8_t)(v);
   w->data[offset + 1] = (uint8_t)(v >> 8);
   w->data[offset + 2] = (uint8_t)(v >> 16);
   w->data[offset + 3] = (uint8_t)(v >> 24);
}

static uint32_t
r_u32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

static bool
d12_rs_span(size_t size, uint32_t offset, uint32_t count, size_t stride)
{
   return (size_t)offset <= size && count <= (size - offset) / stride;
}

static float
d12_rs_float(const uint8_t *p)
{
   uint32_t bits = r_u32(p);
   float value;
   memcpy(&value, &bits, sizeof(value));
   return value;
}

static void
store_u32(uint8_t *p, uint32_t v)
{
   p[0] = v;
   p[1] = v >> 8;
   p[2] = v >> 16;
   p[3] = v >> 24;
}

/* DXBC uses the RFC 1321 MD5 compression function with a container-specific
 * final block.  The finalization below was derived from SDK-produced DXBC
 * fixtures; keeping it here avoids a crypto dependency for a file checksum.
 */
static uint32_t
d12_rotl32(uint32_t x, unsigned n)
{
   return (x << n) | (x >> (32 - n));
}

static void
d12_md5_transform(uint32_t state[4], const uint8_t block[64])
{
   static const uint8_t shifts[64] = {
      7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
      5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
   };
   static const uint32_t constants[64] = {
      0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
      0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
      0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
      0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
      0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
      0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
      0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
      0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
      0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
      0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
      0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
      0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
      0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
      0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
      0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
      0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
   };
   uint32_t words[16];
   for (unsigned i = 0; i < 16; i++)
      words[i] = r_u32(block + i * 4);

   uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
   for (unsigned i = 0; i < 64; i++) {
      uint32_t f, word;

      if (i < 16) {
         f = (b & c) | (~b & d);
         word = i;
      } else if (i < 32) {
         f = (d & b) | (~d & c);
         word = (5 * i + 1) & 15;
      } else if (i < 48) {
         f = b ^ c ^ d;
         word = (3 * i + 5) & 15;
      } else {
         f = c ^ (b | ~d);
         word = (7 * i) & 15;
      }

      uint32_t old_d = d;
      d = c;
      c = b;
      b += d12_rotl32(a + f + constants[i] + words[word], shifts[i]);
      a = old_d;
   }

   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;
}

static void
d12_dxbc_checksum(const uint8_t *container, size_t size, uint32_t digest[4])
{
   const uint8_t *data = container + 20;
   size_t length = size - 20;
   size_t remaining = length;
   uint32_t state[4] = {
      0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476,
   };

   while (remaining >= 64) {
      d12_md5_transform(state, data);
      data += 64;
      remaining -= 64;
   }

   uint32_t num_bits = (uint32_t)(length * 8);
   uint32_t num_bits2 = (num_bits >> 2) | 1;
   uint8_t block[64] = {0};

   if (remaining >= 56) {
      memcpy(block, data, remaining);
      block[remaining] = 0x80;
      d12_md5_transform(state, block);

      memset(block, 0, sizeof(block));
      store_u32(block, num_bits);
   } else {
      store_u32(block, num_bits);
      memcpy(block + 4, data, remaining);
      block[4 + remaining] = 0x80;
   }

   store_u32(block + 60, num_bits2);
   d12_md5_transform(state, block);
   memcpy(digest, state, sizeof(state));
}

/*
 * Serialize. Offsets inside the RTS0 payload are relative to the start of
 * the payload, not to the file, so everything is written into a standalone
 * buffer and the container is wrapped around it afterwards.
 */
static bool
serialize_rts0(const D3D12_ROOT_SIGNATURE_DESC *desc, struct d12_writer *w)
{
   const uint32_t nparams = desc->NumParameters;
   const uint32_t nsamplers = desc->NumStaticSamplers;

   /* Parameters come first, then each parameter's payload, then any
    * descriptor ranges, then static samplers.
    */
   const uint32_t params_off = D12_RS_HEADER_SIZE;

   w_u32(w, 1); /* D3D_ROOT_SIGNATURE_VERSION_1_0 */
   w_u32(w, nparams);
   w_u32(w, params_off);
   w_u32(w, nsamplers);
   w_u32(w, 0); /* static sampler offset, patched below */
   w_u32(w, desc->Flags);

   /* Reserve the parameter records; their payload offsets are patched as
    * each payload is appended.
    */
   size_t param_record = w->used;
   for (uint32_t i = 0; i < nparams; i++) {
      w_u32(w, desc->pParameters[i].ParameterType);
      w_u32(w, desc->pParameters[i].ShaderVisibility);
      w_u32(w, 0);
   }

   for (uint32_t i = 0; i < nparams; i++) {
      const D3D12_ROOT_PARAMETER *p = &desc->pParameters[i];
      w_u32_at(w, param_record + i * D12_RS_PARAM_SIZE + 8,
               (uint32_t)w->used);

      switch (p->ParameterType) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         const D3D12_ROOT_DESCRIPTOR_TABLE *t = &p->DescriptorTable;
         if (t->NumDescriptorRanges && !t->pDescriptorRanges)
            return false;

         w_u32(w, t->NumDescriptorRanges);
         /* Ranges follow immediately after this 8-byte record. */
         w_u32(w, (uint32_t)(w->used + 4));

         for (uint32_t r = 0; r < t->NumDescriptorRanges; r++) {
            const D3D12_DESCRIPTOR_RANGE *rr = &t->pDescriptorRanges[r];
            w_u32(w, rr->RangeType);
            w_u32(w, rr->NumDescriptors);
            w_u32(w, rr->BaseShaderRegister);
            w_u32(w, rr->RegisterSpace);
            w_u32(w, rr->OffsetInDescriptorsFromTableStart);
         }
         break;
      }

      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         w_u32(w, p->Constants.ShaderRegister);
         w_u32(w, p->Constants.RegisterSpace);
         w_u32(w, p->Constants.Num32BitValues);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         w_u32(w, p->Descriptor.ShaderRegister);
         w_u32(w, p->Descriptor.RegisterSpace);
         break;

      default:
         return false;
      }
   }

   /* Static samplers. Recorded faithfully so the blob round-trips, even
    * though nothing consumes them until texturing.
    */
   uint32_t samplers_off = (uint32_t)w->used;
   for (uint32_t i = 0; i < nsamplers; i++) {
      const D3D12_STATIC_SAMPLER_DESC *s = &desc->pStaticSamplers[i];
      w_u32(w, s->Filter);
      w_u32(w, s->AddressU);
      w_u32(w, s->AddressV);
      w_u32(w, s->AddressW);
      union { float f; uint32_t u; } mip = {.f = s->MipLODBias};
      w_u32(w, mip.u);
      w_u32(w, s->MaxAnisotropy);
      w_u32(w, s->ComparisonFunc);
      w_u32(w, s->BorderColor);
      union { float f; uint32_t u; } lo = {.f = s->MinLOD};
      union { float f; uint32_t u; } hi = {.f = s->MaxLOD};
      w_u32(w, lo.u);
      w_u32(w, hi.u);
      w_u32(w, s->ShaderRegister);
      w_u32(w, s->RegisterSpace);
      w_u32(w, s->ShaderVisibility);
   }

   w_u32_at(w, 16, samplers_off);
   return !w->failed;
}

/* Root signatures 1.1 and 1.2 are wire versions 2 and 3.  They share their
 * parameter layout.  Version 1.2 only extends each static sampler with the
 * D3D12_SAMPLER_FLAGS word declared by the public D3D12 headers.
 */
static bool
serialize_rts0_1_x(uint32_t wire_version, uint32_t nparams,
                   const D3D12_ROOT_PARAMETER1 *params, uint32_t nsamplers,
                   const D3D12_STATIC_SAMPLER_DESC *samplers,
                   const D3D12_STATIC_SAMPLER_DESC1 *samplers1,
                   D3D12_ROOT_SIGNATURE_FLAGS flags, struct d12_writer *w)
{
   const uint32_t params_off = D12_RS_HEADER_SIZE;

   w_u32(w, wire_version);
   w_u32(w, nparams);
   w_u32(w, params_off);
   w_u32(w, nsamplers);
   w_u32(w, 0); /* static sampler offset, patched below */
   w_u32(w, flags);

   size_t param_record = w->used;
   for (uint32_t i = 0; i < nparams; i++) {
      w_u32(w, params[i].ParameterType);
      w_u32(w, params[i].ShaderVisibility);
      w_u32(w, 0);
   }

   for (uint32_t i = 0; i < nparams; i++) {
      const D3D12_ROOT_PARAMETER1 *p = &params[i];
      w_u32_at(w, param_record + i * D12_RS_PARAM_SIZE + 8,
               (uint32_t)w->used);

      switch (p->ParameterType) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         const D3D12_ROOT_DESCRIPTOR_TABLE1 *t = &p->DescriptorTable;
         if (t->NumDescriptorRanges && !t->pDescriptorRanges)
            return false;

         w_u32(w, t->NumDescriptorRanges);
         w_u32(w, (uint32_t)(w->used + 4));

         for (uint32_t r = 0; r < t->NumDescriptorRanges; r++) {
            const D3D12_DESCRIPTOR_RANGE1 *rr = &t->pDescriptorRanges[r];
            w_u32(w, rr->RangeType);
            w_u32(w, rr->NumDescriptors);
            w_u32(w, rr->BaseShaderRegister);
            w_u32(w, rr->RegisterSpace);
            w_u32(w, rr->Flags);
            w_u32(w, rr->OffsetInDescriptorsFromTableStart);
         }
         break;
      }

      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         w_u32(w, p->Constants.ShaderRegister);
         w_u32(w, p->Constants.RegisterSpace);
         w_u32(w, p->Constants.Num32BitValues);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         w_u32(w, p->Descriptor.ShaderRegister);
         w_u32(w, p->Descriptor.RegisterSpace);
         w_u32(w, p->Descriptor.Flags);
         break;

      default:
         return false;
      }
   }

   uint32_t samplers_off = (uint32_t)w->used;
   for (uint32_t i = 0; i < nsamplers; i++) {
      D3D12_STATIC_SAMPLER_DESC sampler;
      const D3D12_STATIC_SAMPLER_DESC *s;
      if (wire_version == 3) {
         d12_copy_sampler_1_2_to_1_0(&sampler, &samplers1[i]);
         s = &sampler;
      } else {
         s = &samplers[i];
      }
      w_u32(w, s->Filter);
      w_u32(w, s->AddressU);
      w_u32(w, s->AddressV);
      w_u32(w, s->AddressW);
      union {
         float f;
         uint32_t u;
      } mip = {.f = s->MipLODBias};
      w_u32(w, mip.u);
      w_u32(w, s->MaxAnisotropy);
      w_u32(w, s->ComparisonFunc);
      w_u32(w, s->BorderColor);
      union {
         float f;
         uint32_t u;
      } lo = {.f = s->MinLOD};
      union {
         float f;
         uint32_t u;
      } hi = {.f = s->MaxLOD};
      w_u32(w, lo.u);
      w_u32(w, hi.u);
      w_u32(w, s->ShaderRegister);
      w_u32(w, s->RegisterSpace);
      w_u32(w, s->ShaderVisibility);
      if (wire_version == 3)
         w_u32(w, samplers1[i].Flags);
   }

   w_u32_at(w, 16, samplers_off);
   return !w->failed;
}

static HRESULT
wrap_rts0(const struct d12_writer *payload, void **out_blob, size_t *out_size)
{
   /* Wrap in a DXBC container: tag, 16-byte checksum, version, total size,
    * part count, one part offset, then the part.
    */
   const size_t header = 4 + 16 + 4 + 4 + 4 + 4;
   const size_t total = header + 8 + payload->used;

   struct d12_writer out = {0};
   w_reserve(&out, total);
   if (out.failed)
      return E_OUTOFMEMORY;

   w_u32(&out, D12_TAG_DXBC);
   /* Reserved until the complete container is available. */
   for (unsigned i = 0; i < 4; i++)
      w_u32(&out, 0);
   w_u32(&out, 1);                  /* container version */
   w_u32(&out, (uint32_t)total);
   w_u32(&out, 1);                  /* part count */
   w_u32(&out, (uint32_t)header);   /* part[0] offset */
   w_u32(&out, D12_TAG_RTS0);
   w_u32(&out, (uint32_t)payload->used);

   w_reserve(&out, payload->used);
   if (out.failed) {
      free(out.data);
      return E_OUTOFMEMORY;
   }
   memcpy(out.data + out.used, payload->data, payload->used);
   out.used += payload->used;

   uint32_t checksum[4];
   d12_dxbc_checksum(out.data, out.used, checksum);
   for (unsigned i = 0; i < 4; i++)
      w_u32_at(&out, 4 + i * 4, checksum[i]);

   *out_blob = out.data;
   *out_size = out.used;
   return S_OK;
}

HRESULT
d12_serialize_root_signature(const D3D12_ROOT_SIGNATURE_DESC *desc,
                             void **out_blob, size_t *out_size)
{
   if (!desc || !out_blob || !out_size)
      return E_INVALIDARG;

   if (desc->NumParameters && !desc->pParameters)
      return E_INVALIDARG;
   if (desc->NumStaticSamplers && !desc->pStaticSamplers)
      return E_INVALIDARG;

   struct d12_writer payload = {0};
   if (!serialize_rts0(desc, &payload)) {
      free(payload.data);
      return E_INVALIDARG;
   }

   HRESULT hr = wrap_rts0(&payload, out_blob, out_size);
   free(payload.data);
   return hr;
}

HRESULT
d12_serialize_versioned_root_signature(
   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc, void **out_blob,
   size_t *out_size)
{
   if (!desc || !out_blob || !out_size)
      return E_INVALIDARG;

   if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_0)
      return d12_serialize_root_signature(&desc->Desc_1_0, out_blob,
                                          out_size);

   uint32_t wire_version, nparams, nsamplers;
   const D3D12_ROOT_PARAMETER1 *params;
   const D3D12_STATIC_SAMPLER_DESC *samplers = NULL;
   const D3D12_STATIC_SAMPLER_DESC1 *samplers1 = NULL;
   D3D12_ROOT_SIGNATURE_FLAGS flags;

   if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
      const D3D12_ROOT_SIGNATURE_DESC1 *v1 = &desc->Desc_1_1;
      wire_version = 2;
      nparams = v1->NumParameters;
      params = v1->pParameters;
      nsamplers = v1->NumStaticSamplers;
      samplers = v1->pStaticSamplers;
      flags = v1->Flags;
   } else if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_2) {
      const D3D12_ROOT_SIGNATURE_DESC2 *v2 = &desc->Desc_1_2;
      wire_version = 3;
      nparams = v2->NumParameters;
      params = v2->pParameters;
      nsamplers = v2->NumStaticSamplers;
      samplers1 = v2->pStaticSamplers;
      flags = v2->Flags;
   } else {
      return E_INVALIDARG;
   }

   if ((nparams && !params) ||
       (nsamplers && wire_version == 2 && !samplers) ||
       (nsamplers && wire_version == 3 && !samplers1))
      return E_INVALIDARG;

   struct d12_writer payload = {0};
   if (!serialize_rts0_1_x(wire_version, nparams, params, nsamplers, samplers,
                           samplers1, flags, &payload)) {
      free(payload.data);
      return E_INVALIDARG;
   }

   HRESULT hr = wrap_rts0(&payload, out_blob, out_size);
   free(payload.data);
   return hr;
}

/*
 * Parse. Bounds are checked against the part, not the file: a blob from an
 * application is untrusted input, and an offset that runs past the end must
 * be a rejection rather than a read into whatever follows.
 */
static bool
find_dxbc_part(const void *blob, size_t size, uint32_t tag,
               const uint8_t **out, size_t *out_size)
{
   const uint8_t *p = blob;

   if (size < 32 || r_u32(p) != D12_TAG_DXBC)
      return false;

   uint32_t total = r_u32(p + 24);
   uint32_t nparts = r_u32(p + 28);
   if (total < 32 || total > size || nparts == 0 || nparts > 64)
      return false;
   if (32 + (size_t)nparts * 4 > total)
      return false;

   for (uint32_t i = 0; i < nparts; i++) {
      uint32_t off = r_u32(p + 32 + i * 4);
      /* size_t, or the addition wraps: off = 0xFFFFFFFC makes this 4, the
       * check passes, and the read below lands ~4 GiB past the blob. The
       * blob is application-supplied.
       */
      if ((size_t)off + 8 > total)
         continue;
      if (r_u32(p + off) != tag)
         continue;

      uint32_t part_size = r_u32(p + off + 4);
      if ((size_t)off + 8 + part_size > total)
         return false;

      *out = p + off + 8;
      *out_size = part_size;
      return true;
   }

   return false;
}

static bool
find_rts0(const void *blob, size_t size, const uint8_t **out, size_t *out_size)
{
   return find_dxbc_part(blob, size, D12_TAG_RTS0, out, out_size);
}

bool
d12_root_signature_has_rts0(const void *blob, size_t blob_size)
{
   const uint8_t *rs;
   size_t rs_size;

   return find_rts0(blob, blob_size, &rs, &rs_size);
}

bool
d12_root_signature_parse(const void *blob, size_t blob_size,
                         struct d12_root_layout *out)
{
   memset(out, 0, sizeof(*out));

   const uint8_t *rs;
   size_t rs_size;
   if (!find_rts0(blob, blob_size, &rs, &rs_size)) {
      mesa_loge("root signature blob has no RTS0 part");
      return false;
   }

   if (rs_size < D12_RS_HEADER_SIZE) {
      mesa_loge("root signature part is truncated");
      return false;
   }

   uint32_t version = r_u32(rs + 0);
   uint32_t nparams = r_u32(rs + 4);
   uint32_t params_off = r_u32(rs + 8);
   uint32_t nsamplers = r_u32(rs + 12);
   uint32_t samplers_off = r_u32(rs + 16);
   out->flags = r_u32(rs + 20);

   if (version < 1 || version > 3) {
      mesa_loge("root signature version %u unsupported", version);
      return false;
   }

   /* Version 1.1 adds a flags word to each descriptor range and to root
    * descriptors. The extra word only describes volatility promises the
    * application makes, so it changes the stride, not the meaning.
    */
   const uint32_t range_size =
      version == 1 ? D12_RS_RANGE_SIZE : D12_RS_RANGE1_SIZE;
   const uint32_t descriptor_size =
      version == 1 ? D12_RS_DESCRIPTOR_SIZE : D12_RS_DESCRIPTOR1_SIZE;
   const uint32_t sampler_size =
      version == 3 ? D12_RS_SAMPLER1_SIZE : D12_RS_SAMPLER_SIZE;

   if (nparams > D12_MAX_ROOT_PARAMS) {
      mesa_loge("root signature has %u parameters, limit is %u", nparams,
                D12_MAX_ROOT_PARAMS);
      return false;
   }

   if (nsamplers > D12_MAX_RESOURCE_BINDINGS) {
      mesa_loge("root signature has %u static samplers, limit is %u",
                nsamplers, D12_MAX_RESOURCE_BINDINGS);
      return false;
   }

   if (nsamplers &&
       !d12_rs_span(rs_size, samplers_off, nsamplers, sampler_size)) {
      mesa_loge("root signature static-sampler table runs past the blob");
      return false;
   }

   if ((size_t)params_off + (size_t)nparams * D12_RS_PARAM_SIZE > rs_size) {
      mesa_loge("root signature parameter table runs past the blob");
      return false;
   }

   uint32_t total_ranges = 0;

   /* Descriptor ranges live in one flat allocation owned by the layout.
    * Count and bounds-check every table before allocating so a malformed
    * container cannot leave a partially constructed ownership graph. */
   for (uint32_t i = 0; i < nparams; i++) {
      const uint8_t *rec = rs + params_off + i * D12_RS_PARAM_SIZE;
      if (r_u32(rec + 0) != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
         continue;

      uint32_t payload = r_u32(rec + 8);
      if ((size_t)payload + D12_RS_TABLE_SIZE > rs_size)
         goto truncated;

      uint32_t nranges = r_u32(rs + payload);
      uint32_t ranges_off = r_u32(rs + payload + 4);
      if (nranges > D12_MAX_ROOT_RANGES - total_ranges) {
         mesa_loge("root signature has more than %u descriptor ranges",
                   D12_MAX_ROOT_RANGES);
         return false;
      }
      if (!d12_rs_span(rs_size, ranges_off, nranges, range_size))
         goto truncated;
      total_ranges += nranges;
   }

   if (total_ranges) {
      out->ranges = calloc(total_ranges, sizeof(*out->ranges));
      if (!out->ranges) {
         mesa_loge("out of memory parsing root-signature ranges");
         return false;
      }
   }
   out->num_ranges = total_ranges;
   out->num_params = nparams;
   uint32_t next_range = 0;

   for (uint32_t i = 0; i < nparams; i++) {
      const uint8_t *rec = rs + params_off + i * D12_RS_PARAM_SIZE;
      struct d12_root_param *rp = &out->params[i];

      rp->type = r_u32(rec + 0);
      rp->visibility = r_u32(rec + 4);
      uint32_t payload = r_u32(rec + 8);

      switch (rp->type) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         if ((size_t)payload + D12_RS_TABLE_SIZE > rs_size)
            goto truncated;

         uint32_t nranges = r_u32(rs + payload);
         uint32_t ranges_off = r_u32(rs + payload + 4);

         if (!d12_rs_span(rs_size, ranges_off, nranges, range_size))
            goto truncated;

         rp->first_range = next_range;
         rp->num_ranges = nranges;
         struct d12_root_range *param_ranges =
            nranges ? &out->ranges[next_range] : NULL;
         for (uint32_t r = 0; r < nranges; r++) {
            const uint8_t *rr = rs + ranges_off + r * range_size;
            struct d12_root_range *out_range = &param_ranges[r];

            out_range->type = r_u32(rr + 0);
            out_range->num_descriptors = r_u32(rr + 4);
            out_range->base_register = r_u32(rr + 8);
            out_range->space = r_u32(rr + 12);
            /* v1.1 puts its flags word before the offset. */
            out_range->offset_in_descriptors =
               r_u32(rr + (version == 1 ? 16 : 20));
         }
         next_range += nranges;

         /*
          * A descriptor table is bound to ONE heap, and D3D12 has two heap
          * types a shader can see -- CBV/SRV/UAV, and SAMPLER. A table
          * mixing sampler ranges with any of the others could not be
          * satisfied by a single heap, so the API rejects it outright.
          *
          * test_create_root_signature builds exactly that table and expects
          * E_INVALIDARG. Accepting it also leaked: the test only releases
          * the root signature on the paths where creation was supposed to
          * fail, so a stray success showed up a second time as a device
          * refcount left at 1.
          */
         if (nranges > 0) {
            bool has_sampler = false, has_other = false;

            for (uint32_t r = 0; r < nranges; r++) {
               if (param_ranges[r].type ==
                   D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)
                  has_sampler = true;
               else
                  has_other = true;
            }

            if (has_sampler && has_other) {
               mesa_loge("descriptor table mixes sampler and non-sampler "
                         "ranges; no single heap can satisfy it");
               goto invalid;
            }
         }
         break;
      }

      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         if ((size_t)payload + D12_RS_CONSTANTS_SIZE > rs_size)
            goto truncated;
         rp->base_register = r_u32(rs + payload + 0);
         rp->space = r_u32(rs + payload + 4);
         rp->num_constants = r_u32(rs + payload + 8);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         if ((size_t)payload + descriptor_size > rs_size)
            goto truncated;
         rp->base_register = r_u32(rs + payload + 0);
         rp->space = r_u32(rs + payload + 4);
         break;

      default:
         mesa_loge("root parameter %u has unknown type %u", i, rp->type);
         goto invalid;
      }
   }

   if (nsamplers) {
      out->static_samplers = calloc(nsamplers, sizeof(*out->static_samplers));
      if (!out->static_samplers) {
         mesa_loge("out of memory parsing root-signature static samplers");
         goto invalid;
      }

      for (uint32_t i = 0; i < nsamplers; i++) {
         const uint8_t *in = rs + samplers_off + i * sampler_size;
         D3D12_STATIC_SAMPLER_DESC1 *s = &out->static_samplers[i];

         s->Filter = r_u32(in + 0);
         s->AddressU = r_u32(in + 4);
         s->AddressV = r_u32(in + 8);
         s->AddressW = r_u32(in + 12);
         s->MipLODBias = d12_rs_float(in + 16);
         s->MaxAnisotropy = r_u32(in + 20);
         s->ComparisonFunc = r_u32(in + 24);
         s->BorderColor = r_u32(in + 28);
         s->MinLOD = d12_rs_float(in + 32);
         s->MaxLOD = d12_rs_float(in + 36);
         s->ShaderRegister = r_u32(in + 40);
         s->RegisterSpace = r_u32(in + 44);
         s->ShaderVisibility = r_u32(in + 48);
         s->Flags = version == 3 ? r_u32(in + 52)
                                 : D3D12_SAMPLER_FLAG_NONE;
      }
   }
   out->num_static_samplers = nsamplers;

   /* Root descriptors and constants each occupy one shader register. Root
    * constants share the CBV namespace; overlapping stage visibility cannot
    * choose between two values for the same register. Disjoint stages may
    * legitimately reuse it and have separate entries in the binding map. */
   for (uint32_t i = 0; i < nparams; i++) {
      const struct d12_root_param *a = &out->params[i];
      if (a->type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
         continue;
      unsigned a_type = a->type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
                           ? D3D12_ROOT_PARAMETER_TYPE_CBV : a->type;
      for (uint32_t j = 0; j < i; j++) {
         const struct d12_root_param *b = &out->params[j];
         unsigned b_type = b->type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
                              ? D3D12_ROOT_PARAMETER_TYPE_CBV : b->type;
         if (a_type == b_type && a->space == b->space &&
             a->base_register == b->base_register &&
             (a->visibility == D3D12_SHADER_VISIBILITY_ALL ||
              b->visibility == D3D12_SHADER_VISIBILITY_ALL ||
              a->visibility == b->visibility)) {
            mesa_loge("root signature aliases shader register in parameters "
                      "%u and %u with overlapping visibility", j, i);
            goto invalid;
         }
      }
   }

   return true;

truncated:
   mesa_loge("root signature parameter payload runs past the blob");
invalid:
   d12_root_layout_cleanup(out);
   return false;
}

void
d12_root_layout_cleanup(struct d12_root_layout *layout)
{
   free(layout->ranges);
   free(layout->static_samplers);
   memset(layout, 0, sizeof(*layout));
}

/* The public deserializer interfaces expose pointers whose lifetime is tied
 * to the COM object.  Decode into one owned canonical 1.1-shaped model, then
 * build stable 1.0, 1.1 and 1.2 views over the same allocation set.
 */
struct d12_root_signature_desc_store {
   D3D_ROOT_SIGNATURE_VERSION source_version;
   D3D12_VERSIONED_ROOT_SIGNATURE_DESC views[3];

   D3D12_ROOT_PARAMETER *params0;
   D3D12_ROOT_PARAMETER1 *params1;
   D3D12_DESCRIPTOR_RANGE *ranges0;
   D3D12_DESCRIPTOR_RANGE1 *ranges1;
   D3D12_STATIC_SAMPLER_DESC *samplers0;
   D3D12_STATIC_SAMPLER_DESC1 *samplers1;
};

static void
d12_root_signature_desc_store_cleanup(struct d12_root_signature_desc_store *s)
{
   free(s->params0);
   free(s->params1);
   free(s->ranges0);
   free(s->ranges1);
   free(s->samplers0);
   free(s->samplers1);
   memset(s, 0, sizeof(*s));
}

static void
d12_copy_sampler_1_2_to_1_0(D3D12_STATIC_SAMPLER_DESC *dst,
                             const D3D12_STATIC_SAMPLER_DESC1 *src)
{
   dst->Filter = src->Filter;
   dst->AddressU = src->AddressU;
   dst->AddressV = src->AddressV;
   dst->AddressW = src->AddressW;
   dst->MipLODBias = src->MipLODBias;
   dst->MaxAnisotropy = src->MaxAnisotropy;
   dst->ComparisonFunc = src->ComparisonFunc;
   dst->BorderColor = src->BorderColor;
   dst->MinLOD = src->MinLOD;
   dst->MaxLOD = src->MaxLOD;
   dst->ShaderRegister = src->ShaderRegister;
   dst->RegisterSpace = src->RegisterSpace;
   dst->ShaderVisibility = src->ShaderVisibility;
}

static HRESULT
d12_root_signature_desc_store_init(struct d12_root_signature_desc_store *s,
                                   const void *blob, size_t blob_size)
{
   const uint8_t *rs;
   size_t rs_size;
   size_t range_count = 0;

   memset(s, 0, sizeof(*s));
   if (!find_rts0(blob, blob_size, &rs, &rs_size) ||
       rs_size < D12_RS_HEADER_SIZE)
      return E_INVALIDARG;

   const uint32_t wire_version = r_u32(rs);
   const uint32_t nparams = r_u32(rs + 4);
   const uint32_t params_off = r_u32(rs + 8);
   const uint32_t nsamplers = r_u32(rs + 12);
   const uint32_t samplers_off = r_u32(rs + 16);
   const D3D12_ROOT_SIGNATURE_FLAGS flags = r_u32(rs + 20);
   const uint32_t range_size = wire_version == 1
      ? D12_RS_RANGE_SIZE : D12_RS_RANGE1_SIZE;
   const uint32_t descriptor_size = wire_version == 1
      ? D12_RS_DESCRIPTOR_SIZE : D12_RS_DESCRIPTOR1_SIZE;
   const uint32_t sampler_size = wire_version == 3
      ? D12_RS_SAMPLER1_SIZE : D12_RS_SAMPLER_SIZE;

   if (wire_version < 1 || wire_version > 3 ||
       (nparams && !d12_rs_span(rs_size, params_off, nparams,
                                D12_RS_PARAM_SIZE)) ||
       (nsamplers && !d12_rs_span(rs_size, samplers_off, nsamplers,
                                  sampler_size)))
      return E_INVALIDARG;

   /* Validate every referenced payload and count the flattened ranges before
    * allocating anything.  The blob is application-controlled input.
    */
   for (uint32_t i = 0; i < nparams; i++) {
      const uint8_t *rec = rs + params_off + i * D12_RS_PARAM_SIZE;
      const uint32_t type = r_u32(rec);
      const uint32_t payload = r_u32(rec + 8);

      switch (type) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         if (!d12_rs_span(rs_size, payload, 1, D12_RS_TABLE_SIZE))
            return E_INVALIDARG;
         const uint32_t nranges = r_u32(rs + payload);
         const uint32_t ranges_off = r_u32(rs + payload + 4);
         if ((nranges && !d12_rs_span(rs_size, ranges_off, nranges,
                                     range_size)) ||
             nranges > SIZE_MAX - range_count)
            return E_INVALIDARG;
         range_count += nranges;
         break;
      }
      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         if (!d12_rs_span(rs_size, payload, 1, D12_RS_CONSTANTS_SIZE))
            return E_INVALIDARG;
         break;
      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         if (!d12_rs_span(rs_size, payload, 1, descriptor_size))
            return E_INVALIDARG;
         break;
      default:
         return E_INVALIDARG;
      }
   }

   if (nparams) {
      s->params0 = calloc(nparams, sizeof(*s->params0));
      s->params1 = calloc(nparams, sizeof(*s->params1));
      if (!s->params0 || !s->params1)
         goto oom;
   }
   if (range_count) {
      s->ranges0 = calloc(range_count, sizeof(*s->ranges0));
      s->ranges1 = calloc(range_count, sizeof(*s->ranges1));
      if (!s->ranges0 || !s->ranges1)
         goto oom;
   }
   if (nsamplers) {
      s->samplers0 = calloc(nsamplers, sizeof(*s->samplers0));
      s->samplers1 = calloc(nsamplers, sizeof(*s->samplers1));
      if (!s->samplers0 || !s->samplers1)
         goto oom;
   }

   size_t next_range = 0;
   for (uint32_t i = 0; i < nparams; i++) {
      const uint8_t *rec = rs + params_off + i * D12_RS_PARAM_SIZE;
      D3D12_ROOT_PARAMETER *p0 = &s->params0[i];
      D3D12_ROOT_PARAMETER1 *p1 = &s->params1[i];
      const uint32_t payload = r_u32(rec + 8);

      p0->ParameterType = p1->ParameterType = r_u32(rec);
      p0->ShaderVisibility = p1->ShaderVisibility = r_u32(rec + 4);

      switch (p0->ParameterType) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         const uint32_t nranges = r_u32(rs + payload);
         const uint32_t ranges_off = r_u32(rs + payload + 4);
         p0->DescriptorTable.NumDescriptorRanges = nranges;
         p1->DescriptorTable.NumDescriptorRanges = nranges;
         p0->DescriptorTable.pDescriptorRanges = nranges
            ? &s->ranges0[next_range] : NULL;
         p1->DescriptorTable.pDescriptorRanges = nranges
            ? &s->ranges1[next_range] : NULL;

         for (uint32_t r = 0; r < nranges; r++) {
            const uint8_t *rr = rs + ranges_off + r * range_size;
            D3D12_DESCRIPTOR_RANGE *r0 = &s->ranges0[next_range + r];
            D3D12_DESCRIPTOR_RANGE1 *r1 = &s->ranges1[next_range + r];

            r0->RangeType = r1->RangeType = r_u32(rr);
            r0->NumDescriptors = r1->NumDescriptors = r_u32(rr + 4);
            r0->BaseShaderRegister = r1->BaseShaderRegister = r_u32(rr + 8);
            r0->RegisterSpace = r1->RegisterSpace = r_u32(rr + 12);
            r1->Flags = wire_version == 1
               ? D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                    D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE
               : r_u32(rr + 16);
            r0->OffsetInDescriptorsFromTableStart =
               r1->OffsetInDescriptorsFromTableStart =
                  r_u32(rr + (wire_version == 1 ? 16 : 20));
         }
         next_range += nranges;
         break;
      }
      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         p0->Constants.ShaderRegister = p1->Constants.ShaderRegister =
            r_u32(rs + payload);
         p0->Constants.RegisterSpace = p1->Constants.RegisterSpace =
            r_u32(rs + payload + 4);
         p0->Constants.Num32BitValues = p1->Constants.Num32BitValues =
            r_u32(rs + payload + 8);
         break;
      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         p0->Descriptor.ShaderRegister = p1->Descriptor.ShaderRegister =
            r_u32(rs + payload);
         p0->Descriptor.RegisterSpace = p1->Descriptor.RegisterSpace =
            r_u32(rs + payload + 4);
         p1->Descriptor.Flags = wire_version == 1
            ? D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE
            : r_u32(rs + payload + 8);
         break;
      }
   }

   for (uint32_t i = 0; i < nsamplers; i++) {
      const uint8_t *in = rs + samplers_off + i * sampler_size;
      D3D12_STATIC_SAMPLER_DESC1 *s1 = &s->samplers1[i];
      s1->Filter = r_u32(in);
      s1->AddressU = r_u32(in + 4);
      s1->AddressV = r_u32(in + 8);
      s1->AddressW = r_u32(in + 12);
      s1->MipLODBias = d12_rs_float(in + 16);
      s1->MaxAnisotropy = r_u32(in + 20);
      s1->ComparisonFunc = r_u32(in + 24);
      s1->BorderColor = r_u32(in + 28);
      s1->MinLOD = d12_rs_float(in + 32);
      s1->MaxLOD = d12_rs_float(in + 36);
      s1->ShaderRegister = r_u32(in + 40);
      s1->RegisterSpace = r_u32(in + 44);
      s1->ShaderVisibility = r_u32(in + 48);
      s1->Flags = wire_version == 3 ? r_u32(in + 52)
                                    : D3D12_SAMPLER_FLAG_NONE;
      d12_copy_sampler_1_2_to_1_0(&s->samplers0[i], s1);
   }

   s->source_version = (D3D_ROOT_SIGNATURE_VERSION)wire_version;
   s->views[0].Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
   s->views[0].Desc_1_0 = (D3D12_ROOT_SIGNATURE_DESC) {
      .NumParameters = nparams,
      .pParameters = nparams ? s->params0 : NULL,
      .NumStaticSamplers = nsamplers,
      .pStaticSamplers = nsamplers ? s->samplers0 : NULL,
      .Flags = flags,
   };
   s->views[1].Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
   s->views[1].Desc_1_1 = (D3D12_ROOT_SIGNATURE_DESC1) {
      .NumParameters = nparams,
      .pParameters = nparams ? s->params1 : NULL,
      .NumStaticSamplers = nsamplers,
      .pStaticSamplers = nsamplers ? s->samplers0 : NULL,
      .Flags = flags,
   };
   s->views[2].Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
   s->views[2].Desc_1_2 = (D3D12_ROOT_SIGNATURE_DESC2) {
      .NumParameters = nparams,
      .pParameters = nparams ? s->params1 : NULL,
      .NumStaticSamplers = nsamplers,
      .pStaticSamplers = nsamplers ? s->samplers1 : NULL,
      .Flags = flags,
   };
   return S_OK;

oom:
   d12_root_signature_desc_store_cleanup(s);
   return E_OUTOFMEMORY;
}

struct d12_root_signature_deserializer {
   ID3D12RootSignatureDeserializer iface;
   uint32_t refcount;
   struct d12_root_signature_desc_store store;
};

static HRESULT STDMETHODCALLTYPE
d12_root_signature_deserializer_QueryInterface(
   ID3D12RootSignatureDeserializer *iface, REFIID riid, void **out)
{
   struct d12_root_signature_deserializer *deserializer = (void *)iface;
   if (!out)
      return E_POINTER;
   *out = NULL;

   /* Match the native factory object's deliberately narrow interface. */
   if (!d12_guid_equal(riid, &IID_ID3D12RootSignatureDeserializer))
      return E_NOINTERFACE;

   p_atomic_inc(&deserializer->refcount);
   *out = iface;
   return S_OK;
}

static ULONG STDMETHODCALLTYPE
d12_root_signature_deserializer_AddRef(ID3D12RootSignatureDeserializer *iface)
{
   struct d12_root_signature_deserializer *deserializer = (void *)iface;
   return p_atomic_inc_return(&deserializer->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_root_signature_deserializer_Release(ID3D12RootSignatureDeserializer *iface)
{
   struct d12_root_signature_deserializer *deserializer = (void *)iface;
   uint32_t refs = p_atomic_dec_return(&deserializer->refcount);
   if (!refs) {
      d12_root_signature_desc_store_cleanup(&deserializer->store);
      free(deserializer);
   }
   return refs;
}

static const D3D12_ROOT_SIGNATURE_DESC *STDMETHODCALLTYPE
d12_root_signature_deserializer_GetRootSignatureDesc(
   ID3D12RootSignatureDeserializer *iface)
{
   struct d12_root_signature_deserializer *deserializer = (void *)iface;
   return &deserializer->store.views[0].Desc_1_0;
}

static const ID3D12RootSignatureDeserializerVtbl
d12_root_signature_deserializer_vtbl = {
   .QueryInterface = d12_root_signature_deserializer_QueryInterface,
   .AddRef = d12_root_signature_deserializer_AddRef,
   .Release = d12_root_signature_deserializer_Release,
   .GetRootSignatureDesc = d12_root_signature_deserializer_GetRootSignatureDesc,
};

struct d12_versioned_root_signature_deserializer {
   ID3D12VersionedRootSignatureDeserializer iface;
   uint32_t refcount;
   struct d12_root_signature_desc_store store;
};

static HRESULT STDMETHODCALLTYPE
d12_versioned_root_signature_deserializer_QueryInterface(
   ID3D12VersionedRootSignatureDeserializer *iface, REFIID riid, void **out)
{
   struct d12_versioned_root_signature_deserializer *deserializer =
      (void *)iface;
   if (!out)
      return E_POINTER;
   *out = NULL;

   if (!d12_guid_equal(riid, &IID_ID3D12VersionedRootSignatureDeserializer))
      return E_NOINTERFACE;

   p_atomic_inc(&deserializer->refcount);
   *out = iface;
   return S_OK;
}

static ULONG STDMETHODCALLTYPE
d12_versioned_root_signature_deserializer_AddRef(
   ID3D12VersionedRootSignatureDeserializer *iface)
{
   struct d12_versioned_root_signature_deserializer *deserializer =
      (void *)iface;
   return p_atomic_inc_return(&deserializer->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_versioned_root_signature_deserializer_Release(
   ID3D12VersionedRootSignatureDeserializer *iface)
{
   struct d12_versioned_root_signature_deserializer *deserializer =
      (void *)iface;
   uint32_t refs = p_atomic_dec_return(&deserializer->refcount);
   if (!refs) {
      d12_root_signature_desc_store_cleanup(&deserializer->store);
      free(deserializer);
   }
   return refs;
}

static HRESULT STDMETHODCALLTYPE
d12_versioned_root_signature_deserializer_GetRootSignatureDescAtVersion(
   ID3D12VersionedRootSignatureDeserializer *iface,
   D3D_ROOT_SIGNATURE_VERSION version,
   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC **out)
{
   struct d12_versioned_root_signature_deserializer *deserializer =
      (void *)iface;
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (version < D3D_ROOT_SIGNATURE_VERSION_1_0 ||
       version > D3D_ROOT_SIGNATURE_VERSION_1_2)
      return E_INVALIDARG;

   *out = &deserializer->store.views[version - 1];
   return S_OK;
}

static const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *STDMETHODCALLTYPE
d12_versioned_root_signature_deserializer_GetUnconvertedRootSignatureDesc(
   ID3D12VersionedRootSignatureDeserializer *iface)
{
   struct d12_versioned_root_signature_deserializer *deserializer =
      (void *)iface;
   return &deserializer->store.views[deserializer->store.source_version - 1];
}

static const ID3D12VersionedRootSignatureDeserializerVtbl
d12_versioned_root_signature_deserializer_vtbl = {
   .QueryInterface = d12_versioned_root_signature_deserializer_QueryInterface,
   .AddRef = d12_versioned_root_signature_deserializer_AddRef,
   .Release = d12_versioned_root_signature_deserializer_Release,
   .GetRootSignatureDescAtVersion =
      d12_versioned_root_signature_deserializer_GetRootSignatureDescAtVersion,
   .GetUnconvertedRootSignatureDesc =
      d12_versioned_root_signature_deserializer_GetUnconvertedRootSignatureDesc,
};

HRESULT
d12_root_signature_deserializer_create(const void *src, size_t size,
                                       REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!src || !riid)
      return E_INVALIDARG;
   if (!d12_guid_equal(riid, &IID_ID3D12RootSignatureDeserializer))
      return E_NOINTERFACE;

   struct d12_root_signature_deserializer *deserializer =
      calloc(1, sizeof(*deserializer));
   if (!deserializer)
      return E_OUTOFMEMORY;

   HRESULT hr = d12_root_signature_desc_store_init(&deserializer->store,
                                                   src, size);
   if (FAILED(hr)) {
      free(deserializer);
      return hr;
   }

   deserializer->iface.lpVtbl =
      (ID3D12RootSignatureDeserializerVtbl *)
         &d12_root_signature_deserializer_vtbl;
   deserializer->refcount = 1;
   *out = &deserializer->iface;
   return S_OK;
}

HRESULT
d12_versioned_root_signature_deserializer_create(const void *src, size_t size,
                                                 REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!src || !riid)
      return E_INVALIDARG;
   if (!d12_guid_equal(riid, &IID_ID3D12VersionedRootSignatureDeserializer))
      return E_NOINTERFACE;

   struct d12_versioned_root_signature_deserializer *deserializer =
      calloc(1, sizeof(*deserializer));
   if (!deserializer)
      return E_OUTOFMEMORY;

   HRESULT hr = d12_root_signature_desc_store_init(&deserializer->store,
                                                   src, size);
   if (FAILED(hr)) {
      free(deserializer);
      return hr;
   }

   deserializer->iface.lpVtbl =
      (ID3D12VersionedRootSignatureDeserializerVtbl *)
         &d12_versioned_root_signature_deserializer_vtbl;
   deserializer->refcount = 1;
   *out = &deserializer->iface;
   return S_OK;
}

/*
 * DXIL libraries store named root-signature subobjects in the public RDAT
 * container format.  The record layout below follows Microsoft's
 * RDAT_SubobjectTypes.inl (University of Illinois Open Source License):
 * kind, string-table reference, then a raw-byte offset/size pair for global
 * and local root signatures.  Parse the wire format directly rather than
 * importing DXC's C++ runtime-reflection reader into this C driver.
 */
enum d12_rdat_part_type {
   D12_RDAT_STRING_BUFFER = 1,
   D12_RDAT_RAW_BYTES = 5,
   D12_RDAT_SUBOBJECT_TABLE = 6,
};

enum d12_rdat_subobject_kind {
   D12_RDAT_GLOBAL_ROOT_SIGNATURE = 1,
   D12_RDAT_LOCAL_ROOT_SIGNATURE = 2,
};

static bool
d12_rdat_name_equal(const uint8_t *string, size_t available,
                    LPCWSTR requested_name)
{
   /* DirectX-Headers uses native wchar_t on Linux even though the D3D ABI is
    * UTF-16.  Test and application callers pass 16-bit wire units. */
   const uint16_t *requested = (const uint16_t *)requested_name;
   size_t offset = 0;

   for (;;) {
      uint32_t cp = *requested++;
      if (cp == 0)
         return offset < available && string[offset] == 0;

      if (cp >= 0xd800 && cp <= 0xdbff) {
         uint32_t low = *requested++;
         if (low < 0xdc00 || low > 0xdfff)
            return false;
         cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
      } else if (cp >= 0xdc00 && cp <= 0xdfff) {
         return false;
      }

      uint8_t encoded[4];
      unsigned length;
      if (cp < 0x80) {
         encoded[0] = cp;
         length = 1;
      } else if (cp < 0x800) {
         encoded[0] = 0xc0 | (cp >> 6);
         encoded[1] = 0x80 | (cp & 0x3f);
         length = 2;
      } else if (cp < 0x10000) {
         encoded[0] = 0xe0 | (cp >> 12);
         encoded[1] = 0x80 | ((cp >> 6) & 0x3f);
         encoded[2] = 0x80 | (cp & 0x3f);
         length = 3;
      } else {
         encoded[0] = 0xf0 | (cp >> 18);
         encoded[1] = 0x80 | ((cp >> 12) & 0x3f);
         encoded[2] = 0x80 | ((cp >> 6) & 0x3f);
         encoded[3] = 0x80 | (cp & 0x3f);
         length = 4;
      }

      if (offset > available || length > available - offset ||
          memcmp(string + offset, encoded, length))
         return false;
      offset += length;
   }
}

HRESULT
d12_versioned_root_signature_deserializer_create_from_library(
   const void *src, size_t size, LPCWSTR subobject_name, REFIID riid,
   void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!src || !subobject_name || !riid)
      return E_INVALIDARG;

   const uint8_t *rdat;
   size_t rdat_size;
   if (!find_dxbc_part(src, size, D12_TAG_RDAT, &rdat, &rdat_size) ||
       rdat_size < 8 || r_u32(rdat) != 0x10)
      return E_INVALIDARG;

   uint32_t part_count = r_u32(rdat + 4);
   if (part_count > 64 || (size_t)part_count * 4 > rdat_size - 8)
      return E_INVALIDARG;

   const uint8_t *strings = NULL, *raw = NULL, *subobjects = NULL;
   size_t strings_size = 0, raw_size = 0, subobjects_size = 0;

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t offset = r_u32(rdat + 8 + (size_t)i * 4);
      if ((size_t)offset + 8 > rdat_size)
         return E_INVALIDARG;

      uint32_t type = r_u32(rdat + offset);
      uint32_t part_size = r_u32(rdat + offset + 4);
      if ((size_t)part_size > rdat_size - ((size_t)offset + 8))
         return E_INVALIDARG;

      const uint8_t *part = rdat + offset + 8;
      if (type == D12_RDAT_STRING_BUFFER) {
         strings = part;
         strings_size = part_size;
      } else if (type == D12_RDAT_RAW_BYTES) {
         raw = part;
         raw_size = part_size;
      } else if (type == D12_RDAT_SUBOBJECT_TABLE) {
         subobjects = part;
         subobjects_size = part_size;
      }
   }

   if (!strings || !raw || !subobjects || subobjects_size < 8)
      return E_INVALIDARG;

   uint32_t record_count = r_u32(subobjects);
   uint32_t record_stride = r_u32(subobjects + 4);
   if (record_stride < 16 ||
       (record_count &&
        (size_t)record_count > (subobjects_size - 8) / record_stride))
      return E_INVALIDARG;

   for (uint32_t i = 0; i < record_count; i++) {
      const uint8_t *record =
         subobjects + 8 + (size_t)i * record_stride;
      uint32_t kind = r_u32(record);
      uint32_t name_offset = r_u32(record + 4);
      uint32_t data_offset = r_u32(record + 8);
      uint32_t data_size = r_u32(record + 12);

      if (kind != D12_RDAT_GLOBAL_ROOT_SIGNATURE &&
          kind != D12_RDAT_LOCAL_ROOT_SIGNATURE)
         continue;
      if (name_offset >= strings_size ||
          !d12_rdat_name_equal(strings + name_offset,
                               strings_size - name_offset, subobject_name))
         continue;
      if (data_offset > raw_size || data_size > raw_size - data_offset)
         return E_INVALIDARG;

      /* RDAT's RootSignature_t stores the RTS0 payload, not a nested DXBC
       * container.  Reuse the normal deserializer by supplying the wrapper
       * it intentionally expects. */
      if (data_size > UINT32_MAX - 44)
         return E_INVALIDARG;
      struct d12_writer payload = {
         .data = (uint8_t *)raw + data_offset,
         .size = data_size,
         .used = data_size,
      };
      void *wrapped = NULL;
      size_t wrapped_size = 0;
      HRESULT hr = wrap_rts0(&payload, &wrapped, &wrapped_size);
      if (FAILED(hr))
         return hr;
      hr = d12_versioned_root_signature_deserializer_create(
         wrapped, wrapped_size, riid, out);
      free(wrapped);
      return hr;
   }

   return E_INVALIDARG;
}

/*
 * The resource class a root descriptor parameter binds.
 */
unsigned
d12_root_param_class(const struct d12_root_param *rp)
{
   switch (rp->type) {
   case D3D12_ROOT_PARAMETER_TYPE_SRV:
      return D12_RES_SRV;
   case D3D12_ROOT_PARAMETER_TYPE_UAV:
      return D12_RES_UAV;
   case D3D12_ROOT_PARAMETER_TYPE_CBV:
   case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
      return D12_RES_CBV;
   default:
      return ~0u;
   }
}

unsigned
d12_descriptor_range_class(D3D12_DESCRIPTOR_RANGE_TYPE type)
{
   switch (type) {
   case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
      return D12_RES_SRV;
   case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
      return D12_RES_UAV;
   case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
      return D12_RES_CBV;
   case D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER:
      return D12_RES_SAMPLER;
   default:
      return ~0u;
   }
}

/*
 * ID3DBlob, the container D3D12 hands serialized root signatures back in.
 * Five methods and a byte array; the payload trails the object so there is
 * one allocation and one free.
 */
struct d12_blob {
   ID3DBlob iface; /* must be first */
   uint32_t refcount;
   size_t size;
   uint8_t data[];
};

static HRESULT STDMETHODCALLTYPE
d12_blob_QueryInterface(ID3DBlob *iface, REFIID riid, void **out)
{
   struct d12_blob *blob = (struct d12_blob *)iface;

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D10Blob)) {
      p_atomic_inc(&blob->refcount);
      *out = iface;
      return S_OK;
   }

   d12_log_unknown_riid("ID3DBlob::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_blob_AddRef(ID3DBlob *iface)
{
   struct d12_blob *blob = (struct d12_blob *)iface;
   return p_atomic_inc_return(&blob->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_blob_Release(ID3DBlob *iface)
{
   struct d12_blob *blob = (struct d12_blob *)iface;
   uint32_t refs = p_atomic_dec_return(&blob->refcount);

   if (refs == 0)
      free(blob);

   return refs;
}

static LPVOID STDMETHODCALLTYPE
d12_blob_GetBufferPointer(ID3DBlob *iface)
{
   struct d12_blob *blob = (struct d12_blob *)iface;
   return blob->data;
}

static SIZE_T STDMETHODCALLTYPE
d12_blob_GetBufferSize(ID3DBlob *iface)
{
   struct d12_blob *blob = (struct d12_blob *)iface;
   return blob->size;
}

static const ID3D10BlobVtbl d12_blob_vtbl = {
   .QueryInterface = d12_blob_QueryInterface,
   .AddRef = d12_blob_AddRef,
   .Release = d12_blob_Release,
   .GetBufferPointer = d12_blob_GetBufferPointer,
   .GetBufferSize = d12_blob_GetBufferSize,
};

ID3DBlob *
d12_blob_create(const void *data, size_t size)
{
   struct d12_blob *blob = malloc(sizeof(*blob) + size);
   if (!blob)
      return NULL;

   blob->iface.lpVtbl = (ID3D10BlobVtbl *)&d12_blob_vtbl;
   blob->refcount = 1;
   blob->size = size;
   memcpy(blob->data, data, size);
   return &blob->iface;
}

/*
 * Assign a binding slot to every resource the signature can bind, in root
 * parameter order. This is the assignment BOTH sides use: shaders are
 * compiled against it and the command list fills the table through it, so
 * it has to be a property of the signature rather than something either
 * side works out for itself.
 *
 * Root constants get a slot too. They appear to a shader as a constant
 * buffer at their register, so they need somewhere for their address to go
 * once the values are uploaded at draw time.
 */
static bool
d12_root_layout_build_map_rooting_at(const struct d12_root_layout *layout,
                                     struct d12_binding_map *out,
                                     uint32_t rooting_threshold)
{
   memset(out, 0, sizeof(*out));
   out->visibility_filter = UINT32_MAX;
   unsigned root_range = 0;

   for (uint32_t i = 0; i < layout->num_params; i++) {
      const struct d12_root_param *rp = &layout->params[i];

      switch (rp->type) {
      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         d12_binding_map_add_range_visible(
            out, D12_RES_CBV, rp->space, rp->base_register, root_range++,
            rp->visibility);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_SRV:
         d12_binding_map_add_range_visible(
            out, D12_RES_SRV, rp->space, rp->base_register, root_range++,
            rp->visibility);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         d12_binding_map_add_range_visible(
            out, D12_RES_UAV, rp->space, rp->base_register, root_range++,
            rp->visibility);
         break;

      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE: {
         const struct d12_root_range *ranges =
            d12_root_param_ranges(layout, rp);
         for (uint32_t r = 0; r < rp->num_ranges; r++) {
            const struct d12_root_range *range = &ranges[r];
            unsigned klass = d12_descriptor_range_class(range->type);
            unsigned this_range = root_range++;
            if (klass >= D12_RES_CLASS_COUNT)
               continue;

            /* An unbounded or oversized range consumes one root slot holding
             * the GPU address of its first descriptor. The shader indexes
             * the coherent descriptor heap from there; enumerating 64K or
             * UINT_MAX registers into a 256-slot root table is the category
             * error that previously made these signatures unusable.
             *
             * Carry the sized range's extent. A range the signature sized
             * explicitly ends, and a shader register past its end belongs to
             * a different root parameter -- resolving it against this base
             * would read an unrelated descriptor out of the heap. */
            if (range->num_descriptors == UINT32_MAX ||
                range->num_descriptors >= rooting_threshold) {
               d12_binding_map_add_rooted_visible(
                  out, klass, range->space, range->base_register,
                  this_range, rp->visibility,
                  range->num_descriptors == UINT32_MAX
                     ? 0u
                     : range->num_descriptors);
               continue;
            }

            for (uint32_t d = 0; d < range->num_descriptors; d++) {
               d12_binding_map_add_range_visible(
                  out, klass, range->space, range->base_register + d,
                  this_range, rp->visibility);
            }
         }
         break;
      }

      default:
         break;
      }
   }

   /* Static samplers participate in shader binding lookup but not in the
    * resource root table. d12_binding_map gives them their own dense slot
    * namespace, so t0 and s0 may both be slot 0 as the API intends. */
   for (uint32_t i = 0; i < layout->num_static_samplers; i++) {
      const D3D12_STATIC_SAMPLER_DESC1 *s = &layout->static_samplers[i];
      d12_binding_map_add_range_visible(
         out, D12_RES_SAMPLER, s->RegisterSpace, s->ShaderRegister,
         root_range++, s->ShaderVisibility);
   }

   return !out->overflow;
}

bool
d12_root_layout_build_map(const struct d12_root_layout *layout,
                          struct d12_binding_map *out)
{
   /*
    * A descriptor-table range is normally expanded one dense slot per
    * register, because a dense slot is what lets the shader address the
    * binding directly.  Only a range too large for the table at all is
    * rooted: its slot holds the GPU address of descriptor zero and the
    * shader indexes the heap from there.
    *
    * That default is the fast path and stays untouched.  But a signature
    * whose sized ranges simply do not fit is still legal D3D12, and
    * refusing it is worse than paying one indirection for its largest
    * ranges.  Retry with progressively more aggressive rooting, and only
    * report failure when even one slot per range does not fit.  Whichever
    * map results is the single assignment both the shader translation and
    * the runtime bind read, so the two cannot disagree.
    */
   static const uint32_t thresholds[] = {
      D12_MAX_RESOURCE_BINDINGS, 64, 16, 4, 1,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(thresholds); i++) {
      if (d12_root_layout_build_map_rooting_at(layout, out, thresholds[i]))
         return true;
   }

   /* An overflowing map names fewer bindings than the signature declares.
    * Say so here: the alternative is a shader translation that blames the
    * root signature for not binding a resource the map simply could not
    * hold. */
   mesa_loge("root signature needs more than %u dense binding slots even "
             "with every descriptor range rooted; %u root parameters and "
             "%u static samplers did not fit",
             (unsigned)D12_MAX_RESOURCE_BINDINGS,
             (unsigned)layout->num_params,
             (unsigned)layout->num_static_samplers);
   return false;
}
