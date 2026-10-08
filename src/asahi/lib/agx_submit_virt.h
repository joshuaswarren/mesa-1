/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdint.h>

/*
 * Optional frontend-to-winsys metadata which does not belong in the public
 * drm_asahi_submit UAPI. Linux and virtio backends may ignore it. The D3DKMT
 * backend consumes a complete read set to publish CPU cache ownership to the
 * WDDM miniport without enumerating every live allocation.
 *
 * GPU addresses are absolute in both the main and USC VA heaps. A frontend
 * may set READ_SET_COMPLETE only when every GPU-read range is present. An
 * absent marker deliberately selects the conservative winsys fallback.
 */
#define AGX_SUBMIT_VIRT_READ_SET_COMPLETE UINT32_C(0x00000001)
#define AGX_SUBMIT_VIRT_VALID_FLAGS AGX_SUBMIT_VIRT_READ_SET_COMPLETE

/* AppleAgxWddm contract v3+ bounds the canonical retained/read vector at
 * this width.  Frontends keep the same bound without including a KMD header,
 * so Linux and portable host builds retain one platform-neutral ABI. */
#define AGX_SUBMIT_MAX_READ_RANGES 256u

#define AGX_SUBMIT_FENCE_WAIT           UINT32_C(0x00000001)
#define AGX_SUBMIT_FENCE_SIGNAL         UINT32_C(0x00000002)
#define AGX_SUBMIT_FENCE_BOTTOM_OF_PIPE UINT32_C(0x00000004)

struct agx_submit_read_range {
   uint64_t gpu_va;
   uint64_t length;
};

/* D3D12 runtime-owned placement fences do not have D3DKMT sync handles.
 * The Windows winsys serializes this pointer-free description into the
 * AppleAgxWddm v4 submit-private record. Other winsys backends ignore it. */
struct agx_submit_fence_dependency {
   uint64_t current_gpu_va;
   uint64_t monitored_gpu_va;
   uint64_t value;
   uint32_t flags;
   uint32_t reserved;
};
