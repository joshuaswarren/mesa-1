/*
 * Copyright © 2022 Konstantin Seurer
 * Copyright © 2024 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Push-constant blocks shared between the Honeykrisp acceleration-structure
 * encode shaders and the C code that dispatches them.  Everything the
 * encoder needs travels in push constants; there are no descriptor sets, so
 * a build never disturbs the application's bound descriptors.
 */

#ifndef HK_BVH_BUILD_INTERFACE_H
#define HK_BVH_BUILD_INTERFACE_H

#ifdef VULKAN
#include "vk_build_helpers.h"
#include "hk_bvh.h"

TYPE(hk_bvh_triangle_node, 4);
TYPE(hk_bvh_aabb_node, 4);
TYPE(hk_bvh_instance_node, 8);
TYPE(hk_bvh_box_node, 4);
TYPE(hk_bvh_header, 4);
#else
#include <stdint.h>

#include "hk_bvh.h"

#define REF(type) uint64_t
#define VOID_REF  uint64_t
#endif

struct hk_encode_args {
   /* Base of the common builder's intermediate representation: the packed
    * leaf array, immediately followed by the internal box-node array. */
   VOID_REF intermediate_bvh;
   /* Base of the acceleration structure we are writing. */
   VOID_REF output_bvh;
   REF(vk_ir_header) header;
   /* Byte offset of the internal box-node array inside intermediate_bvh,
    * i.e. leaf_node_count * the IR leaf size. */
   uint32_t root_offset;
   /* Upper bound the dispatch was sized from; the shader still clamps
    * against the counts the builder wrote into the header. */
   uint32_t internal_node_bound;
   uint32_t leaf_node_bound;
   uint32_t geometry_type;
   /* What get_as_size() reserved, recorded into the header so a compacting
    * copy can find it without recomputing. */
   uint32_t as_size;
};

#endif /* HK_BVH_BUILD_INTERFACE_H */
