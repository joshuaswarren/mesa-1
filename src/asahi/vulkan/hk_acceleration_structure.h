/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_ACCELERATION_STRUCTURE_H
#define HK_ACCELERATION_STRUCTURE_H

#include "hk_bvh.h"

#include "vk_acceleration_structure.h"

struct hk_device;

VkResult hk_device_init_accel_struct_state(struct hk_device *device);
void hk_device_finish_accel_struct_state(struct hk_device *device);

#endif /* HK_ACCELERATION_STRUCTURE_H */
