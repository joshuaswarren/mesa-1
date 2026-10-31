/*
 * Copyright 2024 Sergio Lopez
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include "agx_device.h"

int agx_virtio_simple_ioctl(struct agx_device *dev, unsigned cmd, void *_req);

bool agx_virtio_open_device(struct agx_device *dev);

void agx_virtio_bo_mmap(struct agx_device *dev, struct agx_bo *bo,
                        void *fixed_addr, size_t length);
