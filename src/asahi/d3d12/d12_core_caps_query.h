/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include "d12_core_caps.h"
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
HRESULT d12_core_caps_query(const struct d12_core_caps *caps, D3D12_FEATURE feature,
                            void *data, UINT data_size);
struct agx_device;
void d12_core_caps_init_agx(struct d12_core_caps *caps, const struct agx_device *dev);
