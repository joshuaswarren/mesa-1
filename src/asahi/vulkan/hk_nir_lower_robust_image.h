/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

struct nir_shader;

bool hk_nir_lower_robust_image_access(struct nir_shader *nir,
                                      bool lower_loads_and_stores);
