/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

struct nir_shader;

bool hk_nir_widen_16bit_varyings(struct nir_shader *nir);
