/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct hk_device;
struct hk_cs;
struct hk_rendering_state;
struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;

/* HK_PASS_PROFILE=<path>: per-command GPU timing census.
 *
 * Every submitted render and compute command gets the firmware's start and
 * end timestamps (render: vertex start, fragment end) in a device-owned
 * ring. Completed records are grouped by pass shape (size, attachment
 * formats, load/store ops, compression, position of the stream within its
 * render pass, ZLS word) and appended to <path>.<pid> every two seconds,
 * sorted by the GPU busy time each group adds, with the busy time of the
 * window. Commands that already carry an application timestamp are left
 * alone.
 *
 * Diagnostic only: off unless the variable is set. */

void hk_pass_profile_init(struct hk_device *dev);
void hk_pass_profile_fini(struct hk_device *dev);

/* Record the API-level description of the render pass being begun. */
void hk_pass_profile_describe(struct hk_rendering_state *render,
                              const VkRenderingInfo *info,
                              bool incomplete_render_area);

void hk_pass_profile_render(struct hk_device *dev, const struct hk_cs *cs,
                            struct drm_asahi_cmd_render *c);
void hk_pass_profile_compute(struct hk_device *dev, const struct hk_cs *cs,
                             struct drm_asahi_cmd_compute *c);

/* Fold completed records; called after each submission. */
void hk_pass_profile_harvest(struct hk_device *dev);
