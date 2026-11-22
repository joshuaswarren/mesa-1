/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "agx_drm_name.h"

#include <gtest/gtest.h>

TEST(drm_name, native_names_are_admitted)
{
   EXPECT_TRUE(agx_drm_name_is_native("asahi"));
   EXPECT_TRUE(agx_drm_name_is_native("aurora"));
}

TEST(drm_name, other_names_are_rejected)
{
   EXPECT_FALSE(agx_drm_name_is_native(NULL));
   EXPECT_FALSE(agx_drm_name_is_native("virtio_gpu"));
   EXPECT_FALSE(agx_drm_name_is_native("aurora-dri"));
   EXPECT_FALSE(agx_drm_name_is_native(""));
   EXPECT_FALSE(agx_drm_name_is_native("ASDF"));
}
