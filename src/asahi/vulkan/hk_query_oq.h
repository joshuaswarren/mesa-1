/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdint.h>

/*
 * Native AGX3 occlusion availability.
 *
 * A render's retirement record carries four timestamp slots (vertex start
 * and end, fragment start and end). The kernel writes every requested slot
 * when it retires the render, which is after the end-of-render cache
 * maintenance that makes the render's occlusion counts visible, and before
 * it signals the submission. A nonzero timestamp in a query's 64-bit
 * availability slot is therefore exactly "available".
 *
 * The queued post-graphics store remains the general mechanism: it is
 * recorded as usual, and the stream that would carry it is kept empty until
 * the end of the command buffer. Only then is it known whether any later
 * compute work in this command buffer depends on the render; such work would
 * have to wait for the render's host-side retirement once the render writes
 * timestamps, where it now waits in firmware. When nothing does and the
 * render has free slots, the render carries the availability and the extra
 * compute command is dropped. Otherwise the store is emitted as before.
 */
struct hk_oq_stamp {
   uint32_t handle;
   uint32_t offset_B;
   uint32_t render_seq;
};
