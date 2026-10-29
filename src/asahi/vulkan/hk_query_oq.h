/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

/*
 * Native AGX3 occlusion availability.
 *
 * A render's retirement record carries four timestamp slots (vertex start
 * and end, fragment start and end). The G17 kernel writes every requested
 * slot when it retires the render, which is after the end-of-render cache
 * maintenance that makes the render's occlusion counts visible, and before
 * it signals the submission. A nonzero timestamp in a query's 64-bit
 * availability slot is therefore exactly "available" there.
 *
 * That holds only for a kernel that publishes the slots at retirement (see
 * hk_oq_render_stamps_supported()). The G15 firmware writes each slot from
 * its stage's own start or end record: the vertex end lands when tiling
 * finishes, before the fragment work writes the counts. Those chips keep the
 * queued store and the submission wait on readback.
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
/* Whether renders on @chip may carry occlusion availability in their
 * retirement timestamp slots. Only qualified on the G17P kernel; other chips
 * whose kernel writes the slots at retirement can be added once measured. */
bool hk_oq_render_stamps_supported(enum agx_chip chip);

struct hk_oq_stamp {
   uint32_t handle;
   uint32_t offset_B;
   uint32_t render_seq;
};
