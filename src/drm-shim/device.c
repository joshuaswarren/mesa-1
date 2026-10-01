/*
 * Copyright © 2018 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/** @file
 *
 * Implements core GEM support (particularly ioctls) underneath the libc ioctl
 * wrappers, and calls into the driver-specific code as necessary.
 */

#include <c11/threads.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/memfd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "drm-uapi/drm.h"
#include "drm_shim.h"
#include "util/hash_table.h"
#include "util/u_atomic.h"

#define SHIM_MEM_SIZE (4ull * 1024 * 1024 * 1024)

#ifndef HAVE_MEMFD_CREATE
#include <sys/syscall.h>

static inline int
memfd_create(const char *name, unsigned int flags)
{
   return syscall(SYS_memfd_create, name, flags);
}
#endif

/* Global state for the shim shared between libc, core, and driver. */
struct shim_device shim_device;

long shim_page_size;
static bool failed_prime_export_once;
static bool failed_dmabuf_export_once;
static bool failed_dmabuf_import_once;
static unsigned dmabuf_fallback_sync_calls;
static bool waited_after_dmabuf_import_failure;
static unsigned prime_export_calls;
static bool timed_out_sync_wait_once;
static unsigned queue_progress_syncobj_creates;
static unsigned queue_progress_syncobj_destroys;
static unsigned queue_progress_timeline_signals;
static unsigned queue_progress_transfers;
static uint64_t queue_progress_created_handles;
static uint64_t queue_progress_destroyed_handles;
static uint32_t queue_progress_initial_handle;
static uint64_t queue_progress_initial_point;
static uint32_t queue_progress_transfer_src_handle;
static uint32_t queue_progress_transfer_dst_handle;
static uint64_t queue_progress_transfer_src_point;
static uint64_t queue_progress_transfer_dst_point;
static uint32_t queue_progress_transfer_flags;
static bool queue_progress_signal_shape_valid;
static bool queue_progress_transfer_shape_valid;
static bool failed_queue_timeline_signal_once;
static unsigned wsi_rollback_syncobj_creates;
static unsigned wsi_rollback_syncobj_destroys;
static unsigned wsi_rollback_sync_imports;
static unsigned wsi_rollback_transfers;
static uint64_t wsi_rollback_created_handles;
static uint64_t wsi_rollback_destroyed_handles;
static uint32_t wsi_rollback_success_handle;
static uint32_t wsi_rollback_failed_handle;
static bool wsi_rollback_import_failed;
static bool wsi_rollback_cleanup_before_marker;
static bool wsi_rollback_marker_seen;
static unsigned wsi_rollback_dmabuf_exports;
static unsigned wsi_rollback_dmabuf_imports;
static unsigned wsi_rollback_failed_dmabuf_import;
static unsigned wsi_rollback_sync_file_exports;
static unsigned wsi_rollback_failed_sync_file_export;
static unsigned wsi_rollback_waits_after_sync_failure;
static unsigned wsi_rollback_syncobj_resets_after_failure;
static unsigned wsi_rollback_failed_transfer;
static uint32_t wsi_rollback_fallback_wait_handle;
static uint32_t wsi_rollback_fallback_wait_flags;
static uint32_t wsi_rollback_failed_sync_file_export_handle;
static uint32_t wsi_rollback_first_sync_file_export_handle;
static uint32_t wsi_rollback_first_reset_handle;
static uint32_t wsi_rollback_partial_signal_handle;
static uint32_t wsi_rollback_failed_transfer_handle;
static uint64_t wsi_rollback_submit_reset_handles;
static bool wsi_rollback_dmabuf_import_failed;
static bool wsi_rollback_sync_file_export_failed;
static bool wsi_rollback_fallback_wait_shape_valid;
static bool wsi_rollback_first_reset_shape_valid;
static bool wsi_rollback_submit_transfer_failed;
static bool wsi_rollback_submit_cleanup_before_retry;
static bool wsi_rollback_submit_retry_marker_seen;
static bool wsi_rollback_stale_fence_wait_rejected;

#define TIMELINE_HOST_MAX_SYNCOBJS 4096
static struct {
   bool alive;
   uint64_t value;
} timeline_host_syncobjs[TIMELINE_HOST_MAX_SYNCOBJS];
static uint32_t timeline_host_next_handle = 1;

static bool
timeline_host_values_enabled(void)
{
   return getenv("ASAHI_SHIM_TIMELINE_HOST_VALUES") != NULL;
}

static bool
timeline_host_handle_valid(uint32_t handle)
{
   return handle > 0 && handle < TIMELINE_HOST_MAX_SYNCOBJS &&
          timeline_host_syncobjs[handle].alive;
}

static const char *
queue_progress_lifecycle_case(void)
{
   return getenv("ASAHI_SHIM_EXPECT_QUEUE_PROGRESS_LIFECYCLE");
}

static const char *
wsi_rollback_case(void)
{
   const char *test_case = getenv("ASAHI_SHIM_EXPECT_ACQUIRE_ROLLBACK");
   return test_case ? test_case :
      getenv("ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK");
}

static void __attribute__((destructor))
check_injected_sync_failures(void)
{
   if (getenv("ASAHI_SHIM_EXPECT_PRIME_EXPORT_FAILURE") &&
       !failed_prime_export_once) {
      fprintf(stderr, "shim: expected PRIME export failure was not injected\n");
      _Exit(EXIT_FAILURE);
   }

   if (getenv("ASAHI_SHIM_EXPECT_DMABUF_SYNC_FAILURES") &&
       (!failed_dmabuf_export_once || !failed_dmabuf_import_once ||
        dmabuf_fallback_sync_calls != 2 ||
        !waited_after_dmabuf_import_failure)) {
      fprintf(stderr,
              "shim: dma-buf sync fallback incomplete: export=%u import=%u "
              "cpu_sync=%u out_wait=%u\n",
              failed_dmabuf_export_once, failed_dmabuf_import_once,
              dmabuf_fallback_sync_calls, waited_after_dmabuf_import_failure);
      _Exit(EXIT_FAILURE);
   }

   if (getenv("ASAHI_SHIM_EXPECT_SINGLE_PRIME_EXPORT") &&
       prime_export_calls != 1) {
      fprintf(stderr, "shim: observed %u PRIME exports for one BO\n",
              prime_export_calls);
      _Exit(EXIT_FAILURE);
   }

   if (getenv("ASAHI_SHIM_EXPECT_SYNC_WAIT_TIMEOUT") &&
       !timed_out_sync_wait_once) {
      fprintf(stderr, "shim: expected syncobj wait timeout was not injected\n");
      _Exit(EXIT_FAILURE);
   }

   if (failed_prime_export_once)
      fprintf(stderr, "shim: injected PRIME export failure exactly once\n");
   if (failed_dmabuf_export_once && failed_dmabuf_import_once)
      fprintf(stderr, "shim: injected dma-buf sync failures exactly once\n");
   if (getenv("ASAHI_SHIM_EXPECT_SINGLE_PRIME_EXPORT"))
      fprintf(stderr,
              "shim: one concurrent BO transition made one PRIME export\n");
   if (timed_out_sync_wait_once)
      fprintf(stderr, "shim: injected syncobj wait timeout exactly once\n");
}

static void __attribute__((destructor))
check_queue_progress_lifecycle(void)
{
   const char *test_case = queue_progress_lifecycle_case();
   if (!test_case)
      return;

   const bool success = !strcmp(test_case, "success");
   const bool init_failure = !strcmp(test_case, "init-failure");
   const bool submit_failure = !strcmp(test_case, "submit-failure");
   /* One create/destroy pair probes the physical-device syncobj features.
    * The logical-device cases then add the queue timeline, plus the fence
    * and QueueWaitIdle's temporary syncobj where those objects are reached. */
   const unsigned expected_syncobjs = success ? 4 : init_failure ? 2 : 3;
   const unsigned expected_transfers = success ? 1 : 0;
   const uint64_t expected_handles =
      (UINT64_C(1) << (expected_syncobjs + 1)) - 2;
   const bool exact =
      (success || init_failure || submit_failure) &&
      queue_progress_syncobj_creates == expected_syncobjs &&
      queue_progress_syncobj_destroys == expected_syncobjs &&
      queue_progress_created_handles == expected_handles &&
      queue_progress_destroyed_handles == expected_handles &&
      queue_progress_timeline_signals == 1 &&
      queue_progress_signal_shape_valid &&
      queue_progress_initial_handle == 2 &&
      queue_progress_initial_point == 1 &&
      queue_progress_transfers == expected_transfers &&
      failed_queue_timeline_signal_once == init_failure &&
      (!success ||
       (queue_progress_transfer_shape_valid &&
        queue_progress_transfer_src_handle == 2 &&
        queue_progress_transfer_dst_handle == 4 &&
        queue_progress_transfer_src_point == 2 &&
        queue_progress_transfer_dst_point == 0 &&
        queue_progress_transfer_flags == 0));

   if (!exact) {
      fprintf(stderr,
              "shim: bad queue progress lifecycle '%s': create=%u "
              "destroy=%u signal=%u shape=%u initial=%llu transfer=%u "
              "handles=%#llx/%#llx initial-handle=%u "
              "transfer-shape=%u src=%u:%llu dst=%u:%llu flags=%#x "
              "signal-failed=%u\n",
              test_case, queue_progress_syncobj_creates,
              queue_progress_syncobj_destroys,
              queue_progress_timeline_signals,
              queue_progress_signal_shape_valid,
              (unsigned long long)queue_progress_initial_point,
              queue_progress_transfers,
              (unsigned long long)queue_progress_created_handles,
              (unsigned long long)queue_progress_destroyed_handles,
              queue_progress_initial_handle,
              queue_progress_transfer_shape_valid,
              queue_progress_transfer_src_handle,
              (unsigned long long)queue_progress_transfer_src_point,
              queue_progress_transfer_dst_handle,
              (unsigned long long)queue_progress_transfer_dst_point,
              queue_progress_transfer_flags,
              failed_queue_timeline_signal_once);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: exact queue progress lifecycle '%s': bootstrap=1 "
           "wait-idle-source=%llu syncobjs=%u/%u\n",
           test_case,
           (unsigned long long)queue_progress_transfer_src_point,
           queue_progress_syncobj_creates,
           queue_progress_syncobj_destroys);
}

static void __attribute__((destructor))
check_wsi_rollback_lifecycle(void)
{
   const char *test_case = wsi_rollback_case();
   if (!test_case)
      return;

   const bool semaphore_failure = !strcmp(test_case, "semaphore-failure");
   const bool fence_failure = !strcmp(test_case, "fence-failure");
   const bool present_import_failure =
      !strcmp(test_case, "present-import-failure");
   const bool present_export_failure =
      !strcmp(test_case, "present-export-failure");
   const bool present_submit_failure =
      !strcmp(test_case, "present-submit-failure");
   const bool empty_submit_failure =
      !strcmp(test_case, "empty-submit-failure");
   const bool release_failure = !strcmp(test_case, "release-failure");
   const bool present_priority = !strcmp(test_case, "present-priority");
   const bool wayland_ownership = !strcmp(test_case, "wayland-ownership");
   const bool wayland_present_allocation =
      !strcmp(test_case, "wayland-present-allocation");
   const bool wayland_surface_init_retry =
      !strcmp(test_case, "wayland-surface-init-retry");
   const bool wayland_present_wrapper_retry =
      !strcmp(test_case, "wayland-present-wrapper-retry");
   const bool wayland_present_queue_retry =
      !strcmp(test_case, "wayland-present-queue-retry");
   const bool wayland_presentation_wrapper_retry =
      !strcmp(test_case, "wayland-presentation-wrapper-retry");
   const unsigned expected_imports = semaphore_failure ? 3 : 4;
   const unsigned expected_syncobjs = semaphore_failure ? 12 : 13;
   const uint64_t expected_handles =
      (UINT64_C(1) << (expected_syncobjs + 1)) - 2;
   const bool exact_acquire =
      (semaphore_failure || fence_failure) &&
      wsi_rollback_syncobj_creates == expected_syncobjs &&
      wsi_rollback_syncobj_destroys == expected_syncobjs &&
      wsi_rollback_created_handles == expected_handles &&
      wsi_rollback_destroyed_handles == expected_handles &&
      wsi_rollback_sync_imports == expected_imports &&
      wsi_rollback_transfers == 5 &&
      wsi_rollback_import_failed &&
      wsi_rollback_marker_seen &&
      wsi_rollback_cleanup_before_marker &&
      wsi_rollback_failed_handle != 0 &&
      (!fence_failure ||
       (wsi_rollback_success_handle != 0 &&
        wsi_rollback_success_handle != wsi_rollback_failed_handle));
   const bool exact_present =
      (present_import_failure || present_export_failure) &&
      wsi_rollback_syncobj_creates == 12 &&
      wsi_rollback_syncobj_destroys == 12 &&
      wsi_rollback_created_handles == UINT64_C(0x1ffe) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x1ffe) &&
      wsi_rollback_sync_imports == 3 &&
      wsi_rollback_transfers == 13 &&
      wsi_rollback_dmabuf_exports == 4 &&
      wsi_rollback_dmabuf_imports ==
         (present_import_failure ? 4 : 3) &&
      wsi_rollback_sync_file_exports == 3 &&
      wsi_rollback_dmabuf_import_failed == present_import_failure &&
      wsi_rollback_failed_dmabuf_import ==
         (present_import_failure ? 2 : 0) &&
      wsi_rollback_sync_file_export_failed == present_export_failure &&
      wsi_rollback_failed_sync_file_export ==
         (present_export_failure ? 1 : 0) &&
      wsi_rollback_waits_after_sync_failure == 3 &&
      wsi_rollback_fallback_wait_shape_valid &&
      wsi_rollback_fallback_wait_handle == 8 &&
      wsi_rollback_fallback_wait_flags ==
         (DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
          DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT) &&
      wsi_rollback_syncobj_resets_after_failure == 6 &&
      wsi_rollback_first_reset_shape_valid &&
      wsi_rollback_first_sync_file_export_handle == 3 &&
      wsi_rollback_first_reset_handle ==
         wsi_rollback_first_sync_file_export_handle &&
      (!present_export_failure ||
       wsi_rollback_failed_sync_file_export_handle ==
          wsi_rollback_first_sync_file_export_handle);
   const bool exact_submit =
      present_submit_failure &&
      wsi_rollback_syncobj_creates == 13 &&
      wsi_rollback_syncobj_destroys == 13 &&
      wsi_rollback_created_handles == UINT64_C(0x3ffe) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x3ffe) &&
      wsi_rollback_sync_imports == 3 &&
      wsi_rollback_transfers == 15 &&
      wsi_rollback_dmabuf_exports == 4 &&
      wsi_rollback_dmabuf_imports == 4 &&
      !wsi_rollback_dmabuf_import_failed &&
      wsi_rollback_failed_dmabuf_import == 0 &&
      wsi_rollback_sync_file_exports == 3 &&
      wsi_rollback_first_sync_file_export_handle == 3 &&
      !wsi_rollback_sync_file_export_failed &&
      wsi_rollback_failed_sync_file_export == 0 &&
      wsi_rollback_waits_after_sync_failure == 2 &&
      wsi_rollback_fallback_wait_shape_valid &&
      wsi_rollback_fallback_wait_handle == 9 &&
      wsi_rollback_fallback_wait_flags ==
         (DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
          DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT) &&
      wsi_rollback_syncobj_resets_after_failure == 8 &&
      wsi_rollback_first_reset_shape_valid &&
      wsi_rollback_first_reset_handle == 3 &&
      wsi_rollback_submit_transfer_failed &&
      wsi_rollback_failed_transfer == 4 &&
      wsi_rollback_partial_signal_handle == 3 &&
      wsi_rollback_failed_transfer_handle == 8 &&
      wsi_rollback_submit_reset_handles == UINT64_C(0x228) &&
      wsi_rollback_submit_retry_marker_seen &&
      wsi_rollback_submit_cleanup_before_retry &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact_empty_submit =
      empty_submit_failure &&
      wsi_rollback_syncobj_creates == 6 &&
      wsi_rollback_syncobj_destroys == 6 &&
      wsi_rollback_created_handles == UINT64_C(0x7e) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x7e) &&
      wsi_rollback_sync_imports == 1 &&
      wsi_rollback_transfers == 2 &&
      wsi_rollback_dmabuf_exports == 2 &&
      wsi_rollback_dmabuf_imports == 1 &&
      wsi_rollback_sync_file_exports == 0 &&
      wsi_rollback_waits_after_sync_failure == 0 &&
      wsi_rollback_syncobj_resets_after_failure == 0 &&
      wsi_rollback_submit_transfer_failed &&
      wsi_rollback_failed_transfer == 2 &&
      wsi_rollback_partial_signal_handle == 2 &&
      wsi_rollback_failed_transfer_handle == 5 &&
      !wsi_rollback_submit_retry_marker_seen &&
      !wsi_rollback_submit_cleanup_before_retry &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact_release =
      release_failure &&
      wsi_rollback_syncobj_creates == 7 &&
      wsi_rollback_syncobj_destroys == 7 &&
      wsi_rollback_created_handles == UINT64_C(0xfe) &&
      wsi_rollback_destroyed_handles == UINT64_C(0xfe) &&
      wsi_rollback_sync_imports == 2 &&
      wsi_rollback_transfers == 0 &&
      wsi_rollback_dmabuf_exports == 3 &&
      wsi_rollback_dmabuf_imports == 1 &&
      !wsi_rollback_dmabuf_import_failed &&
      wsi_rollback_failed_dmabuf_import == 0 &&
      wsi_rollback_sync_file_exports == 0 &&
      !wsi_rollback_sync_file_export_failed &&
      wsi_rollback_failed_sync_file_export == 0 &&
      wsi_rollback_waits_after_sync_failure == 0 &&
      wsi_rollback_syncobj_resets_after_failure == 0 &&
      wsi_rollback_submit_reset_handles == 0 &&
      !wsi_rollback_import_failed &&
      !wsi_rollback_submit_transfer_failed &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact_present_priority =
      present_priority &&
      wsi_rollback_syncobj_creates == 32 &&
      wsi_rollback_syncobj_destroys == 32 &&
      wsi_rollback_created_handles == UINT64_C(0x1fffffffe) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x1fffffffe) &&
      wsi_rollback_sync_imports == 16 &&
      wsi_rollback_transfers == 57 &&
      wsi_rollback_dmabuf_exports == 17 &&
      wsi_rollback_dmabuf_imports == 17 &&
      !wsi_rollback_dmabuf_import_failed &&
      wsi_rollback_failed_dmabuf_import == 0 &&
      wsi_rollback_sync_file_exports == 16 &&
      !wsi_rollback_sync_file_export_failed &&
      wsi_rollback_failed_sync_file_export == 0 &&
      wsi_rollback_waits_after_sync_failure == 0 &&
      wsi_rollback_syncobj_resets_after_failure == 0 &&
      wsi_rollback_submit_reset_handles == 0 &&
      !wsi_rollback_import_failed &&
      !wsi_rollback_submit_transfer_failed &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact_wayland_ownership =
      (wayland_ownership || wayland_surface_init_retry ||
       wayland_present_wrapper_retry || wayland_present_queue_retry ||
       wayland_presentation_wrapper_retry) &&
      wsi_rollback_syncobj_creates == 2 &&
      wsi_rollback_syncobj_destroys == 2 &&
      wsi_rollback_created_handles == UINT64_C(0x6) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x6) &&
      wsi_rollback_sync_imports == 0 &&
      wsi_rollback_transfers == 0 &&
      wsi_rollback_dmabuf_exports == 0 &&
      wsi_rollback_dmabuf_imports == 0 &&
      !wsi_rollback_dmabuf_import_failed &&
      wsi_rollback_failed_dmabuf_import == 0 &&
      wsi_rollback_sync_file_exports == 0 &&
      !wsi_rollback_sync_file_export_failed &&
      wsi_rollback_failed_sync_file_export == 0 &&
      wsi_rollback_waits_after_sync_failure == 0 &&
      wsi_rollback_syncobj_resets_after_failure == 0 &&
      wsi_rollback_submit_reset_handles == 0 &&
      !wsi_rollback_import_failed &&
      !wsi_rollback_submit_transfer_failed &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact_wayland_present_allocation =
      wayland_present_allocation &&
      wsi_rollback_syncobj_creates == 10 &&
      wsi_rollback_syncobj_destroys == 10 &&
      wsi_rollback_created_handles == UINT64_C(0x7fe) &&
      wsi_rollback_destroyed_handles == UINT64_C(0x7fe) &&
      wsi_rollback_sync_imports == 0 &&
      wsi_rollback_transfers == 9 &&
      wsi_rollback_dmabuf_exports == 0 &&
      wsi_rollback_dmabuf_imports == 0 &&
      !wsi_rollback_dmabuf_import_failed &&
      wsi_rollback_failed_dmabuf_import == 0 &&
      wsi_rollback_sync_file_exports == 0 &&
      !wsi_rollback_sync_file_export_failed &&
      wsi_rollback_failed_sync_file_export == 0 &&
      wsi_rollback_waits_after_sync_failure == 0 &&
      wsi_rollback_syncobj_resets_after_failure == 0 &&
      wsi_rollback_submit_reset_handles == 0 &&
      !wsi_rollback_import_failed &&
      !wsi_rollback_submit_transfer_failed &&
      !wsi_rollback_stale_fence_wait_rejected;
   const bool exact = exact_acquire || exact_present || exact_submit ||
                      exact_empty_submit || exact_release ||
                      exact_present_priority || exact_wayland_ownership ||
                      exact_wayland_present_allocation;

   if (!exact) {
      fprintf(stderr,
              "shim: bad WSI rollback '%s': syncobjs=%u/%u "
              "handles=%#llx/%#llx imports=%u failed=%u success-h=%u "
              "failed-h=%u marker=%u clean-before-marker=%u transfers=%u "
              "dmabuf=%u/%u dmabuf-failed=%u@%u sync-exports=%u "
              "sync-export-first=%u failed=%u@%u:%u fallback-waits=%u "
              "wait=%u/%#x resets=%u first-reset=%u/%u "
              "submit-fail=%u@%u partial=%u failed=%u retry=%u/%u "
              "stale-wait=%u reset-mask=%#llx\n",
              test_case, wsi_rollback_syncobj_creates,
              wsi_rollback_syncobj_destroys,
              (unsigned long long)wsi_rollback_created_handles,
              (unsigned long long)wsi_rollback_destroyed_handles,
              wsi_rollback_sync_imports,
              wsi_rollback_import_failed,
              wsi_rollback_success_handle,
              wsi_rollback_failed_handle,
              wsi_rollback_marker_seen,
              wsi_rollback_cleanup_before_marker,
              wsi_rollback_transfers,
              wsi_rollback_dmabuf_exports,
              wsi_rollback_dmabuf_imports,
              wsi_rollback_dmabuf_import_failed,
              wsi_rollback_failed_dmabuf_import,
              wsi_rollback_sync_file_exports,
              wsi_rollback_first_sync_file_export_handle,
              wsi_rollback_sync_file_export_failed,
              wsi_rollback_failed_sync_file_export,
              wsi_rollback_failed_sync_file_export_handle,
              wsi_rollback_waits_after_sync_failure,
              wsi_rollback_fallback_wait_handle,
              wsi_rollback_fallback_wait_flags,
              wsi_rollback_syncobj_resets_after_failure,
              wsi_rollback_first_reset_shape_valid,
              wsi_rollback_first_reset_handle,
              wsi_rollback_submit_transfer_failed,
              wsi_rollback_failed_transfer,
              wsi_rollback_partial_signal_handle,
              wsi_rollback_failed_transfer_handle,
              wsi_rollback_submit_retry_marker_seen,
              wsi_rollback_submit_cleanup_before_retry,
              wsi_rollback_stale_fence_wait_rejected,
              (unsigned long long)wsi_rollback_submit_reset_handles);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: exact WSI rollback '%s': syncobjs=%u/%u imports=%u "
           "clean-before-marker=%u transfers=%u dmabuf=%u/%u@%u "
           "sync-exports=%u fallback-waits=%u wait=%u/%#x resets=%u "
           "submit=%u@%u partial=%u failed=%u retry=%u/%u "
           "reset-mask=%#llx\n",
           test_case, wsi_rollback_syncobj_creates,
           wsi_rollback_syncobj_destroys,
           wsi_rollback_sync_imports,
           wsi_rollback_cleanup_before_marker,
           wsi_rollback_transfers,
           wsi_rollback_dmabuf_exports,
           wsi_rollback_dmabuf_imports,
           wsi_rollback_failed_dmabuf_import,
           wsi_rollback_sync_file_exports,
           wsi_rollback_waits_after_sync_failure,
           wsi_rollback_fallback_wait_handle,
           wsi_rollback_fallback_wait_flags,
           wsi_rollback_syncobj_resets_after_failure,
           wsi_rollback_submit_transfer_failed,
           wsi_rollback_failed_transfer,
           wsi_rollback_partial_signal_handle,
           wsi_rollback_failed_transfer_handle,
           wsi_rollback_submit_retry_marker_seen,
           wsi_rollback_submit_cleanup_before_retry,
           (unsigned long long)wsi_rollback_submit_reset_handles);
}

static uint32_t
uint_key_hash(const void *key)
{
   return (uintptr_t)key;
}

static bool
uint_key_compare(const void *a, const void *b)
{
   return a == b;
}

/**
 * Called when the first libc shim is called, to initialize GEM simulation
 * state (other than the shims themselves).
 */
void
drm_shim_device_init(void)
{
   shim_device.fd_map = _mesa_hash_table_create(NULL,
                                                uint_key_hash,
                                                uint_key_compare);
   shim_device.prime_fd_map = _mesa_hash_table_create(NULL,
                                                      uint_key_hash,
                                                      uint_key_compare);
   mtx_init(&shim_device.prime_fd_lock, mtx_plain);

   shim_device.offset_map = _mesa_hash_table_u64_create(NULL);

   mtx_init(&shim_device.mem_lock, mtx_plain);

   shim_device.mem_fd = memfd_create("shim mem", MFD_CLOEXEC);
   assert(shim_device.mem_fd != -1);

   ASSERTED int ret = ftruncate(shim_device.mem_fd, SHIM_MEM_SIZE);
   assert(ret == 0);

   /* The man page for mmap() says
    *
    *    offset must be a multiple of the page size as returned by
    *    sysconf(_SC_PAGE_SIZE).
    *
    * Depending on the configuration of the kernel, this may not be 4096. Get
    * this page size once and use it as the page size throughout, ensuring that
    * are offsets are page-size aligned as required. Otherwise, mmap will fail
    * with EINVAL.
    */

   shim_page_size = sysconf(_SC_PAGESIZE);

   util_vma_heap_init(&shim_device.mem_heap, shim_page_size,
                      SHIM_MEM_SIZE - shim_page_size);

   drm_shim_driver_init();
}

static struct shim_fd *
drm_shim_file_create(int fd)
{
   struct shim_fd *shim_fd = calloc(1, sizeof(*shim_fd));

   shim_fd->fd = fd;
   p_atomic_set(&shim_fd->refcount, 1);
   mtx_init(&shim_fd->handle_lock, mtx_plain);
   shim_fd->handles = _mesa_hash_table_create(NULL,
                                              uint_key_hash,
                                              uint_key_compare);

   return shim_fd;
}

/**
 * Called when the libc shims have interposed an open or dup of our simulated
 * DRM device.
 */
void drm_shim_fd_register(int fd, struct shim_fd *shim_fd)
{
   if (!shim_fd)
      shim_fd = drm_shim_file_create(fd);
   else
      p_atomic_inc(&shim_fd->refcount);

   _mesa_hash_table_insert(shim_device.fd_map, (void *)(uintptr_t)(fd + 1), shim_fd);
}

static void handle_delete_fxn(struct hash_entry *entry)
{
   drm_shim_bo_put(entry->data);
}

void drm_shim_fd_unregister(int fd)
{
   if (fd == -1)
      return;

   struct hash_entry *entry =
         _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));
   if (!entry)
      return;
   struct shim_fd *shim_fd = entry->data;
   _mesa_hash_table_remove(shim_device.fd_map, entry);

   if (!p_atomic_dec_zero(&shim_fd->refcount))
      return;

   _mesa_hash_table_destroy(shim_fd->handles, handle_delete_fxn);
   free(shim_fd);
}

struct shim_fd *
drm_shim_fd_lookup(int fd)
{
   if (fd == -1)
      return NULL;

   struct hash_entry *entry =
      _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));

   if (!entry)
      return NULL;
   return entry->data;
}

void
drm_shim_prime_fd_register(int fd, struct shim_bo *bo)
{
   if (fd < 0 || !bo)
      return;

   mtx_lock(&shim_device.prime_fd_lock);
   drm_shim_bo_get(bo);
   _mesa_hash_table_insert(shim_device.prime_fd_map,
                           (void *)(uintptr_t)(fd + 1), bo);
   mtx_unlock(&shim_device.prime_fd_lock);
}

struct shim_bo *
drm_shim_prime_fd_lookup(int fd)
{
   if (fd < 0)
      return NULL;

   mtx_lock(&shim_device.prime_fd_lock);
   struct hash_entry *entry = _mesa_hash_table_search(
      shim_device.prime_fd_map, (void *)(uintptr_t)(fd + 1));
   struct shim_bo *bo = entry ? entry->data : NULL;
   if (bo)
      drm_shim_bo_get(bo);
   mtx_unlock(&shim_device.prime_fd_lock);
   return bo;
}

void
drm_shim_prime_fd_unregister(int fd)
{
   if (fd < 0)
      return;

   mtx_lock(&shim_device.prime_fd_lock);
   struct hash_entry *entry = _mesa_hash_table_search(
      shim_device.prime_fd_map, (void *)(uintptr_t)(fd + 1));
   struct shim_bo *bo = entry ? entry->data : NULL;
   if (entry)
      _mesa_hash_table_remove(shim_device.prime_fd_map, entry);
   mtx_unlock(&shim_device.prime_fd_lock);

   if (bo)
      drm_shim_bo_put(bo);
}

int
drm_shim_prime_ioctl(int fd, unsigned long request, void *arg)
{
   struct shim_bo *bo = drm_shim_prime_fd_lookup(fd);
   if (!bo)
      return -EBADF;
   drm_shim_bo_put(bo);

   switch (request) {
   case DMA_BUF_IOCTL_SYNC: {
      const struct dma_buf_sync *sync = arg;
      if (!sync)
         return -EINVAL;
      const uint64_t access = sync->flags & DMA_BUF_SYNC_RW;
      const uint64_t phase = sync->flags & (DMA_BUF_SYNC_START |
                                             DMA_BUF_SYNC_END);
      const uint64_t known = DMA_BUF_SYNC_RW | DMA_BUF_SYNC_START |
                             DMA_BUF_SYNC_END;
      const bool valid = access && (phase == DMA_BUF_SYNC_START ||
                                    phase == DMA_BUF_SYNC_END) &&
                         !(sync->flags & ~known);
      if (valid && failed_dmabuf_export_once &&
          !failed_dmabuf_import_once)
         dmabuf_fallback_sync_calls++;
      return valid ? 0 : -EINVAL;
   }
   case DMA_BUF_IOCTL_EXPORT_SYNC_FILE: {
      struct dma_buf_export_sync_file *export = arg;
      if (!export || !(export->flags & DMA_BUF_SYNC_RW) ||
          (export->flags & ~DMA_BUF_SYNC_RW))
         return -EINVAL;
      if (getenv("ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK"))
         wsi_rollback_dmabuf_exports++;
      if (getenv("ASAHI_SHIM_FAIL_DMABUF_EXPORT_ONCE") &&
          !failed_dmabuf_export_once) {
         failed_dmabuf_export_once = true;
         errno = EMFILE;
         return -EMFILE;
      }
      export->fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
      return export->fd < 0 ? -errno : 0;
   }
   case DMA_BUF_IOCTL_IMPORT_SYNC_FILE: {
      const struct dma_buf_import_sync_file *import = arg;
      if (!import || import->fd < 0 ||
          !(import->flags & DMA_BUF_SYNC_RW) ||
          (import->flags & ~DMA_BUF_SYNC_RW))
         return -EINVAL;
      const char *wsi_case = getenv("ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK");
      if (wsi_case) {
         wsi_rollback_dmabuf_imports++;
         if (!strcmp(wsi_case, "present-import-failure") &&
             wsi_rollback_sync_file_exports > 0 &&
             !wsi_rollback_dmabuf_import_failed) {
            wsi_rollback_dmabuf_import_failed = true;
            wsi_rollback_failed_dmabuf_import =
               wsi_rollback_dmabuf_imports;
            errno = ENOMEM;
            return -ENOMEM;
         }
      }
      if (getenv("ASAHI_SHIM_FAIL_DMABUF_IMPORT_ONCE") &&
          failed_dmabuf_export_once && !failed_dmabuf_import_once) {
         failed_dmabuf_import_once = true;
         errno = ENOMEM;
         return -ENOMEM;
      }
      return 0;
   }
   default:
      return -ENOTTY;
   }
}

/* ioctl used by drmGetVersion() */
static int
drm_shim_ioctl_version(int fd, unsigned long request, void *arg)
{
   struct drm_version *args = arg;
   const char *date = "20190320";
   const char *desc = "shim";

   args->version_major = shim_device.version_major;
   args->version_minor = shim_device.version_minor;
   args->version_patchlevel = shim_device.version_patchlevel;

   if (args->name)
      strncpy(args->name, shim_device.driver_name, args->name_len);
   if (args->date)
      strncpy(args->date, date, args->date_len);
   if (args->desc)
      strncpy(args->desc, desc, args->desc_len);
   args->name_len = strlen(shim_device.driver_name);
   args->date_len = strlen(date);
   args->desc_len = strlen(desc);

   return 0;
}

static int
drm_shim_ioctl_get_unique(int fd, unsigned long request, void *arg)
{
   struct drm_unique *gu = arg;

   if (gu->unique && shim_device.unique)
      strncpy(gu->unique, shim_device.unique, gu->unique_len);
   gu->unique_len = shim_device.unique ? strlen(shim_device.unique) : 0;

   return 0;
}

static int
drm_shim_ioctl_get_cap(int fd, unsigned long request, void *arg)
{
   struct drm_get_cap *gc = arg;

   switch (gc->capability) {
   case DRM_CAP_PRIME:
      gc->value = DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;
      return 0;
   case DRM_CAP_SYNCOBJ:
   case DRM_CAP_SYNCOBJ_TIMELINE:
   case DRM_CAP_ADDFB2_MODIFIERS:
      gc->value = 1;
      return 0;

   default:
      fprintf(stderr, "DRM_IOCTL_GET_CAP: unhandled 0x%x\n",
              (int)gc->capability);
      return -1;
   }
}

static int
drm_shim_ioctl_gem_close(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_gem_close *c = arg;

   if (!c->handle)
      return 0;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)c->handle);
   if (!entry) {
      mtx_unlock(&shim_fd->handle_lock);
      return -EINVAL;
   }

   struct shim_bo *bo = entry->data;
   _mesa_hash_table_remove(shim_fd->handles, entry);
   drm_shim_bo_put(bo);
   mtx_unlock(&shim_fd->handle_lock);
   return 0;
}

static int
drm_shim_ioctl_syncobj_create(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_create *create = arg;

   if (timeline_host_values_enabled()) {
      if (timeline_host_next_handle >= TIMELINE_HOST_MAX_SYNCOBJS)
         return -ENOMEM;

      create->handle = timeline_host_next_handle++;
      timeline_host_syncobjs[create->handle].alive = true;
      timeline_host_syncobjs[create->handle].value =
         (create->flags & DRM_SYNCOBJ_CREATE_SIGNALED) ? 1 : 0;
   } else if (wsi_rollback_case()) {
      if (wsi_rollback_import_failed &&
          !wsi_rollback_marker_seen) {
         const bool failed_handle_valid =
            wsi_rollback_failed_handle > 0 &&
            wsi_rollback_failed_handle < 64;
         const bool success_handle_valid =
            wsi_rollback_success_handle == 0 ||
            wsi_rollback_success_handle < 64;
         const uint64_t failed_bit = failed_handle_valid
            ? UINT64_C(1) << wsi_rollback_failed_handle : 0;
         const uint64_t success_bit = wsi_rollback_success_handle > 0 &&
                                      success_handle_valid
            ? UINT64_C(1) << wsi_rollback_success_handle : 0;
         wsi_rollback_cleanup_before_marker =
            failed_handle_valid && success_handle_valid &&
            (wsi_rollback_destroyed_handles & failed_bit) &&
            (!success_bit ||
             (wsi_rollback_destroyed_handles & success_bit));
         wsi_rollback_marker_seen = true;
      }

      if (wsi_rollback_submit_transfer_failed &&
          !wsi_rollback_submit_retry_marker_seen) {
         const bool failed_handle_valid =
            wsi_rollback_failed_transfer_handle > 0 &&
            wsi_rollback_failed_transfer_handle < 64;
         const bool partial_handle_valid =
            wsi_rollback_partial_signal_handle > 0 &&
            wsi_rollback_partial_signal_handle < 64;
         const uint64_t failed_bit = failed_handle_valid
            ? UINT64_C(1) << wsi_rollback_failed_transfer_handle : 0;
         const uint64_t partial_bit = partial_handle_valid
            ? UINT64_C(1) << wsi_rollback_partial_signal_handle : 0;
         wsi_rollback_submit_cleanup_before_retry =
            failed_handle_valid && partial_handle_valid &&
            (wsi_rollback_destroyed_handles & failed_bit) &&
            (wsi_rollback_submit_reset_handles & partial_bit);
         wsi_rollback_submit_retry_marker_seen = true;
      }

      wsi_rollback_syncobj_creates++;
      create->handle = wsi_rollback_syncobj_creates;
      if (create->handle < 64) {
         wsi_rollback_created_handles |=
            UINT64_C(1) << create->handle;
      }
   } else if (queue_progress_lifecycle_case()) {
      queue_progress_syncobj_creates++;
      create->handle = queue_progress_syncobj_creates;
      if (create->handle < 64)
         queue_progress_created_handles |= UINT64_C(1) << create->handle;
   } else {
      create->handle = 1; /* 0 is invalid */
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_destroy(int fd, unsigned long request, void *arg)
{
   const struct drm_syncobj_destroy *destroy = arg;

   if (timeline_host_values_enabled()) {
      if (!timeline_host_handle_valid(destroy->handle))
         return -EINVAL;
      timeline_host_syncobjs[destroy->handle].alive = false;
   } else if (wsi_rollback_case()) {
      wsi_rollback_syncobj_destroys++;
      if (destroy->handle < 64) {
         wsi_rollback_destroyed_handles |=
            UINT64_C(1) << destroy->handle;
      }
   } else if (queue_progress_lifecycle_case()) {
      queue_progress_syncobj_destroys++;
      if (destroy->handle < 64)
         queue_progress_destroyed_handles |= UINT64_C(1) << destroy->handle;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_transfer(int fd, unsigned long request, void *arg)
{
   const struct drm_syncobj_transfer *transfer = arg;
   const char *test_case = wsi_rollback_case();

   if (test_case) {
      wsi_rollback_transfers++;
      if (!strcmp(test_case, "present-submit-failure")) {
         if (wsi_rollback_transfers == 3)
            wsi_rollback_partial_signal_handle = transfer->dst_handle;

         if (!wsi_rollback_submit_transfer_failed &&
             wsi_rollback_transfers == 4) {
            wsi_rollback_submit_transfer_failed = true;
            wsi_rollback_failed_transfer = wsi_rollback_transfers;
            wsi_rollback_failed_transfer_handle = transfer->dst_handle;
            errno = ENOMEM;
            return -1;
         }
      } else if (!strcmp(test_case, "empty-submit-failure")) {
         if (wsi_rollback_transfers == 1)
            wsi_rollback_partial_signal_handle = transfer->dst_handle;

         if (!wsi_rollback_submit_transfer_failed &&
             wsi_rollback_transfers == 2) {
            wsi_rollback_submit_transfer_failed = true;
            wsi_rollback_failed_transfer = wsi_rollback_transfers;
            wsi_rollback_failed_transfer_handle = transfer->dst_handle;
            errno = ENOMEM;
            return -1;
         }
      }
   }

   if (queue_progress_lifecycle_case()) {
      queue_progress_transfers++;
      queue_progress_transfer_shape_valid =
         transfer->src_handle != 0 && transfer->dst_handle != 0;
      queue_progress_transfer_src_handle = transfer->src_handle;
      queue_progress_transfer_dst_handle = transfer->dst_handle;
      queue_progress_transfer_src_point = transfer->src_point;
      queue_progress_transfer_dst_point = transfer->dst_point;
      queue_progress_transfer_flags = transfer->flags;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_fd_to_handle(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_handle *import = arg;
   const char *test_case = wsi_rollback_case();
   if (!test_case ||
       import->flags != DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE)
      return 0;

   wsi_rollback_sync_imports++;
   if (!strncmp(test_case, "present-", strlen("present-")) ||
       !strcmp(test_case, "empty-submit-failure") ||
       !strcmp(test_case, "release-failure") ||
       !strcmp(test_case, "present-priority"))
      return 0;

   const unsigned fail_at = !strcmp(test_case, "fence-failure") ? 2 : 1;
   if (!wsi_rollback_import_failed &&
       wsi_rollback_sync_imports == fail_at) {
      wsi_rollback_failed_handle = import->handle;
      wsi_rollback_import_failed = true;
      errno = ENOMEM;
      return -1;
   }

   if (!wsi_rollback_import_failed)
      wsi_rollback_success_handle = import->handle;

   return 0;
}

static int
drm_shim_ioctl_syncobj_timeline_signal(int fd, unsigned long request,
                                       void *arg)
{
   const struct drm_syncobj_timeline_array *signal = arg;

   if (timeline_host_values_enabled()) {
      const uint32_t *handles = (const void *)(uintptr_t)signal->handles;
      const uint64_t *points = (const void *)(uintptr_t)signal->points;
      if (!handles || !points || signal->flags)
         return -EINVAL;

      for (uint32_t i = 0; i < signal->count_handles; ++i) {
         if (!timeline_host_handle_valid(handles[i]) ||
             points[i] < timeline_host_syncobjs[handles[i]].value)
            return -EINVAL;
         timeline_host_syncobjs[handles[i]].value = points[i];
      }
   }

   if (queue_progress_lifecycle_case()) {
      const uint32_t *handles = (const void *)(uintptr_t)signal->handles;
      const uint64_t *points = (const void *)(uintptr_t)signal->points;
      queue_progress_timeline_signals++;
      queue_progress_signal_shape_valid =
         signal->count_handles == 1 && signal->flags == 0 && handles &&
         points && handles[0] != 0;
      if (handles && points && signal->count_handles) {
         queue_progress_initial_handle = handles[0];
         queue_progress_initial_point = points[0];
      }
   }

   if (getenv("ASAHI_SHIM_FAIL_QUEUE_TIMELINE_SIGNAL_ONCE") &&
       !failed_queue_timeline_signal_once) {
      failed_queue_timeline_signal_once = true;
      errno = ENOMEM;
      return -1;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_query(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_timeline_array *query = arg;
   if (!timeline_host_values_enabled())
      return 0;

   const uint32_t *handles = (const void *)(uintptr_t)query->handles;
   uint64_t *points = (void *)(uintptr_t)query->points;
   if (!handles || !points ||
       (query->flags & ~DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED))
      return -EINVAL;

   for (uint32_t i = 0; i < query->count_handles; ++i) {
      if (!timeline_host_handle_valid(handles[i]))
         return -EINVAL;
      points[i] = timeline_host_syncobjs[handles[i]].value;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_timeline_wait(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_timeline_wait *wait = arg;
   if (!timeline_host_values_enabled())
      return 0;

   const uint32_t *handles = (const void *)(uintptr_t)wait->handles;
   const uint64_t *points = (const void *)(uintptr_t)wait->points;
   if (!handles || !points || !wait->count_handles)
      return -EINVAL;

   const bool wait_all = wait->flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
   bool ready = wait_all;
   for (uint32_t i = 0; i < wait->count_handles; ++i) {
      if (!timeline_host_handle_valid(handles[i]))
         return -EINVAL;

      const bool this_ready =
         timeline_host_syncobjs[handles[i]].value >= points[i];
      if (!wait_all && this_ready) {
         wait->first_signaled = i;
         ready = true;
         break;
      }
      if (wait_all && !this_ready)
         ready = false;
   }

   if (!ready) {
      errno = ETIME;
      return -ETIME;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_handle *handle = arg;

   /* No-op submissions complete immediately, so their completion sync file
    * must be immediately pollable. Ordinary syncobj-fd export stays closed:
    * an eventfd models a sync file, not another syncobj handle. */
   if (!handle->handle ||
       handle->flags != DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE)
      return -EINVAL;

   const char *wsi_case = getenv("ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK");
   if (wsi_case) {
      wsi_rollback_sync_file_exports++;
      if (wsi_rollback_sync_file_exports == 1)
         wsi_rollback_first_sync_file_export_handle = handle->handle;
      if (!strcmp(wsi_case, "present-export-failure") &&
          !wsi_rollback_sync_file_export_failed) {
         wsi_rollback_sync_file_export_failed = true;
         wsi_rollback_failed_sync_file_export =
            wsi_rollback_sync_file_exports;
         wsi_rollback_failed_sync_file_export_handle = handle->handle;
         errno = EMFILE;
         return -1;
      }
   }

   handle->fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
   return handle->fd < 0 ? -errno : 0;
}

static int
drm_shim_ioctl_syncobj_reset(int fd, unsigned long request, void *arg)
{
   const struct drm_syncobj_array *reset = arg;

   if (wsi_rollback_dmabuf_import_failed ||
       wsi_rollback_sync_file_export_failed ||
       wsi_rollback_submit_transfer_failed) {
      const uint32_t *handles = (const void *)(uintptr_t)reset->handles;
      wsi_rollback_syncobj_resets_after_failure++;
      if (wsi_rollback_syncobj_resets_after_failure == 1) {
         wsi_rollback_first_reset_shape_valid =
            reset->count_handles == 1 && handles && handles[0] != 0;
         if (handles && reset->count_handles)
            wsi_rollback_first_reset_handle = handles[0];
      }

      if (wsi_rollback_submit_transfer_failed && handles) {
         for (uint32_t i = 0; i < reset->count_handles; i++) {
            if (handles[i] < 64)
               wsi_rollback_submit_reset_handles |=
                  UINT64_C(1) << handles[i];
         }
      }
   }

   return 0;
}

static int
drm_shim_ioctl_prime_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *prime = arg;
   if (!shim_fd || !prime->handle ||
       (prime->flags & ~(DRM_CLOEXEC | DRM_RDWR)))
      return -EINVAL;

   p_atomic_inc(&prime_export_calls);
   if (getenv("ASAHI_SHIM_DELAY_PRIME_EXPORT"))
      usleep(25000);

   if (getenv("ASAHI_SHIM_FAIL_PRIME_EXPORT_ONCE") &&
       !failed_prime_export_once) {
      failed_prime_export_once = true;
      errno = EMFILE;
      return -1;
   }

   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, prime->handle);
   if (!bo)
      return -ENOENT;

   const unsigned flags = (prime->flags & DRM_CLOEXEC) ? MFD_CLOEXEC : 0;
   const int prime_fd = memfd_create("drm-shim-prime", flags);
   if (prime_fd < 0 || ftruncate(prime_fd, bo->size)) {
      const int error = errno;
      if (prime_fd >= 0)
         close(prime_fd);
      drm_shim_bo_put(bo);
      return -error;
   }

   drm_shim_prime_fd_register(prime_fd, bo);
   drm_shim_bo_put(bo);
   prime->fd = prime_fd;
   return 0;
}

static int
drm_shim_ioctl_prime_fd_to_handle(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *prime = arg;
   if (!shim_fd || prime->fd < 0 || prime->flags)
      return -EINVAL;

   struct shim_bo *bo = drm_shim_prime_fd_lookup(prime->fd);
   if (!bo)
      return -ENOENT;

   prime->handle = drm_shim_bo_get_handle(shim_fd, bo);
   drm_shim_bo_put(bo);
   return prime->handle ? 0 : -ENOMEM;
}

static int
drm_shim_ioctl_syncobj_wait(int fd, unsigned long request, void *arg)
{
   const struct drm_syncobj_wait *wait = arg;
   const uint32_t *handles = (const void *)(uintptr_t)wait->handles;

   if (wsi_rollback_submit_transfer_failed &&
       !wsi_rollback_submit_retry_marker_seen && handles) {
      for (uint32_t i = 0; i < wait->count_handles; i++) {
         if (handles[i] == wsi_rollback_failed_transfer_handle) {
            wsi_rollback_stale_fence_wait_rejected = true;
            errno = ETIME;
            return -ETIME;
         }
      }
   }

   if (getenv("ASAHI_SHIM_TIMEOUT_SYNC_WAIT_ONCE") &&
       !timed_out_sync_wait_once) {
      timed_out_sync_wait_once = true;
      errno = ETIME;
      return -ETIME;
   }

   if (failed_dmabuf_import_once)
      waited_after_dmabuf_import_failure = true;

   if (wsi_rollback_dmabuf_import_failed ||
       wsi_rollback_sync_file_export_failed ||
       wsi_rollback_submit_transfer_failed) {
      wsi_rollback_waits_after_sync_failure++;
      if (wsi_rollback_waits_after_sync_failure == 1) {
         wsi_rollback_fallback_wait_shape_valid =
            wait->count_handles == 1 && handles && handles[0] != 0;
         if (handles && wait->count_handles)
            wsi_rollback_fallback_wait_handle = handles[0];
         wsi_rollback_fallback_wait_flags = wait->flags;
      }
   }
   return 0;
}

ioctl_fn_t core_ioctls[] = {
   [_IOC_NR(DRM_IOCTL_VERSION)] = drm_shim_ioctl_version,
   [_IOC_NR(DRM_IOCTL_GET_UNIQUE)] = drm_shim_ioctl_get_unique,
   [_IOC_NR(DRM_IOCTL_GET_CAP)] = drm_shim_ioctl_get_cap,
   [_IOC_NR(DRM_IOCTL_GEM_CLOSE)] = drm_shim_ioctl_gem_close,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_CREATE)] = drm_shim_ioctl_syncobj_create,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_DESTROY)] = drm_shim_ioctl_syncobj_destroy,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD)] =
      drm_shim_ioctl_syncobj_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE)] =
      drm_shim_ioctl_syncobj_fd_to_handle,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_WAIT)] = drm_shim_ioctl_syncobj_wait,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TRANSFER)] = drm_shim_ioctl_syncobj_transfer,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_RESET)] = drm_shim_ioctl_syncobj_reset,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL)] =
      drm_shim_ioctl_syncobj_timeline_signal,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT)] =
      drm_shim_ioctl_syncobj_timeline_wait,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_QUERY)] = drm_shim_ioctl_syncobj_query,
   [_IOC_NR(DRM_IOCTL_PRIME_HANDLE_TO_FD)] = drm_shim_ioctl_prime_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_PRIME_FD_TO_HANDLE)] = drm_shim_ioctl_prime_fd_to_handle,
};

/**
 * Implements the GEM core ioctls, and calls into driver-specific ioctls.
 */
int
drm_shim_ioctl(int fd, unsigned long request, void *arg)
{
   ASSERTED int type = _IOC_TYPE(request);
   int nr = _IOC_NR(request);

   assert(type == DRM_IOCTL_BASE);

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      int driver_nr = nr - DRM_COMMAND_BASE;

      if (driver_nr < shim_device.driver_ioctl_count &&
          shim_device.driver_ioctls[driver_nr]) {
         return shim_device.driver_ioctls[driver_nr](fd, request, arg);
      }
   } else {
      if (nr < ARRAY_SIZE(core_ioctls) && core_ioctls[nr]) {
         return core_ioctls[nr](fd, request, arg);
      }
   }

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      fprintf(stderr,
              "DRM_SHIM: unhandled driver DRM ioctl %d (0x%08lx)\n",
              nr - DRM_COMMAND_BASE, request);
   } else {
      fprintf(stderr,
              "DRM_SHIM: unhandled core DRM ioctl 0x%X (0x%08lx)\n",
              nr, request);
   }

   return -EINVAL;
}

int
drm_shim_bo_init(struct shim_bo *bo, size_t size)
{

   mtx_lock(&shim_device.mem_lock);
   bo->mem_addr = util_vma_heap_alloc(&shim_device.mem_heap, size, shim_page_size);
   mtx_unlock(&shim_device.mem_lock);

   if (!bo->mem_addr)
      return -ENOMEM;

   bo->size = size;
   p_atomic_set(&bo->refcount, 1);

   return 0;
}

struct shim_bo *
drm_shim_bo_lookup(struct shim_fd *shim_fd, int handle)
{
   if (!handle)
      return NULL;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)handle);
   struct shim_bo *bo = entry ? entry->data : NULL;
   mtx_unlock(&shim_fd->handle_lock);

   if (bo)
      p_atomic_inc(&bo->refcount);

   return bo;
}

void
drm_shim_bo_get(struct shim_bo *bo)
{
   p_atomic_inc(&bo->refcount);
}

void
drm_shim_bo_put(struct shim_bo *bo)
{
   if (p_atomic_dec_return(&bo->refcount) != 0)
      return;

   if (shim_device.driver_bo_free)
      shim_device.driver_bo_free(bo);

   /* Give the pages back. The backing memfd is a 4 GiB sparse file that is
    * never truncated, so without this every page a process ever touched
    * stays resident until it exits -- a long CTS run on the shim then holds
    * gigabytes of dead buffers. It also hands the next allocation zeroed
    * memory, as a fresh kernel BO would. */
   fallocate(shim_device.mem_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
             bo->mem_addr, ALIGN_POT(bo->size, shim_page_size));

   mtx_lock(&shim_device.mem_lock);
   util_vma_heap_free(&shim_device.mem_heap, bo->mem_addr, bo->size);
   mtx_unlock(&shim_device.mem_lock);
   free(bo);
}

int
drm_shim_bo_get_handle(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   /* We should probably have some real datastructure for finding the free
    * number.
    */
   mtx_lock(&shim_fd->handle_lock);
   for (int new_handle = 1; ; new_handle++) {
      void *key = (void *)(uintptr_t)new_handle;
      if (!_mesa_hash_table_search(shim_fd->handles, key)) {
         drm_shim_bo_get(bo);
         _mesa_hash_table_insert(shim_fd->handles, key, bo);
         mtx_unlock(&shim_fd->handle_lock);
         return new_handle;
      }
   }
   mtx_unlock(&shim_fd->handle_lock);

   return 0;
}

/* Creates an mmap offset for the BO in the DRM fd.
 */
uint64_t
drm_shim_bo_get_mmap_offset(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   mtx_lock(&shim_device.mem_lock);
   _mesa_hash_table_u64_insert(shim_device.offset_map, bo->mem_addr, bo);
   mtx_unlock(&shim_device.mem_lock);

   /* reuse the buffer address as the mmap offset: */
   return bo->mem_addr;
}

void
drm_shim_init_iomem_region(off64_t offset, size_t size,
                           void *(*mmap_handler)(size_t, int, int, off64_t))
{
   shim_device.iomem_region.mmap = mmap_handler;
   shim_device.iomem_region.start = offset;
   shim_device.iomem_region.size = size;
}

/* For mmap() on the DRM fd, look up the BO from the "offset" and map the BO's
 * fd.
 */
void *
drm_shim_mmap(struct shim_fd *shim_fd, size_t length, int prot, int flags,
              int fd, off64_t offset)
{
   if (shim_device.iomem_region.mmap &&
       offset >= shim_device.iomem_region.start &&
       offset + length <= shim_device.iomem_region.start + shim_device.iomem_region.size) {
      return shim_device.iomem_region.mmap(length, prot, flags, offset);
   }

   mtx_lock(&shim_device.mem_lock);
   struct shim_bo *bo = _mesa_hash_table_u64_search(shim_device.offset_map, offset);
   mtx_unlock(&shim_device.mem_lock);

   if (!bo)
      return MAP_FAILED;

   if (length > bo->size)
      return MAP_FAILED;

   /* The offset we pass to mmap must be aligned to the page size */
   assert((bo->mem_addr & (shim_page_size - 1)) == 0);

   return mmap(NULL, length, prot, flags, shim_device.mem_fd, bo->mem_addr);
}
