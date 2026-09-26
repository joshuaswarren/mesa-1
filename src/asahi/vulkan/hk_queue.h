/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "hk_sync.h"
#include "vk_queue.h"

struct hk_device;

/* Camera interaction can enqueue hundreds of small ANGLE submits between the
 * causal fragment launch and the ioctl that observes DeviceLost.  Keep a
 * deeper diagnostic-only ring so the exact graphics stream is not overwritten
 * by downstream synchronization submits. */
#define HK_G17P_FAULTTRACE_HISTORY 128

struct hk_g17p_faulttrace_payload {
   void *data;
   size_t size;
   void *extra;
   size_t extra_size;
   uint64_t sequence;
};

struct hk_queue {
   struct vk_queue vk;

   struct {
      /* Asahi kernel queue ID */
      uint32_t id;

      /* Timeline syncobj backing the queue */
      uint32_t syncobj;

      /* Ring-idx used with virtgpu, equal to priority + 1 */
      uint32_t virt_ring_idx;

      /* Current maximum timeline value for the queue's syncobj. If the
       * syncobj's value equals timeline_value, then all work is complete.
       */
      uint64_t timeline_value;
   } drm;

   /* Device-lifetime command storage for the G17P SSTT maintenance encoder.
    * Sparse image remaps submit this record before their signal operations. */
   struct agx_bo *g17p_sstt_flush_bo;

   /* Kinds of streams this queue has ever run, see hk_sync.h. */
   struct hk_sync_queue sync;

   /* Failure-only rolling ioctl history.  A GPU fault is reported to
    * userspace by a later submit, so decoding only the rejected ioctl loses
    * the causal command.  Allocated only when faulttrace is explicitly armed. */
   struct hk_g17p_faulttrace_payload
      g17p_faulttrace[HK_G17P_FAULTTRACE_HISTORY];
   uint64_t g17p_faulttrace_sequence;
   unsigned g17p_faulttrace_head;
   unsigned g17p_faulttrace_count;
};

static inline struct hk_device *
hk_queue_device(struct hk_queue *queue)
{
   return (struct hk_device *)queue->vk.base.device;
}

VkResult hk_queue_init(struct hk_device *dev,
                       const VkDeviceQueueCreateInfo *pCreateInfo,
                       uint32_t index_in_family);

void hk_queue_finish(struct hk_device *dev, struct hk_queue *queue);
