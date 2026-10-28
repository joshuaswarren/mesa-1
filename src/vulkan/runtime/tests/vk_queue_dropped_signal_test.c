/* SPDX-License-Identifier: MIT */

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vk_sync_timeline.h"
#include "util/os_time.h"

#include <stdio.h>
#include <stdlib.h>

struct test_sync {
   struct vk_sync base;
   bool signalled;
};

static unsigned signal_calls;
static unsigned fail_call;
static int wait_entered;
static VkResult (*real_timeline_wait)(struct vk_device *, struct vk_sync *,
                                     uint64_t, enum vk_sync_wait_flags, uint64_t);

static void
require(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      exit(EXIT_FAILURE);
   }
}

static VkResult
sync_init(struct vk_device *device, struct vk_sync *sync, uint64_t value)
{
   ((struct test_sync *)sync)->signalled = value != 0;
   return VK_SUCCESS;
}

static void
sync_finish(struct vk_device *device, struct vk_sync *sync)
{
}

static VkResult
sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   ((struct test_sync *)sync)->signalled = false;
   return VK_SUCCESS;
}

static VkResult
sync_signal(struct vk_device *device, struct vk_sync *sync, uint64_t value)
{
   if (++signal_calls == fail_call)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   ((struct test_sync *)sync)->signalled = true;
   return VK_SUCCESS;
}

static VkResult
sync_wait(struct vk_device *device, struct vk_sync *sync, uint64_t value,
          enum vk_sync_wait_flags flags, uint64_t timeout)
{
   /* Model a backend that releases its waits on loss. The public wait must
    * still return DEVICE_LOST, regardless of these untrusted payloads.
    */
   return vk_device_is_lost_no_report(device) ||
          ((struct test_sync *)sync)->signalled ? VK_SUCCESS : VK_TIMEOUT;
}

static VkResult
timeline_wait_witness(struct vk_device *device, struct vk_sync *sync,
                       uint64_t value, enum vk_sync_wait_flags flags,
                       uint64_t timeout)
{
   /* This is after WaitSemaphores' initial device-loss check. A wait entered
    * here before the submission fails must not remain blocked forever.
    */
   p_atomic_inc(&wait_entered);
   return real_timeline_wait(device, sync, value, flags, timeout);
}

struct wait_context {
   struct vk_device *device;
   VkSemaphore semaphores[2];
   VkResult result;
   int done;
};

static int
wait_thread(void *data)
{
   struct wait_context *ctx = data;
   const uint64_t values[] = {5, 5};
   const VkSemaphoreWaitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 2,
      .pSemaphores = ctx->semaphores,
      .pValues = values,
   };
   ctx->result = vk_common_WaitSemaphores(vk_device_to_handle(ctx->device),
                                          &info, UINT64_MAX);
   p_atomic_set(&ctx->done, 1);
   return 0;
}

static void
wait_for(int *flag, const char *message)
{
   const int64_t deadline = os_time_get_nano() + 2 * ONE_SECOND_IN_NS;
   while (!p_atomic_read(flag)) {
      require(os_time_get_nano() < deadline, message);
      const struct timespec delay = {.tv_nsec = 1000000};
      thrd_sleep(&delay, NULL);
   }
}

static const struct vk_sync_type test_sync_type = {
   .size = sizeof(struct test_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_GPU_MULTI_WAIT | VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_CPU_SIGNAL,
   .init = sync_init,
   .finish = sync_finish,
   .reset = sync_reset,
   .signal = sync_signal,
   .wait = sync_wait,
};

static VkResult
fail_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   /* The submit thread normally records loss after submit_final returns.
    * Exercise that interval when the best-effort signal itself fails: loss
    * must be recorded before publishing the point and waking its waiters.
    */
   if (fail_call)
      return VK_ERROR_DEVICE_LOST;
   return vk_queue_set_lost(queue, "injected submission failure");
}

static void
run(bool timeline, unsigned fail)
{
   struct vk_device device = {
      .alloc = *vk_default_allocator(),
      .submit_mode = VK_QUEUE_SUBMIT_MODE_IMMEDIATE,
      .timeline_mode = timeline ? VK_DEVICE_TIMELINE_MODE_EMULATED :
                                 VK_DEVICE_TIMELINE_MODE_NONE,
   };
   vk_object_base_init(NULL, &device.base, VK_OBJECT_TYPE_DEVICE);
   device._lost.reported = true;
   list_inithead(&device.queues);
   struct vk_queue queue;
   const VkDeviceQueueCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
   };
   require(vk_queue_init(&queue, &device, &create, 0) == VK_SUCCESS,
           "queue initialization failed");
   queue.driver_submit = fail_submit;

   struct vk_sync_timeline_type timeline_type =
      vk_sync_timeline_get_type(&test_sync_type);
   real_timeline_wait = timeline_type.sync.wait;
   timeline_type.sync.wait = timeline_wait_witness;
   const struct vk_sync_type *type = timeline ? &timeline_type.sync :
                                               &test_sync_type;
   struct vk_semaphore *semaphores[2];
   VkSemaphoreSubmitInfo signals[2];
   for (unsigned i = 0; i < 2; ++i) {
      semaphores[i] = calloc(1, sizeof(struct vk_semaphore) + type->size);
      require(semaphores[i] != NULL, "semaphore allocation failed");
      vk_object_base_init(&device, &semaphores[i]->base,
                          VK_OBJECT_TYPE_SEMAPHORE);
      semaphores[i]->type = timeline ? VK_SEMAPHORE_TYPE_TIMELINE :
                                      VK_SEMAPHORE_TYPE_BINARY;
      require(vk_sync_init(&device, &semaphores[i]->permanent, type,
                           timeline ? VK_SYNC_IS_TIMELINE : 0, 0) == VK_SUCCESS,
              "sync initialization failed");
      signals[i] = (VkSemaphoreSubmitInfo) {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
         .semaphore = vk_semaphore_to_handle(semaphores[i]),
         .value = timeline ? 5 : 0,
         .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      };
   }
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 2,
      .pSignalSemaphoreInfos = signals,
   };
   signal_calls = 0;
   fail_call = fail;
   struct wait_context wait = {
      .device = &device,
      .semaphores = {signals[0].semaphore, signals[1].semaphore},
   };
   thrd_t thread;
   if (timeline) {
      p_atomic_set(&wait_entered, 0);
      require(thrd_create(&thread, wait_thread, &wait) == thrd_success,
              "wait thread creation failed");
      wait_for(&wait_entered, "wait did not enter the timeline implementation");
   }
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit,
                                  VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST,
           "submission failure was not propagated");
   require(signal_calls == 2, "a failed signal prevented later signals");
   require(vk_device_is_lost_no_report(&device),
           "signal failure was not recorded before waking waiters");
   if (timeline) {
      wait_for(&wait.done, "device loss stranded an existing timeline waiter");
      require(thrd_join(thread, NULL) == thrd_success, "wait thread join failed");
      require(wait.result == VK_ERROR_DEVICE_LOST,
              "waiter trusted a payload after device loss");
   }
   for (unsigned i = 0; i < 2; ++i) {
      const bool expected = fail != i + 1;
      struct vk_sync *sync = &semaphores[i]->permanent;
      if (timeline) {
         struct vk_sync_timeline *tl = vk_sync_as_timeline(sync);
         require(tl->state->highest_pending == 5 && tl->state->highest_past == 5,
                 "timeline did not release its pending wait");
         require(tl->state->refcount == 1,
                 "dropped submission leaked its time point reference");
      } else {
         require(((struct test_sync *)sync)->signalled == expected,
                 "incorrect binary signal state");
      }
      vk_sync_finish(&device, sync);
      vk_object_base_finish(&semaphores[i]->base);
      free(semaphores[i]);
   }
   vk_queue_finish(&queue);
   vk_object_base_finish(&device.base);
   printf("%s, failed signal %u: PASS\n", timeline ? "timeline" : "binary", fail);
}

int
main(void)
{
   for (unsigned fail = 0; fail <= 2; ++fail) {
      run(false, fail);
      run(true, fail);
   }
   return EXIT_SUCCESS;
}
