/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Live queue/fence contract checks: GPU Signal waited for on the queue
 * timeline directly, CPU Signal moving a fence backwards after a GPU
 * Signal (and the GPU value never re-arriving), waits satisfied by a later
 * CPU Signal, multi-fence ANY/ALL waits, cross-queue Wait/Signal, and
 * queue release while a CPU waiter is parked on that queue's timeline or
 * the submit worker is parked on a queue Wait.
 *
 * Every GPU submission is a buffer copy whose result is checked after the
 * fence wait, so a wait that returns early is caught as stale data.  Each
 * submission that may still be executing when the next one is recorded
 * uses its own allocator/list slot: Reset on an allocator with work in
 * flight is an application error that recycles storage the GPU is still
 * reading.  A watchdog aborts the process if any wait hangs.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define COPY_BYTES 4096
#define SLOTS 8
#define WATCHDOG_SECONDS 60

static unsigned failures;

static void
watchdog(int sig)
{
   (void)sig;
   static const char msg[] = "FAIL - watchdog: a wait did not complete\n"
                             "QUEUE FENCE LIVE: HUNG\n";
   (void)!write(1, msg, sizeof(msg) - 1);
   _exit(2);
}

#define CHECK(cond, what)                                                    \
   do {                                                                      \
      if (cond) {                                                            \
         printf("ok   - %s\n", what);                                        \
      } else {                                                               \
         printf("FAIL - %s\n", what);                                        \
         failures++;                                                         \
      }                                                                      \
   } while (0)

#define REQUIRE(cond, what)                                                  \
   do {                                                                      \
      if (!(cond)) {                                                         \
         printf("FAIL - %s\n", what);                                        \
         printf("QUEUE FENCE LIVE: ABORTED\n");                              \
         return 1;                                                           \
      }                                                                      \
   } while (0)

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static ID3D12Resource *
make_upload_buffer(ID3D12Device *dev, UINT64 size)
{
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = size,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *res = NULL;
   HRESULT hr = ID3D12Device_CreateCommittedResource(
      dev, &heap, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
      (void **)&res);
   return hr == S_OK ? res : NULL;
}

static void
fill_buffer(ID3D12Resource *res, uint32_t seed)
{
   uint32_t *map = NULL;
   if (ID3D12Resource_Map(res, 0, NULL, (void **)&map) != S_OK || !map)
      return;
   for (unsigned i = 0; i < COPY_BYTES / 4; i++)
      map[i] = seed * 0x9e3779b1u + i;
   ID3D12Resource_Unmap(res, 0, NULL);
}

static bool
buffer_matches(ID3D12Resource *res, uint32_t seed)
{
   uint32_t *map = NULL;
   D3D12_RANGE none = {0, 0};
   if (ID3D12Resource_Map(res, 0, NULL, (void **)&map) != S_OK || !map)
      return false;
   bool ok = true;
   for (unsigned i = 0; i < COPY_BYTES / 4 && ok; i++)
      ok = map[i] == seed * 0x9e3779b1u + i;
   ID3D12Resource_Unmap(res, 0, &none);
   return ok;
}

struct ctx {
   ID3D12Device *dev;
   ID3D12Device1 *dev1;
   ID3D12CommandAllocator *alloc[SLOTS];
   ID3D12GraphicsCommandList *list[SLOTS];
   ID3D12Resource *src;
   ID3D12Resource *dst;
   uint32_t seed;
};

/* Records and submits one copy of a fresh pattern from the given slot; the
 * caller verifies the destination after its fence wait with
 * buffer_matches(ctx->dst, seed).  A slot may only be reused once its
 * previous submission has been waited for. */
static bool
submit_copy(struct ctx *c, ID3D12CommandQueue *queue, unsigned slot,
            uint32_t *seed_out)
{
   c->seed++;
   fill_buffer(c->src, c->seed);
   if (ID3D12CommandAllocator_Reset(c->alloc[slot]) != S_OK ||
       ID3D12GraphicsCommandList_Reset(c->list[slot], c->alloc[slot],
                                       NULL) != S_OK)
      return false;
   ID3D12GraphicsCommandList_CopyBufferRegion(c->list[slot], c->dst, 0,
                                              c->src, 0, COPY_BYTES);
   if (ID3D12GraphicsCommandList_Close(c->list[slot]) != S_OK)
      return false;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)c->list[slot]};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   *seed_out = c->seed;
   return true;
}

static ID3D12CommandQueue *
make_queue(ID3D12Device *dev)
{
   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   if (ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
                                       (void **)&queue) != S_OK)
      return NULL;
   return queue;
}

static ID3D12Fence *
make_fence(ID3D12Device *dev, UINT64 initial)
{
   ID3D12Fence *fence = NULL;
   if (ID3D12Device_CreateFence(dev, initial, D3D12_FENCE_FLAG_NONE,
                                &IID_ID3D12Fence, (void **)&fence) != S_OK)
      return NULL;
   return fence;
}

struct wait_thread {
   ID3D12Fence *fence;
   UINT64 value;
   HRESULT hr;
   UINT64 completed;
   double wait_ms;
   pthread_t thread;
};

static void *
wait_thread_main(void *arg)
{
   struct wait_thread *w = arg;
   double t0 = now_ms();
   w->hr = ID3D12Fence_SetEventOnCompletion(w->fence, w->value, NULL);
   w->wait_ms = now_ms() - t0;
   w->completed = ID3D12Fence_GetCompletedValue(w->fence);
   return NULL;
}

static bool
wait_thread_start(struct wait_thread *w, ID3D12Fence *fence, UINT64 value)
{
   memset(w, 0, sizeof(*w));
   w->fence = fence;
   w->value = value;
   return pthread_create(&w->thread, NULL, wait_thread_main, w) == 0;
}

static void
sleep_ms(unsigned ms)
{
   struct timespec ts = {.tv_sec = ms / 1000,
                         .tv_nsec = (long)(ms % 1000) * 1000000L};
   nanosleep(&ts, NULL);
}

int
main(void)
{
   struct ctx c = {0};
   /* Line-buffered so a watchdog abort keeps every check already printed. */
   setvbuf(stdout, NULL, _IOLBF, 0);
   signal(SIGALRM, watchdog);
   alarm(WATCHDOG_SECONDS);

   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&c.dev);
   REQUIRE(hr == S_OK && c.dev, "D3D12CreateDevice");
   hr = ID3D12Device_QueryInterface(c.dev, &IID_ID3D12Device1,
                                    (void **)&c.dev1);
   REQUIRE(hr == S_OK && c.dev1, "ID3D12Device1");

   for (unsigned i = 0; i < SLOTS; i++) {
      hr = ID3D12Device_CreateCommandAllocator(
         c.dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
         (void **)&c.alloc[i]);
      REQUIRE(hr == S_OK, "CreateCommandAllocator");
      hr = ID3D12Device_CreateCommandList(
         c.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.alloc[i], NULL,
         &IID_ID3D12GraphicsCommandList, (void **)&c.list[i]);
      REQUIRE(hr == S_OK, "CreateCommandList");
      REQUIRE(ID3D12GraphicsCommandList_Close(c.list[i]) == S_OK,
              "initial Close");
   }
   c.src = make_upload_buffer(c.dev, COPY_BYTES);
   c.dst = make_upload_buffer(c.dev, COPY_BYTES);
   REQUIRE(c.src && c.dst, "copy buffers");

   ID3D12CommandQueue *queue = make_queue(c.dev);
   ID3D12Fence *fence = make_fence(c.dev, 0);
   REQUIRE(queue && fence, "queue and fence");

   /* 1. GPU Signal, blocking CPU wait, data visible. */
   {
      bool all_ok = true;
      UINT64 value = 0;
      for (unsigned i = 0; i < 50 && all_ok; i++) {
         uint32_t seed;
         all_ok = submit_copy(&c, queue, 0, &seed);
         value++;
         all_ok = all_ok &&
                  ID3D12CommandQueue_Signal(queue, fence, value) == S_OK;
         all_ok = all_ok &&
                  ID3D12Fence_SetEventOnCompletion(fence, value, NULL) ==
                     S_OK;
         all_ok = all_ok && ID3D12Fence_GetCompletedValue(fence) == value;
         all_ok = all_ok && buffer_matches(c.dst, seed);
      }
      CHECK(all_ok, "50 x submit + queue Signal + blocking wait see the "
                    "copied data");

      /* A Signal with no submission ahead of it completes immediately. */
      value++;
      hr = ID3D12CommandQueue_Signal(queue, fence, value);
      CHECK(hr == S_OK &&
               ID3D12Fence_SetEventOnCompletion(fence, value, NULL) == S_OK &&
               ID3D12Fence_GetCompletedValue(fence) == value,
            "queue Signal on an idle queue reaches the CPU");

      /* Two GPU signals pending at once, waited for out of order. */
      uint32_t seed_a, seed_b;
      REQUIRE(submit_copy(&c, queue, 0, &seed_a), "submit a");
      ID3D12CommandQueue_Signal(queue, fence, value + 1);
      REQUIRE(submit_copy(&c, queue, 1, &seed_b), "submit b");
      ID3D12CommandQueue_Signal(queue, fence, value + 2);
      hr = ID3D12Fence_SetEventOnCompletion(fence, value + 2, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == value + 2 &&
               buffer_matches(c.dst, seed_b),
            "waiting for the later of two pending GPU signals");
      hr = ID3D12Fence_SetEventOnCompletion(fence, value + 1, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == value + 2,
            "the earlier pending value is already satisfied");
      value += 2;

      /* 2. CPU Signal moves the fence backwards after a GPU Signal and the
       * GPU value does not re-arrive when the retirement worker catches
       * up. */
      uint32_t seed;
      REQUIRE(submit_copy(&c, queue, 0, &seed), "submit for rewind");
      ID3D12CommandQueue_Signal(queue, fence, 1000);
      hr = ID3D12Fence_SetEventOnCompletion(fence, 1000, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == 1000,
            "GPU Signal to 1000 observed");
      hr = ID3D12Fence_Signal(fence, 5);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == 5,
            "CPU Signal rewinds the fence to 5");
      sleep_ms(30);
      CHECK(ID3D12Fence_GetCompletedValue(fence) == 5,
            "the retired GPU value does not re-arrive after the rewind");
      hr = ID3D12Fence_SetEventOnCompletion(fence, 5, NULL);
      CHECK(hr == S_OK, "waiting for the rewound value returns at once");

      /* A GPU Signal after the rewind, then a CPU rewind racing the
       * retirement worker: the GPU value must be published exactly once. */
      bool once_ok = true;
      for (unsigned i = 0; i < 20 && once_ok; i++) {
         REQUIRE(submit_copy(&c, queue, 0, &seed), "submit for race");
         ID3D12CommandQueue_Signal(queue, fence, 2000 + i);
         once_ok = ID3D12Fence_SetEventOnCompletion(fence, 2000 + i, NULL) ==
                   S_OK;
         ID3D12Fence_Signal(fence, 7);
         sleep_ms(2);
         once_ok = once_ok && ID3D12Fence_GetCompletedValue(fence) == 7;
      }
      CHECK(once_ok, "20 x GPU Signal then CPU rewind: value published once");
      value = 7;

      /* 3. A wait with nothing pending on the GPU is satisfied by a CPU
       * Signal from another thread. */
      struct wait_thread w;
      REQUIRE(wait_thread_start(&w, fence, 50), "wait thread");
      sleep_ms(20);
      ID3D12Fence_Signal(fence, 50);
      pthread_join(w.thread, NULL);
      CHECK(w.hr == S_OK && w.completed == 50 && w.wait_ms >= 15.0,
            "blocked wait is released by a CPU Signal");
      value = 50;

      /* A wait that straddles a GPU signal scheduled after it started. */
      REQUIRE(wait_thread_start(&w, fence, value + 1), "wait thread 2");
      sleep_ms(10);
      REQUIRE(submit_copy(&c, queue, 0, &seed), "submit after wait started");
      ID3D12CommandQueue_Signal(queue, fence, value + 1);
      pthread_join(w.thread, NULL);
      CHECK(w.hr == S_OK && w.completed == value + 1 &&
               buffer_matches(c.dst, seed),
            "blocked wait picks up a GPU Signal scheduled later");
      value++;
   }

   /* 4. Multi-fence waits. */
   {
      ID3D12Fence *a = make_fence(c.dev, 0);
      ID3D12Fence *b = make_fence(c.dev, 0);
      ID3D12Fence *never = make_fence(c.dev, 0);
      REQUIRE(a && b && never, "multi fences");
      uint32_t seed;
      REQUIRE(submit_copy(&c, queue, 0, &seed), "submit for multi");
      ID3D12CommandQueue_Signal(queue, a, 1);
      REQUIRE(submit_copy(&c, queue, 1, &seed), "submit for multi b");
      ID3D12CommandQueue_Signal(queue, b, 1);
      ID3D12Fence *both[] = {a, b};
      UINT64 ones[] = {1, 1};
      hr = ID3D12Device1_SetEventOnMultipleFenceCompletion(
         c.dev1, both, ones, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(a) == 1 &&
               ID3D12Fence_GetCompletedValue(b) == 1 &&
               buffer_matches(c.dst, seed),
            "ALL wait on two GPU-signalled fences");

      REQUIRE(submit_copy(&c, queue, 0, &seed), "submit for any");
      ID3D12CommandQueue_Signal(queue, b, 2);
      ID3D12Fence *any[] = {never, b};
      UINT64 any_values[] = {1, 2};
      double t0 = now_ms();
      hr = ID3D12Device1_SetEventOnMultipleFenceCompletion(
         c.dev1, any, any_values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY,
         NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(b) == 2 &&
               ID3D12Fence_GetCompletedValue(never) == 0 &&
               now_ms() - t0 < 2000.0,
            "ANY wait returns on the GPU-signalled fence");

      hr = ID3D12Device1_SetEventOnMultipleFenceCompletion(
         c.dev1, any, any_values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY,
         NULL);
      CHECK(hr == S_OK, "ANY wait already satisfied returns at once");

      /* ANY released by a CPU Signal on the other fence. */
      UINT64 any_values2[] = {1, 3};
      ID3D12Fence *any2[] = {never, b};
      struct wait_thread w;
      REQUIRE(wait_thread_start(&w, never, 1), "never wait thread");
      sleep_ms(10);
      ID3D12Fence_Signal(never, 1);
      pthread_join(w.thread, NULL);
      CHECK(w.hr == S_OK && w.completed == 1, "CPU Signal releases a waiter");
      hr = ID3D12Device1_SetEventOnMultipleFenceCompletion(
         c.dev1, any2, any_values2, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY,
         NULL);
      CHECK(hr == S_OK, "ANY wait satisfied by the CPU-signalled fence");

      ID3D12Fence_Release(a);
      ID3D12Fence_Release(b);
      ID3D12Fence_Release(never);
   }

   /* 5. Cross-queue Wait then Signal through the submit worker. */
   {
      ID3D12CommandQueue *q2 = make_queue(c.dev);
      ID3D12Fence *f = make_fence(c.dev, 0);
      ID3D12Fence *g = make_fence(c.dev, 0);
      REQUIRE(q2 && f && g, "second queue");
      uint32_t seed;
      hr = ID3D12CommandQueue_Wait(q2, f, 1);
      CHECK(hr == S_OK, "queue Wait enqueued ahead of its Signal");
      REQUIRE(submit_copy(&c, q2, 0, &seed), "submit behind queue Wait");
      ID3D12CommandQueue_Signal(q2, g, 1);
      sleep_ms(10);
      CHECK(ID3D12Fence_GetCompletedValue(g) == 0,
            "work behind a queue Wait has not run");
      ID3D12CommandQueue_Signal(queue, f, 1);
      hr = ID3D12Fence_SetEventOnCompletion(g, 1, NULL);
      CHECK(hr == S_OK && buffer_matches(c.dst, seed),
            "queue Wait released by another queue's Signal");

      /* Release q2 while its worker is parked on a Wait that never
       * completes: teardown must cancel the parked wait promptly. */
      hr = ID3D12CommandQueue_Wait(q2, f, 1000);
      CHECK(hr == S_OK, "queue Wait that never completes");
      double t0 = now_ms();
      ID3D12CommandQueue_Release(q2);
      double release_ms = now_ms() - t0;
      CHECK(release_ms < 1000.0,
            "releasing a queue parked on a Wait returns promptly");
      ID3D12Fence_Release(f);
      ID3D12Fence_Release(g);
   }

   /* 6. Release the queue while a CPU waiter is parked on its timeline. */
   {
      bool all_ok = true;
      for (unsigned i = 0; i < 10 && all_ok; i++) {
         ID3D12CommandQueue *q3 = make_queue(c.dev);
         ID3D12Fence *f = make_fence(c.dev, 0);
         REQUIRE(q3 && f, "queue for release race");
         uint32_t seed;
         /* Several copies so the timeline point takes a while. */
         for (unsigned k = 0; k < SLOTS; k++)
            REQUIRE(submit_copy(&c, q3, k, &seed), "submit for release race");
         ID3D12CommandQueue_Signal(q3, f, 1);
         struct wait_thread w;
         REQUIRE(wait_thread_start(&w, f, 1), "release race wait thread");
         ID3D12CommandQueue_Release(q3);
         pthread_join(w.thread, NULL);
         all_ok = w.hr == S_OK && w.completed == 1 &&
                  buffer_matches(c.dst, seed);
         ID3D12Fence_Release(f);
      }
      CHECK(all_ok, "10 x queue released while a CPU waiter is parked on "
                    "its timeline");
   }

   hr = ID3D12Device_GetDeviceRemovedReason(c.dev);
   CHECK(hr == S_OK, "device healthy at end");

   ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   ID3D12Resource_Release(c.src);
   ID3D12Resource_Release(c.dst);
   for (unsigned i = 0; i < SLOTS; i++) {
      ID3D12GraphicsCommandList_Release(c.list[i]);
      ID3D12CommandAllocator_Release(c.alloc[i]);
   }
   ID3D12Device1_Release(c.dev1);
   ULONG refs = ID3D12Device_Release(c.dev);
   CHECK(refs == 0, "device references reach zero");

   printf("QUEUE FENCE LIVE: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures ? 1 : 0;
}
