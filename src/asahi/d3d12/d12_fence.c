/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12Fence userspace timeline.  Linux DRM timeline syncobjs are monotonic,
 * while D3D12 explicitly permits CPU and queue signals to move a fence value
 * backwards.  Queue work is ordered by d12_queue's worker and publishes its
 * completion here; CPU Signal publishes directly.
 */

#include "d12_private.h"
#include "drm-uapi/drm.h"

#include "util/os_time.h"
#include "util/u_thread.h"

#include <errno.h>

struct d12_fence_event_waiter {
   struct d12_fence_event_waiter *next;
   uint64_t value;
   HANDLE event;
   /* Internal waiters are told under fence->lock instead of through an
    * event; the multi-fence "any" wait uses this.  They stay on the list
    * until their registrant removes them: the registrant's handle is the
    * node's address, and a node freed by the fence could be reallocated
    * for another waiter before the registrant looks for it. */
   void (*notify)(void *cookie);
   void *cookie;
   bool fired;
};

static uint64_t
d12_placement_load(const volatile uint64_t *address)
{
#ifdef _WIN32
   return (uint64_t)InterlockedCompareExchange64(
      (volatile LONG64 *)(uintptr_t)address, 0, 0);
#else
   return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#endif
}

static void
d12_placement_store(volatile uint64_t *address, uint64_t value)
{
#ifdef _WIN32
   (void)InterlockedExchange64((volatile LONG64 *)(uintptr_t)address,
                               (LONG64)value);
#else
   __atomic_store_n(address, value, __ATOMIC_RELEASE);
#endif
}

HRESULT
d12_placement_fence_create(
   struct d12_device *device, uint32_t count,
   const struct d12_placement_fence_entry *entries,
   struct d12_placement_fence **out_fence)
{
   size_t size;
   struct d12_placement_fence *fence;

   if (!device || !count || count > 32 || !entries || !out_fence)
      return E_INVALIDARG;
   *out_fence = NULL;
   for (uint32_t index = 0; index < count; ++index) {
      if (!entries[index].current_gpu_address ||
          !entries[index].monitored_gpu_address ||
          !entries[index].current_cpu_address ||
          (entries[index].current_gpu_address & 7) ||
          (entries[index].monitored_gpu_address & 7))
         return E_INVALIDARG;
   }

   size = offsetof(struct d12_placement_fence, entries) +
          (size_t)count * sizeof(*entries);
   fence = calloc(1, size);
   if (!fence)
      return E_OUTOFMEMORY;
   if (mtx_init(&fence->lock, mtx_plain) != thrd_success) {
      free(fence);
      return E_FAIL;
   }
   if (cnd_init(&fence->changed) != thrd_success) {
      mtx_destroy(&fence->lock);
      free(fence);
      return E_FAIL;
   }
   fence->device = device;
   fence->lifetime_refcount = 1;
   fence->count = count;
   memcpy(fence->entries, entries, (size_t)count * sizeof(*entries));
   d12_device_add_ref(device);
   *out_fence = fence;
   return S_OK;
}

void
d12_placement_fence_ref(struct d12_placement_fence *fence)
{
   p_atomic_inc(&fence->lifetime_refcount);
}

void
d12_placement_fence_unref(struct d12_placement_fence *fence)
{
   if (p_atomic_dec_return(&fence->lifetime_refcount) != 0)
      return;
   for (uint32_t index = 0; index < fence->count; ++index) {
      if (fence->entries[index].release_lifetime)
         fence->entries[index].release_lifetime(
            fence->entries[index].lifetime_cookie);
   }
   cnd_destroy(&fence->changed);
   mtx_destroy(&fence->lock);
   d12_device_release(fence->device);
   free(fence);
}

void
d12_placement_fence_destroy(struct d12_placement_fence *fence)
{
   if (!fence)
      return;
   /* Queue Wait/Signal operations already hold renderer lifetime references.
    * Destroying the runtime-owned DDI wrapper must not cancel those ordered
    * operations; queue/device shutdown supplies their explicit cancel path. */
   d12_placement_fence_unref(fence);
}

bool
d12_placement_fence_wait(struct d12_placement_fence *fence, uint64_t value,
                         const uint32_t *cancel)
{
   bool completed = false;

   mtx_lock(&fence->lock);
   for (;;) {
      completed = true;
      for (uint32_t index = 0; index < fence->count; ++index) {
         if (d12_placement_load(
                fence->entries[index].current_cpu_address) < value) {
            completed = false;
            break;
         }
      }
      if (completed || (cancel && p_atomic_read(cancel)))
         break;

      /* A signal from another process is a coherent GPU/CPU memory update,
       * not a condition-variable notification.  The bounded timed wait is
       * therefore both the local fast path and the external-progress poll. */
      struct timespec deadline;
      timespec_get(&deadline, TIME_UTC);
      deadline.tv_nsec += 1000 * 1000;
      if (deadline.tv_nsec >= 1000 * 1000 * 1000) {
         deadline.tv_sec++;
         deadline.tv_nsec -= 1000 * 1000 * 1000;
      }
      cnd_timedwait(&fence->changed, &fence->lock, &deadline);
   }
   mtx_unlock(&fence->lock);
   return completed;
}

/*
 * A wait that a command-queue worker may need to abandon at queue teardown.
 * On Linux the teardown path raises the stop flag and then broadcasts the
 * fence the worker is parked on (d12_fence_wake_waiters), so a plain
 * condition wait is sufficient.  Windows keeps the bounded wait that the
 * queue primitives also use, as insurance against the lost wakes observed
 * on that platform's CONDITION_VARIABLE.
 */
static void
d12_fence_cancelable_wait(struct d12_fence *fence)
{
#ifdef _WIN32
   struct timespec deadline;
   timespec_get(&deadline, TIME_UTC);
   deadline.tv_nsec += 10 * 1000 * 1000;
   if (deadline.tv_nsec >= 1000 * 1000 * 1000) {
      deadline.tv_sec++;
      deadline.tv_nsec -= 1000 * 1000 * 1000;
   }
   cnd_timedwait(&fence->changed, &fence->lock, &deadline);
#else
   cnd_wait(&fence->changed, &fence->lock);
#endif
}

void
d12_fence_wake_waiters(struct d12_fence *fence)
{
   mtx_lock(&fence->lock);
   cnd_broadcast(&fence->changed);
   mtx_unlock(&fence->lock);
}

static void
d12_fence_signal_waiters_locked(struct d12_fence *fence, bool all)
{
   struct d12_fence_event_waiter **link = &fence->event_waiters;
   while (*link) {
      struct d12_fence_event_waiter *waiter = *link;
      if (!all && p_atomic_read(&fence->completed_value) < waiter->value) {
         link = &waiter->next;
         continue;
      }

      if (waiter->notify) {
         if (!waiter->fired) {
            waiter->fired = true;
            waiter->notify(waiter->cookie);
         }
         link = &waiter->next;
         continue;
      }

      *link = waiter->next;
      d12_platform_signal_event(waiter->event);
      free(waiter);
   }
}

/* Registers an internal waiter.  Returns 1 when the value is already
 * reached (or the fence destroyed) and nothing was registered, 0 when
 * *handle identifies the waiter for d12_fence_notify_unregister, and -1
 * when out of memory. */
static int
d12_fence_notify_register(struct d12_fence *fence, uint64_t value,
                          void (*notify)(void *cookie), void *cookie,
                          struct d12_fence_event_waiter **handle)
{
   struct d12_fence_event_waiter *waiter = calloc(1, sizeof(*waiter));
   *handle = NULL;
   if (!waiter)
      return -1;
   waiter->value = value;
   waiter->notify = notify;
   waiter->cookie = cookie;

   mtx_lock(&fence->lock);
   if (fence->destroyed ||
       p_atomic_read(&fence->completed_value) >= value) {
      mtx_unlock(&fence->lock);
      free(waiter);
      return 1;
   }
   waiter->next = fence->event_waiters;
   fence->event_waiters = waiter;
   mtx_unlock(&fence->lock);
   *handle = waiter;
   return 0;
}

/* Only the registrant frees an internal waiter, fired or not, so the
 * handle is always on the list here. */
static void
d12_fence_notify_unregister(struct d12_fence *fence,
                            struct d12_fence_event_waiter *handle)
{
   if (!handle)
      return;
   mtx_lock(&fence->lock);
   struct d12_fence_event_waiter **link = &fence->event_waiters;
   while (*link && *link != handle)
      link = &(*link)->next;
   assert(*link == handle);
   if (*link == handle)
      *link = handle->next;
   mtx_unlock(&fence->lock);
   free(handle);
}

static void
d12_fence_satisfy_tickets_locked(struct d12_fence *fence, bool all)
{
   struct d12_fence_ticket **link = &fence->tickets;
   while (*link) {
      struct d12_fence_ticket *ticket = *link;
      if (!all && p_atomic_read(&fence->completed_value) < ticket->value) {
         link = &ticket->next;
         continue;
      }

      *link = ticket->next;
      ticket->registered = false;
      p_atomic_set(&ticket->satisfied, 1);
   }
}

static HRESULT STDMETHODCALLTYPE
d12_fence_QueryInterface(ID3D12Fence *iface, REFIID riid, void **out)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12Fence)) {
      p_atomic_inc(&fence->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&fence->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12Fence::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_fence_AddRef(ID3D12Fence *iface)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);
   return p_atomic_inc_return(&fence->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_fence_Release(ID3D12Fence *iface)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&fence->refcount);

   if (refs == 0) {
      /* Releasing the public object unblocks queue and event waiters, just
       * like closing the native D3D12 fence handle.  Those waiters hold
       * private lifetime references which deliberately do not affect the
       * COM refcount returned to applications. */
      mtx_lock(&fence->lock);
      fence->destroyed = true;
      d12_fence_signal_waiters_locked(fence, true);
      d12_fence_satisfy_tickets_locked(fence, true);
      cnd_broadcast(&fence->changed);
      mtx_unlock(&fence->lock);
      d12_fence_internal_unref(fence);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_fence, ID3D12Fence,
                                d12_fence_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_fence_GetDevice(ID3D12Fence *iface, REFIID riid, void **out)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);
   return ID3D12Device_QueryInterface(&fence->device->iface, riid, out);
}

static UINT64 STDMETHODCALLTYPE
d12_fence_GetCompletedValue(ID3D12Fence *iface)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);
   /*
    * This is a polling API.  Taking the waiter-list mutex here allowed a
    * tight application poll loop to repeatedly steal the lock from command
    * queue workers publishing progress.  Besides turning a read into a
    * scalability bottleneck, that could make a live wait-before-signal chain
    * look deadlocked.  The value itself is independent of the protected
    * waiter/ticket lists, so publish and read it atomically.
    */
   uint64_t point = p_atomic_read(&fence->completed_value);
   if (fence->shared_syncobj) {
      uint64_t shared_point;
      if (SUCCEEDED(d12_platform_query_shared_sync(
             fence->device, fence->shared_syncobj, &shared_point))) {
         point = shared_point;
         p_atomic_set(&fence->completed_value, point);
      }
   }

   if (d12_debug()->dump_sync)
      mesa_logi("SYNC GetCompletedValue: fence %p -> %llu", (void *)fence,
                (unsigned long long)point);

   return point;
}

void
d12_fence_internal_ref(struct d12_fence *fence)
{
   p_atomic_inc(&fence->lifetime_refcount);
}

void
d12_fence_internal_unref(struct d12_fence *fence)
{
   if (p_atomic_dec_return(&fence->lifetime_refcount) != 0)
      return;

   if (fence->shared_syncobj)
      fence->device->sync->destroy(fence->device->sync,
                                   fence->shared_syncobj);
   cnd_destroy(&fence->changed);
   mtx_destroy(&fence->lock);
   d12_private_data_store_destroy(&fence->private_data);
   d12_device_release(fence->device);
   free(fence);
}

static void
d12_fence_signal_value_locked(struct d12_fence *fence, uint64_t value)
{
   if (fence->destroyed)
      return;

   uint64_t old = p_atomic_read(&fence->completed_value);

   /*
    * Registration checks the current value while holding this same lock,
    * so an idempotent signal cannot make any existing waiter or ticket
    * newly ready.  Avoid rescanning thousands of queued wait tickets on
    * every iteration of an application's fence-rewind polling loop.
    */
   if (old != value) {
      p_atomic_set(&fence->completed_value, value);
      d12_fence_signal_waiters_locked(fence, false);
      d12_fence_satisfy_tickets_locked(fence, false);
      cnd_broadcast(&fence->changed);
   }
}

HRESULT
d12_fence_signal_value(struct d12_fence *fence, uint64_t value)
{
   if (fence->shared_syncobj) {
      uint32_t handle = fence->shared_syncobj;
      uint64_t point = value;
      if (fence->device->sync->timeline_signal(
             fence->device->sync, &handle, &point, 1)) {
         d12_device_mark_removed(fence->device,
                                 DXGI_ERROR_DEVICE_REMOVED);
         return DXGI_ERROR_DEVICE_REMOVED;
      }
   }

   mtx_lock(&fence->lock);
   d12_fence_signal_value_locked(fence, value);
   mtx_unlock(&fence->lock);
   return S_OK;
}

void
d12_fence_pending_signal_add(struct d12_fence *fence,
                             struct d12_fence_pending_signal *pending,
                             struct d12_queue *queue, uint64_t point,
                             uint64_t value)
{
   /* A shared fence is published through its syncobj by the retirement
    * worker; CPU waiters already wait on that syncobj directly. */
   if (fence->shared_syncobj)
      return;

   pending->next = NULL;
   pending->queue = queue;
   pending->point = point;
   pending->value = value;
   pending->published = false;

   /* Append so the list is in scheduling order: for one queue that is
    * ascending point order, which d12_fence_publish_completed_locked
    * relies on. */
   mtx_lock(&fence->lock);
   struct d12_fence_pending_signal **link = &fence->pending_signals;
   while (*link)
      link = &(*link)->next;
   *link = pending;
   pending->registered = true;
   cnd_broadcast(&fence->changed);
   mtx_unlock(&fence->lock);
}

/* Publishes every unpublished pending signal from queue whose point is at or
 * before the one just observed complete, in queue order.  Both the
 * retirement worker and a direct CPU waiter call this; each record is
 * published once, by whichever gets there first. */
static void
d12_fence_publish_completed_locked(struct d12_fence *fence,
                                   struct d12_queue *queue, uint64_t point)
{
   for (struct d12_fence_pending_signal *p = fence->pending_signals; p;
        p = p->next) {
      if (p->queue == queue && p->point <= point && !p->published) {
         p->published = true;
         d12_fence_signal_value_locked(fence, p->value);
      }
   }
}

void
d12_fence_pending_signal_retire(struct d12_fence *fence,
                                struct d12_fence_pending_signal *pending,
                                bool completed)
{
   if (!pending->registered) {
      /* Shared fences (never registered) publish through their syncobj. */
      if (completed && fence->shared_syncobj)
         d12_fence_signal_value(fence, pending->value);
      return;
   }

   mtx_lock(&fence->lock);
   if (completed)
      d12_fence_publish_completed_locked(fence, pending->queue,
                                         pending->point);
   struct d12_fence_pending_signal **link = &fence->pending_signals;
   while (*link && *link != pending)
      link = &(*link)->next;
   if (*link == pending)
      *link = pending->next;
   pending->registered = false;
   pending->next = NULL;
   /* A direct waiter parked on this record must re-evaluate. */
   cnd_broadcast(&fence->changed);
   mtx_unlock(&fence->lock);
}

/* The first unpublished GPU signal a wait for value can ride on. */
static struct d12_fence_pending_signal *
d12_fence_find_pending_locked(struct d12_fence *fence, uint64_t value)
{
   for (struct d12_fence_pending_signal *p = fence->pending_signals; p;
        p = p->next) {
      if (!p->published && p->value >= value)
         return p;
   }
   return NULL;
}

/*
 * Wait for a pending GPU signal's timeline point from the calling thread.
 * Called and returns with fence->lock held; the lock is dropped around the
 * kernel wait.  The waiter is counted on the queue across the wait so its
 * syncobj cannot be destroyed (and the handle reused) underneath it.
 * Returns false when the kernel wait failed, in which case the caller must
 * not retry the same record without sleeping: the retirement worker will
 * either publish or unlink it, and both broadcast fence->changed.
 */
static bool
d12_fence_wait_pending_locked(struct d12_fence *fence,
                              struct d12_fence_pending_signal *pending)
{
   struct d12_queue *queue = pending->queue;
   const uint64_t point = pending->point;
   const uint64_t value = pending->value;

   d12_queue_direct_wait_begin(queue);
   mtx_unlock(&fence->lock);

   const int64_t start = d12_debug()->dump_sync ? os_time_get_nano() : 0;
   const bool reached = d12_queue_wait_timeline(queue, point);
   if (d12_debug()->dump_sync) {
      mesa_logi("SYNC wait direct: fence %p value %llu queue %p point %llu "
                "waited %lld us%s",
                (void *)fence, (unsigned long long)value, (void *)queue,
                (unsigned long long)point,
                (long long)(os_time_get_nano() - start) / 1000,
                reached ? "" : " (failed)");
   }

   mtx_lock(&fence->lock);
   if (reached)
      d12_fence_publish_completed_locked(fence, queue, point);
   mtx_unlock(&fence->lock);

   /* Teardown may free the queue as soon as this returns; it takes
    * queue->lock, so it is not called under fence->lock. */
   d12_queue_direct_wait_end(queue);
   mtx_lock(&fence->lock);
   return reached;
}

bool
d12_fence_wait_value(struct d12_fence *fence, uint64_t value,
                     const uint32_t *cancel)
{
   if (fence->shared_syncobj) {
      for (;;) {
         mtx_lock(&fence->lock);
         bool destroyed = fence->destroyed;
         mtx_unlock(&fence->lock);
         if (destroyed)
            return false;
         if (d12_fence_GetCompletedValue(&fence->iface) >= value)
            return true;
         if (cancel && p_atomic_read(cancel))
            return false;

         /* A cancelable wait is bounded so the stop flag is re-read; it
          * still blocks in the kernel rather than sleeping and polling.
          * Provider timeouts are absolute CLOCK_MONOTONIC deadlines.  The
          * point may not have been submitted by the other process yet: a
          * plain timeline wait on DRM refuses such a point outright
          * (EINVAL, not a timeout), which would let a queue Wait fall
          * through before its signal; wait for the submission as well. */
         uint32_t handle = fence->shared_syncobj;
         uint64_t point = value;
         const int64_t timeout_ns =
            cancel ? os_time_get_nano() + 10 * 1000 * 1000 : INT64_MAX;
         int ret = fence->device->sync->timeline_wait(
            fence->device->sync, &handle, &point, 1, timeout_ns,
            DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
         if (ret && !(cancel && (ret == -ETIME || errno == ETIME)))
            return false;
      }
   }

   bool completed = true;
   struct d12_fence_pending_signal *failed = NULL;

   mtx_lock(&fence->lock);
   while (!fence->destroyed &&
          p_atomic_read(&fence->completed_value) < value) {
      if (!cancel) {
         /* A value pending on a queue Signal is waited for on the queue's
          * timeline directly rather than through the retirement worker's
          * hop; anything else (a CPU Signal, a Signal not yet scheduled
          * behind a queue Wait) arrives through the condition variable. */
         struct d12_fence_pending_signal *pending =
            d12_fence_find_pending_locked(fence, value);
         if (pending && pending != failed) {
            if (!d12_fence_wait_pending_locked(fence, pending))
               failed = pending;
            continue;
         }
         cnd_wait(&fence->changed, &fence->lock);
         continue;
      }

      if (p_atomic_read(cancel)) {
         completed = false;
         break;
      }

      d12_fence_cancelable_wait(fence);
   }
   mtx_unlock(&fence->lock);

   return completed;
}

void
d12_fence_ticket_init(struct d12_fence *fence,
                      struct d12_fence_ticket *ticket, uint64_t value)
{
   memset(ticket, 0, sizeof(*ticket));
   ticket->fence = fence;
   ticket->value = value;

   if (fence->shared_syncobj) {
      if (d12_fence_GetCompletedValue(&fence->iface) >= value)
         p_atomic_set(&ticket->satisfied, 1);
      return;
   }

   mtx_lock(&fence->lock);
   if (fence->destroyed ||
       p_atomic_read(&fence->completed_value) >= value) {
      p_atomic_set(&ticket->satisfied, 1);
   } else {
      ticket->next = fence->tickets;
      fence->tickets = ticket;
      ticket->registered = true;
   }
   mtx_unlock(&fence->lock);
}

void
d12_fence_ticket_fini(struct d12_fence_ticket *ticket)
{
   struct d12_fence *fence = ticket->fence;
   if (!fence)
      return;

   mtx_lock(&fence->lock);
   if (ticket->registered) {
      struct d12_fence_ticket **link = &fence->tickets;
      while (*link && *link != ticket)
         link = &(*link)->next;
      if (*link == ticket)
         *link = ticket->next;
      ticket->registered = false;
   }
   mtx_unlock(&fence->lock);
   ticket->fence = NULL;
}

bool
d12_fence_ticket_wait(struct d12_fence_ticket *ticket,
                      const uint32_t *cancel)
{
   struct d12_fence *fence = ticket->fence;
   if (fence->shared_syncobj) {
      bool completed = d12_fence_wait_value(fence, ticket->value, cancel);
      if (completed)
         p_atomic_set(&ticket->satisfied, 1);
      return completed;
   }

   bool completed = true;

   mtx_lock(&fence->lock);
   while (!p_atomic_read(&ticket->satisfied) && !fence->destroyed) {
      if (cancel && p_atomic_read(cancel)) {
         completed = false;
         break;
      }

      if (!cancel) {
         cnd_wait(&fence->changed, &fence->lock);
         continue;
      }

      d12_fence_cancelable_wait(fence);
   }
   mtx_unlock(&fence->lock);
   return completed;
}

struct d12_fence_shared_event_waiter {
   struct d12_fence *fence;
   uint64_t value;
   HANDLE event;
   thrd_t thread;
};

static int
d12_fence_shared_event_wait(void *data)
{
   struct d12_fence_shared_event_waiter *waiter = data;
   if (d12_fence_wait_value(waiter->fence, waiter->value, NULL))
      d12_platform_signal_event(waiter->event);
   d12_fence_internal_unref(waiter->fence);
   free(waiter);
   return 0;
}

static HRESULT STDMETHODCALLTYPE
d12_fence_SetEventOnCompletion(ID3D12Fence *iface, UINT64 value,
                               HANDLE event)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);

   if (fence->shared_syncobj) {
      if (!event)
         return d12_fence_wait_value(fence, value, NULL) ? S_OK : E_FAIL;
      if (d12_fence_GetCompletedValue(iface) >= value) {
         d12_platform_signal_event(event);
         return S_OK;
      }

      struct d12_fence_shared_event_waiter *waiter =
         calloc(1, sizeof(*waiter));
      if (!waiter)
         return E_OUTOFMEMORY;
      waiter->fence = fence;
      waiter->value = value;
      waiter->event = event;
      d12_fence_internal_ref(fence);
      if (thrd_create(&waiter->thread, d12_fence_shared_event_wait, waiter) !=
          thrd_success) {
         d12_fence_internal_unref(fence);
         free(waiter);
         return E_FAIL;
      }
      thrd_detach(waiter->thread);
      return S_OK;
   }

   if (event != NULL) {
      struct d12_fence_event_waiter *waiter = calloc(1, sizeof(*waiter));
      if (!waiter)
         return E_OUTOFMEMORY;
      waiter->value = value;
      waiter->event = event;

      mtx_lock(&fence->lock);
      if (fence->destroyed ||
          p_atomic_read(&fence->completed_value) >= value) {
         d12_platform_signal_event(waiter->event);
         free(waiter);
      } else {
         waiter->next = fence->event_waiters;
         fence->event_waiters = waiter;
      }
      mtx_unlock(&fence->lock);
      return S_OK;
   }

   /* NULL event: block until the fence reaches the value. */
   d12_fence_wait_value(fence, value, NULL);
   return S_OK;
}

struct d12_fence_multi_waiter {
   struct d12_device *device;
   struct d12_fence **fences;
   uint64_t *values;
   unsigned count;
   bool wait_all;
   HANDLE event;
   thrd_t thread;
};

struct d12_fence_any_waiter {
   mtx_t lock;
   cnd_t changed;
   bool fired;
};

static void
d12_fence_any_notify(void *cookie)
{
   struct d12_fence_any_waiter *any = cookie;
   mtx_lock(&any->lock);
   any->fired = true;
   cnd_broadcast(&any->changed);
   mtx_unlock(&any->lock);
}

static int
d12_fence_multi_wait(struct d12_fence_multi_waiter *w)
{
   if (w->wait_all) {
      /* Each wait takes the direct timeline path when its value is pending
       * on a queue Signal.  A destroyed fence counts as reached, as the
       * native runtime treats a closed fence handle. */
      for (unsigned i = 0; i < w->count; i++)
         (void)d12_fence_wait_value(w->fences[i], w->values[i], NULL);
      return 0;
   }

   /* Wait for any: park on one condition variable that every fence's
    * internal waiter notifies.  Registration is checked against the current
    * value under each fence's lock, so a value reached before or after the
    * registration both fire. */
   struct d12_fence_any_waiter any = {.fired = false};
   if (mtx_init(&any.lock, mtx_plain) != thrd_success)
      return -1;
   if (cnd_init(&any.changed) != thrd_success) {
      mtx_destroy(&any.lock);
      return -1;
   }

   struct d12_fence_event_waiter **handles =
      calloc(w->count, sizeof(*handles));
   if (!handles) {
      cnd_destroy(&any.changed);
      mtx_destroy(&any.lock);
      return -1;
   }

   /* A shared fence's value moves in another process; its cached value
    * only refreshes through GetCompletedValue, so such fences are polled
    * (refreshed before registration and on a bounded wait) as the
    * cancelable wait above does.  Local fences notify. */
   bool any_shared = false;
   bool reached = false;
   int ret = 0;
   for (unsigned i = 0; i < w->count && !reached && !ret; i++) {
      if (w->fences[i]->shared_syncobj) {
         any_shared = true;
         if (d12_fence_GetCompletedValue(&w->fences[i]->iface) >=
             w->values[i]) {
            reached = true;
            break;
         }
      }
      ret = d12_fence_notify_register(w->fences[i], w->values[i],
                                      d12_fence_any_notify, &any,
                                      &handles[i]);
      reached = ret > 0;
      if (ret > 0)
         ret = 0;
   }

   while (!reached && !ret) {
      mtx_lock(&any.lock);
      if (!any.fired) {
         if (any_shared) {
            struct timespec deadline;
            timespec_get(&deadline, TIME_UTC);
            deadline.tv_nsec += 1000 * 1000;
            if (deadline.tv_nsec >= 1000 * 1000 * 1000) {
               deadline.tv_sec++;
               deadline.tv_nsec -= 1000 * 1000 * 1000;
            }
            cnd_timedwait(&any.changed, &any.lock, &deadline);
         } else {
            cnd_wait(&any.changed, &any.lock);
         }
      }
      reached = any.fired;
      mtx_unlock(&any.lock);
      if (reached || !any_shared)
         continue;
      for (unsigned i = 0; i < w->count && !reached; i++) {
         if (w->fences[i]->shared_syncobj &&
             d12_fence_GetCompletedValue(&w->fences[i]->iface) >=
                w->values[i])
            reached = true;
      }
   }

   /* Every registration is removed before the cookie goes out of scope. */
   for (unsigned i = 0; i < w->count; i++)
      d12_fence_notify_unregister(w->fences[i], handles[i]);
   free(handles);
   cnd_destroy(&any.changed);
   mtx_destroy(&any.lock);
   return ret;
}

static void
d12_fence_multi_waiter_destroy(struct d12_fence_multi_waiter *w)
{
   for (unsigned i = 0; i < w->count; i++)
      d12_fence_internal_unref(w->fences[i]);

   free(w->values);
   free(w->fences);
   free(w);
}

static int
d12_fence_multi_wait_thread(void *data)
{
   struct d12_fence_multi_waiter *w = data;
   int ret = d12_fence_multi_wait(w);

   if (ret)
      mesa_loge("multiple fence event wait failed: %d", ret);

   d12_platform_signal_event(w->event);

   d12_fence_multi_waiter_destroy(w);
   return 0;
}

HRESULT
d12_fence_set_event_multiple(struct d12_device *device,
                             ID3D12Fence *const *fences,
                             const UINT64 *values, UINT count,
                             D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags,
                             HANDLE event)
{
   if (!fences || !values || count == 0 ||
       (flags != D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL &&
        flags != D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY))
      return E_INVALIDARG;

   struct d12_fence_multi_waiter *w = calloc(1, sizeof(*w));
   if (!w)
      return E_OUTOFMEMORY;

   w->fences = calloc(count, sizeof(*w->fences));
   w->values = calloc(count, sizeof(*w->values));
   if (!w->fences || !w->values) {
      free(w->values);
      free(w->fences);
      free(w);
      return E_OUTOFMEMORY;
   }

   w->device = device;
   w->count = count;
   w->wait_all = flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL;
   w->event = event;

   for (unsigned i = 0; i < count; i++) {
      if (!fences[i]) {
         w->count = i;
         d12_fence_multi_waiter_destroy(w);
         return E_INVALIDARG;
      }

      struct d12_fence *fence = d12_fence_from_iface(fences[i]);
      if (fence->device != device) {
         w->count = i;
         d12_fence_multi_waiter_destroy(w);
         return E_INVALIDARG;
      }

      d12_fence_internal_ref(fence);
      w->fences[i] = fence;
      w->values[i] = values[i];
   }

   if (!event) {
      int ret = d12_fence_multi_wait(w);
      d12_fence_multi_waiter_destroy(w);
      if (ret) {
         mesa_loge("multiple fence wait failed: %d", ret);
         return E_FAIL;
      }
      return S_OK;
   }

   if (thrd_create(&w->thread, d12_fence_multi_wait_thread, w) !=
       thrd_success) {
      d12_fence_multi_waiter_destroy(w);
      return E_FAIL;
   }

   thrd_detach(w->thread);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_fence_Signal(ID3D12Fence *iface, UINT64 value)
{
   struct d12_fence *fence = d12_fence_from_iface(iface);
   return d12_fence_signal_value(fence, value);
}

static const ID3D12FenceVtbl d12_fence_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12Fence_DEFAULTS,
   .QueryInterface = d12_fence_QueryInterface,
   .AddRef = d12_fence_AddRef,
   .Release = d12_fence_Release,
   .GetPrivateData = d12_fence_GetPrivateData,
   .SetPrivateData = d12_fence_SetPrivateData,
   .SetPrivateDataInterface = d12_fence_SetPrivateDataInterface,
   .SetName = d12_fence_SetName,
   .GetDevice = d12_fence_GetDevice,
   .GetCompletedValue = d12_fence_GetCompletedValue,
   .SetEventOnCompletion = d12_fence_SetEventOnCompletion,
   .Signal = d12_fence_Signal,
};

static HRESULT
d12_fence_create_internal(struct d12_device *device, UINT64 initial_value,
                          D3D12_FENCE_FLAGS flags, uint32_t shared_syncobj,
                          REFIID riid, void **out)
{
   /* CreateFence is the odd D3D12 creation call whose NULL output contract is
    * S_FALSE, even when riid itself is unsupported. QueryInterface retains
    * the normal E_POINTER rule; this applies only to creation.
    */
   if (!out)
      return S_FALSE;

   struct d12_fence *fence = calloc(1, sizeof(*fence));
   if (!fence) {
      if (shared_syncobj)
         device->sync->destroy(device->sync, shared_syncobj);
      return E_OUTOFMEMORY;
   }

   if (mtx_init(&fence->lock, mtx_plain) != thrd_success) {
      if (shared_syncobj)
         device->sync->destroy(device->sync, shared_syncobj);
      free(fence);
      return E_FAIL;
   }
   if (cnd_init(&fence->changed) != thrd_success) {
      if (shared_syncobj)
         device->sync->destroy(device->sync, shared_syncobj);
      mtx_destroy(&fence->lock);
      free(fence);
      return E_FAIL;
   }

   fence->iface.lpVtbl = (ID3D12FenceVtbl *)&d12_fence_vtbl;
   fence->refcount = 1;
   fence->lifetime_refcount = 1;
   fence->device = device;
   fence->flags = flags;
   fence->shared_syncobj = shared_syncobj;
   p_atomic_set(&fence->completed_value, initial_value);
   d12_private_data_store_init(&fence->private_data,
                               (IUnknown *)&fence->iface);
   d12_device_add_ref(device);

   HRESULT hr = d12_fence_QueryInterface(&fence->iface, riid, out);
   d12_fence_Release(&fence->iface);
   return hr;
}

HRESULT
d12_fence_create(struct d12_device *device, UINT64 initial_value,
                 D3D12_FENCE_FLAGS flags, REFIID riid, void **out)
{
   if (!out)
      return S_FALSE;
   *out = NULL;

   if (flags != D3D12_FENCE_FLAG_NONE &&
       flags != D3D12_FENCE_FLAG_SHARED) {
      mesa_loge("d12_fence_create: fence flags 0x%x not implemented",
                (unsigned)flags);
      return E_NOTIMPL;
   }

   uint32_t shared_syncobj = 0;
   if (flags == D3D12_FENCE_FLAG_SHARED) {
      HRESULT hr = d12_platform_create_shared_sync(
         device, initial_value, &shared_syncobj);
      if (FAILED(hr))
         return hr;
   }

   return d12_fence_create_internal(device, initial_value, flags,
                                    shared_syncobj, riid, out);
}

HRESULT
d12_fence_create_shared_handle(
   struct d12_device *device, ID3D12DeviceChild *object,
   const SECURITY_ATTRIBUTES *attributes, DWORD access, LPCWSTR name,
   HANDLE *handle)
{
   if (!handle)
      return E_POINTER;
   *handle = NULL;
   if (!object)
      return E_INVALIDARG;

   ID3D12Fence *fence_iface = NULL;
   HRESULT hr = ID3D12DeviceChild_QueryInterface(
      object, &IID_ID3D12Fence, (void **)&fence_iface);
   if (FAILED(hr))
      return E_NOTIMPL;

   if (fence_iface->lpVtbl != &d12_fence_vtbl) {
      ID3D12Fence_Release(fence_iface);
      return DXGI_ERROR_INVALID_CALL;
   }

   struct d12_fence *fence = d12_fence_from_iface(fence_iface);
   if (fence->device != device ||
       !(fence->flags & D3D12_FENCE_FLAG_SHARED) ||
       !fence->shared_syncobj) {
      ID3D12Fence_Release(fence_iface);
      return DXGI_ERROR_INVALID_CALL;
   }

   hr = d12_platform_export_shared_sync(
      device, fence->shared_syncobj, attributes, access, name, handle);
   ID3D12Fence_Release(fence_iface);
   return hr;
}

HRESULT
d12_fence_open_shared_handle(struct d12_device *device, HANDLE handle,
                             REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;
   if (!riid || d12_handle_is_invalid(handle))
      return E_INVALIDARG;

   uint32_t syncobj = 0;
   HRESULT hr = d12_platform_import_shared_sync(device, handle, &syncobj);
   if (FAILED(hr))
      return hr;

   uint64_t current = 0;
   hr = d12_platform_query_shared_sync(device, syncobj, &current);
   if (FAILED(hr)) {
      device->sync->destroy(device->sync, syncobj);
      return hr;
   }

   return d12_fence_create_internal(device, current,
                                    D3D12_FENCE_FLAG_SHARED, syncobj, riid,
                                    out);
}
