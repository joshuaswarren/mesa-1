/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_bo.h"
#include "agx_device.h"

/* A window like the one agx_open_device builds: 64 GiB of user VA. */
#define USER_START 0x2000000000ull
#define USER_SIZE  0x1000000000ull
#define PAGE       16384ull

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static void
open_va(struct agx_device *dev)
{
   memset(dev, 0, sizeof(*dev));
   simple_mtx_init(&dev->vma_lock, mtx_plain);
   dev->guard_size = PAGE;
   agx_va_init_user(dev, USER_START, USER_SIZE);
}

static void
close_va(struct agx_device *dev)
{
   agx_va_finish_user(dev);
   simple_mtx_destroy(&dev->vma_lock);
}

static bool
overlaps(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size)
{
   return a < b + b_size && b < a + a_size;
}

static bool
in_capture(const struct agx_device *dev, const struct agx_va *va)
{
   return overlaps(va->addr, va->size_B, dev->capture_start, dev->capture_size);
}

#define MAX_HELD 25000
static struct agx_va *held[MAX_HELD];
static unsigned nheld;

static void
free_held(struct agx_device *dev)
{
   for (unsigned i = 0; i < nheld; ++i)
      agx_va_free(dev, held[i], false);
   nheld = 0;
}

static void
test_range_shape(struct agx_device *dev)
{
   CHECK(dev->capture_size == AGX_CAPTURE_VA_SIZE_B);
   CHECK(dev->capture_size == (4ull << 30));
   CHECK(dev->capture_start >= USER_START);
   CHECK(dev->capture_start + dev->capture_size <= USER_START + USER_SIZE);
}

static void
test_ordinary_stays_out(struct agx_device *dev)
{
   srand(1);
   for (int i = 0; i < 20000; ++i) {
      uint64_t size = PAGE << (rand() % 8);
      struct agx_va *va = agx_va_alloc(dev, size, PAGE, 0, 0);
      CHECK(va != NULL);
      if (!va)
         break;
      CHECK(!in_capture(dev, va));
      held[nheld++] = va;
   }
   free_held(dev);
}

static void
test_ordinary_stays_out_when_full(struct agx_device *dev)
{
   for (;;) {
      struct agx_va *va = agx_va_alloc(dev, 1ull << 30, PAGE, 0, 0);
      if (!va)
         break;
      CHECK(!in_capture(dev, va));
      held[nheld++] = va;
   }
   /* 64 GiB window minus the 4 GiB range, in (1 GiB + guard) steps. */
   CHECK(nheld > 50 && nheld < 61);

   /* The capture range is untouched and still usable. */
   struct agx_va *cap = agx_va_alloc(dev, 1ull << 30, PAGE, AGX_VA_CAPTURE, 0);
   CHECK(cap != NULL);
   agx_va_free(dev, cap, false);
   free_held(dev);
}

static void
test_capture_inside_with_guard(struct agx_device *dev)
{
   const uint64_t end = dev->capture_start + dev->capture_size;

   for (int i = 0; i < 100; ++i) {
      uint64_t size = PAGE << (i % 7);
      struct agx_va *va = agx_va_alloc(dev, size, PAGE, AGX_VA_CAPTURE, 0);
      CHECK(va != NULL);
      if (!va)
         break;
      CHECK(va->addr >= dev->capture_start);
      CHECK(va->addr + va->size_B <= end);
      CHECK(va->addr % PAGE == 0);
      CHECK(va->size_B == size + dev->guard_size);
      CHECK(va->flags & AGX_VA_CAPTURE);
      for (unsigned j = 0; j < nheld; ++j)
         CHECK(!overlaps(va->addr, va->size_B, held[j]->addr, held[j]->size_B));
      held[nheld++] = va;
   }
   free_held(dev);
}

/* A capture in one device, a replay in a fresh one with ordinary allocations in
 * between, as the conformance test does.
 */
static void
test_replay_in_fresh_device(struct agx_device *dev)
{
   uint64_t sizes[100], addrs[100];

   srand(7);
   for (int i = 0; i < 100; ++i)
      sizes[i] = 4096ull << (rand() % 11);

   for (int i = 0; i < 100; ++i) {
      uint64_t padded = (sizes[i] + PAGE - 1) & ~(PAGE - 1);
      struct agx_va *va = agx_va_alloc(dev, padded, PAGE, AGX_VA_CAPTURE, 0);
      CHECK(va != NULL);
      if (!va)
         return;
      addrs[i] = va->addr;
      held[nheld++] = va;
   }
   free_held(dev);
   close_va(dev);

   open_va(dev);
   for (int i = 99; i >= 0; --i) {
      uint64_t padded = (sizes[i] + PAGE - 1) & ~(PAGE - 1);
      CHECK(agx_va_capture_valid(dev, addrs[i], padded, PAGE));
      struct agx_va *va = agx_va_alloc(
         dev, padded, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED, addrs[i]);
      CHECK(va != NULL);
      if (!va)
         continue;
      CHECK(va->addr == addrs[i]);
      held[nheld++] = va;

      struct agx_va *mem = agx_va_alloc(dev, padded, PAGE, 0, 0);
      CHECK(mem != NULL);
      if (mem)
         held[nheld++] = mem;
   }
   free_held(dev);
}

static void
test_invalid_requests(struct agx_device *dev)
{
   const uint64_t sz = PAGE * 4;
   const uint64_t start = dev->capture_start;
   const uint64_t end = start + dev->capture_size;

   CHECK(!agx_va_capture_valid(dev, end + PAGE, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start - PAGE, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, end - PAGE, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, UINT64_MAX - PAGE + 1, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, 0, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start + 4096, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start, sz, 0));
   /* A size that would wrap, and one that leaves no room for the guard. */
   CHECK(!agx_va_capture_valid(dev, start, UINT64_MAX - 100, PAGE));
   CHECK(!agx_va_capture_valid(dev, start, dev->capture_size, PAGE));
   CHECK(!agx_va_capture_valid(dev, end - sz, sz, PAGE));
   CHECK(agx_va_capture_valid(dev, end - sz - dev->guard_size, sz, PAGE));
   CHECK(agx_va_capture_valid(dev, start, sz, PAGE));

   CHECK(agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED,
                      end + PAGE) == NULL);
   CHECK(agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED,
                      UINT64_MAX - PAGE + 1) == NULL);
}

static void
test_occupied_request(struct agx_device *dev)
{
   const uint64_t sz = PAGE * 4;
   struct agx_va *a = agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE, 0);
   CHECK(a != NULL);
   if (!a)
      return;
   const uint64_t addr = a->addr;

   CHECK(agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED, addr) ==
         NULL);
   /* The guard page after the allocation is occupied too. */
   CHECK(agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED,
                      addr + sz) == NULL);

   agx_va_free(dev, a, false);
   struct agx_va *again =
      agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE | AGX_VA_FIXED, addr);
   CHECK(again != NULL);
   if (again) {
      CHECK(again->addr == addr);
      agx_va_free(dev, again, false);
   }
}

static unsigned
fill_capture(struct agx_device *dev)
{
   unsigned n = 0;
   for (;;) {
      struct agx_va *va = agx_va_alloc(dev, 1ull << 28, PAGE, AGX_VA_CAPTURE, 0);
      if (!va)
         return n;
      held[nheld++] = va;
      n++;
   }
}

static void
test_exhaustion_and_reuse(struct agx_device *dev)
{
   /* 4 GiB in (256 MiB + guard) steps. */
   CHECK(fill_capture(dev) == 15);

   struct agx_va *ord = agx_va_alloc(dev, 1ull << 30, PAGE, 0, 0);
   CHECK(ord != NULL);
   if (ord) {
      CHECK(!in_capture(dev, ord));
      agx_va_free(dev, ord, false);
   }

   /* Freeing returns every range to the capture heap. */
   free_held(dev);
   CHECK(fill_capture(dev) == 15);
   free_held(dev);
}

static void
test_freed_capture_range_stays_reserved(struct agx_device *dev)
{
   struct agx_va *cap = agx_va_alloc(dev, PAGE * 8, PAGE, AGX_VA_CAPTURE, 0);
   CHECK(cap != NULL);
   if (!cap)
      return;
   const uint64_t addr = cap->addr;
   agx_va_free(dev, cap, false);

   for (int i = 0; i < 5000; ++i) {
      struct agx_va *va = agx_va_alloc(dev, PAGE, PAGE, 0, 0);
      CHECK(va != NULL);
      if (!va)
         break;
      CHECK(va->addr != addr);
      CHECK(!in_capture(dev, va));
      held[nheld++] = va;
   }
   free_held(dev);
}

int
main(void)
{
   static const struct {
      const char *name;
      void (*fn)(struct agx_device *);
   } tests[] = {
      {"range shape", test_range_shape},
      {"ordinary stays out", test_ordinary_stays_out},
      {"ordinary stays out when full", test_ordinary_stays_out_when_full},
      {"capture inside with guard", test_capture_inside_with_guard},
      {"replay in fresh device", test_replay_in_fresh_device},
      {"invalid requests", test_invalid_requests},
      {"occupied request", test_occupied_request},
      {"exhaustion and reuse", test_exhaustion_and_reuse},
      {"freed capture range stays reserved",
       test_freed_capture_range_stays_reserved},
   };
   static struct agx_device dev;

   for (unsigned i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
      unsigned before = failures;
      open_va(&dev);
      tests[i].fn(&dev);
      close_va(&dev);
      printf("%s: %s\n", failures == before ? "PASS" : "FAIL", tests[i].name);
   }

   printf("%u failure(s)\n", failures);
   return failures != 0;
}
