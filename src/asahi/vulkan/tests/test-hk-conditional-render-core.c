/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "hk_conditional_render_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);         \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

static void
check_patch(const uint32_t *record, uint32_t words, uint32_t predicate,
            bool inverted, bool executes)
{
   struct {
      uint32_t before[2];
      uint32_t record[HK_CONDITIONAL_RENDER_MAX_WORDS];
      uint32_t after[2];
   } guarded;

   for (unsigned i = 0; i < 2; ++i) {
      guarded.before[i] = UINT32_C(0xa5a50000) + i;
      guarded.after[i] = UINT32_C(0x5a5a0000) + i;
   }
   for (unsigned i = 0; i < HK_CONDITIONAL_RENDER_MAX_WORDS; ++i)
      guarded.record[i] = UINT32_C(0xcccccccc);

   CHECK(hk_conditional_render_patch_words(guarded.record, record, words,
                                            predicate, inverted));
   for (unsigned i = 0; i < words; ++i)
      CHECK(guarded.record[i] == (executes ? record[i] : 0));
   for (unsigned i = words; i < HK_CONDITIONAL_RENDER_MAX_WORDS; ++i)
      CHECK(guarded.record[i] == UINT32_C(0xcccccccc));
   CHECK(guarded.before[0] == UINT32_C(0xa5a50000));
   CHECK(guarded.before[1] == UINT32_C(0xa5a50001));
   CHECK(guarded.after[0] == UINT32_C(0x5a5a0000));
   CHECK(guarded.after[1] == UINT32_C(0x5a5a0001));
}

static void
test_records_and_results(void)
{
   uint32_t direct[HK_CONDITIONAL_RENDER_MAX_WORDS] = {0};
   CHECK(hk_conditional_render_direct_draw_record(
            direct, false, 9, 3, 7, -123, 11) == 4);
   const uint32_t direct_expected[] = {9, 3, 7, 11};
   CHECK(memcmp(direct, direct_expected, sizeof(direct_expected)) == 0);

   uint32_t indexed[HK_CONDITIONAL_RENDER_MAX_WORDS] = {0};
   CHECK(hk_conditional_render_direct_draw_record(
            indexed, true, 13, 5, 17, -19, 23) == 5);
   const uint32_t indexed_expected[] = {13, 5, 17, UINT32_C(0xffffffed), 23};
   CHECK(memcmp(indexed, indexed_expected, sizeof(indexed_expected)) == 0);

   const uint32_t dispatch[] = {2, 3, 5};
   const struct {
      const uint32_t *record;
      uint32_t words;
   } records[] = {
      {dispatch, 3}, {direct, 4}, {indexed, 5},
   };

   for (unsigned i = 0; i < sizeof(records) / sizeof(records[0]); ++i) {
      check_patch(records[i].record, records[i].words, 0, false, false);
      check_patch(records[i].record, records[i].words, 7, false, true);
      check_patch(records[i].record, records[i].words, 0, true, true);
      check_patch(records[i].record, records[i].words, UINT32_MAX, true,
                  false);
   }

   CHECK(!hk_conditional_render_patch_words(NULL, direct, 4, 1, false));
   CHECK(!hk_conditional_render_patch_words(direct, NULL, 4, 1, false));
   CHECK(!hk_conditional_render_patch_words(direct, direct, 0, 1, false));
   CHECK(!hk_conditional_render_patch_words(direct, direct, 6, 1, false));
   CHECK(hk_conditional_render_direct_draw_record(NULL, false, 1, 1, 0, 0,
                                                   0) == 0);
}

static void
test_secondary_replay(void)
{
   const struct hk_conditional_render_state primary = {
      .address = UINT64_C(0x123456789abc),
      .active = true,
      .inverted = true,
   };

   CHECK(hk_conditional_render_is_active(&primary, false, false));
   CHECK(!hk_conditional_render_is_active(&primary, true, false));
   CHECK(hk_conditional_render_is_active(&primary, true, true));
   CHECK(!hk_conditional_render_is_active(NULL, false, false));

   struct hk_conditional_render_state replay =
      hk_conditional_render_replay_state(&primary, true);
   CHECK(replay.active);
   CHECK(replay.inverted);
   CHECK(replay.inherited);
   CHECK(replay.address == primary.address);

   replay = hk_conditional_render_replay_state(&primary, false);
   CHECK(!replay.active);
   CHECK(!replay.inverted);
   CHECK(!replay.inherited);
   CHECK(replay.address == 0);

   replay = hk_conditional_render_replay_state(NULL, true);
   CHECK(!replay.active && !replay.inherited && replay.address == 0);

   const struct hk_conditional_render_state inactive = {0};
   replay = hk_conditional_render_replay_state(&inactive, true);
   CHECK(!replay.active);
   CHECK(replay.inherited);
}

int
main(void)
{
   test_records_and_results();
   test_secondary_replay();

   if (failures) {
      fprintf(stderr, "hk-conditional-render-core: %u/%u checks failed\n",
              failures, checks);
      return EXIT_FAILURE;
   }

   printf("hk-conditional-render-core: %u checks passed\n", checks);
   return EXIT_SUCCESS;
}
