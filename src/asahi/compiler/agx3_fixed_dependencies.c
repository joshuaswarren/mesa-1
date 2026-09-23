/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_fixed_dependencies.h"
#include <stdlib.h>
#include <string.h>

struct reader {
   uint32_t instruction;
   uint32_t next;
};

bool
agx3_fixed_dependencies(const struct agx3_fixed_access *accesses, size_t count,
                        unsigned tokens, agx3_fixed_dependency_cb callback,
                        void *data)
{
   if ((count && !accesses) || !callback || !tokens ||
       tokens > AGX3_FIXED_DEP_MAX_TOKENS)
      return false;

   size_t reader_count = 0;
   for (size_t i = 0; i < count; ++i) {
      const struct agx3_fixed_access *a = &accesses[i];
      if (!a->count || a->first >= tokens || a->count > tokens - a->first ||
          a->instruction == UINT32_MAX ||
          (i && a->instruction < accesses[i - 1].instruction))
         return false;
      if (!a->write) {
         if (reader_count > UINT32_MAX - a->count)
            return false;
         reader_count += a->count;
      }
   }
   if (reader_count > SIZE_MAX / sizeof(struct reader))
      return false;
   struct reader *readers = reader_count ? malloc(reader_count * sizeof(*readers))
                                         : NULL;
   if (reader_count && !readers)
      return false;

   uint32_t writer[AGX3_FIXED_DEP_MAX_TOKENS];
   uint32_t head[AGX3_FIXED_DEP_MAX_TOKENS];
   memset(writer, 0xff, tokens * sizeof(writer[0]));
   memset(head, 0xff, tokens * sizeof(head[0]));
   uint32_t next_reader = 0;
   for (size_t i = 0; i < count; ++i) {
      const struct agx3_fixed_access *a = &accesses[i];
      for (unsigned h = a->first; h < a->first + a->count; ++h) {
         if (writer[h] != UINT32_MAX && writer[h] != a->instruction)
            callback(data, writer[h], a->instruction);
         if (a->write) {
            for (uint32_t r = head[h]; r != UINT32_MAX; r = readers[r].next)
               if (readers[r].instruction != a->instruction)
                  callback(data, readers[r].instruction, a->instruction);
            head[h] = UINT32_MAX;
            writer[h] = a->instruction;
         } else {
            readers[next_reader] = (struct reader){a->instruction, head[h]};
            head[h] = next_reader++;
         }
      }
   }
   free(readers);
   return true;
}
