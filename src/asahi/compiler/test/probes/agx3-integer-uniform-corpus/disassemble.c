/* SPDX-License-Identifier: MIT */
#include "asahi/isa/disasm.h"

int main(int argc, char **argv)
{
   if (argc != 2)
      return 2;
   FILE *f = fopen(argv[1], "rb");
   if (!f)
      return 2;
   if (fseek(f, 0, SEEK_END)) {
      fclose(f);
      return 2;
   }
   long n = ftell(f);
   if (n < 0 || (unsigned long)n > SIZE_MAX - 16 || fseek(f, 0, SEEK_SET)) {
      fclose(f);
      return 2;
   }
   uint8_t *bytes = calloc((size_t)n + 16, 1);
   if (!bytes || fread(bytes, 1, n, f) != (size_t)n) {
      free(bytes);
      fclose(f);
      return 2;
   }
   fclose(f);
   bool errors = agx3b_disassemble(bytes, n, stdout);
   free(bytes);
   return errors;
}
