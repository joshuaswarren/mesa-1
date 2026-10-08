/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * No GPU access: exercise the Unix ABI handshake and Linux fd ownership.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../d12_wine_unixlib.h"
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr,"FAIL %s\n",#expr); return 1; } } while (0)
typedef uint32_t (*entry)(void *);
int main(int argc, char **argv)
{
   CHECK(argc == 2);
   void *lib = dlopen(argv[1],RTLD_NOW); CHECK(lib);
   entry *calls = dlsym(lib,"__wine_unix_call_funcs"); CHECK(calls);
   struct d12_wine_clock_args clock = {.monotonic_ns = D12_WINE_CLOCK_ABI_QUERY};
   CHECK(calls[D12_WINE_CALL_CLOCK](&clock) == 0);
   CHECK(clock.monotonic_ns == D12_WINE_CLOCK_ABI_VERSION);
   clock.monotonic_ns = 0;
   CHECK(calls[D12_WINE_CALL_CLOCK](&clock) == 0 && clock.monotonic_ns > 0);
   int fd = memfd_create("d12-unixlib-fd-test",MFD_CLOEXEC); CHECK(fd >= 0);
   CHECK(ftruncate(fd,32768) == 0);
   struct d12_wine_dmabuf_size_args seek = {.fd = fd};
   CHECK(calls[D12_WINE_CALL_DMABUF_SIZE](&seek) == 0 && seek.size == 32768);
   struct d12_wine_dupfd_args dup = {.fd = fd};
   CHECK(calls[D12_WINE_CALL_DUPFD_CLOEXEC](&dup) == 0 && dup.new_fd >= 3 && dup.new_fd != fd);
   struct d12_wine_close_args close_args = {.fd = fd};
   CHECK(calls[D12_WINE_CALL_CLOSE](&close_args) == 0);
   seek.fd = dup.new_fd;
   CHECK(calls[D12_WINE_CALL_DMABUF_SIZE](&seek) == 0 && seek.size == 32768);
   close_args.fd = dup.new_fd;
   CHECK(calls[D12_WINE_CALL_CLOSE](&close_args) == 0);
   CHECK(calls[D12_WINE_CALL_DMABUF_SIZE](&seek) == 0 && seek.size == -1 && seek.err != 0);
   dlclose(lib); puts("UNIXLIB_ABI_AND_FD_PASS"); return 0;
}
