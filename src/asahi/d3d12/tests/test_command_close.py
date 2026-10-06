#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Inject finalization failures into the production Close state transition.

Compile the method from d12_cmd.c with stand-in segment finalizers. This runs
the actual COM state transition without requiring DRM or shader compilation.
"""
import argparse
from pathlib import Path
import subprocess


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
typedef int32_t HRESULT;
#define STDMETHODCALLTYPE
#define S_OK ((HRESULT)0)
#define E_FAIL ((HRESULT)0x80004005u)
#define E_OUTOFMEMORY ((HRESULT)0x8007000eu)
#define FAILED(x) ((x) < 0)
#define mesa_loge(...) ((void)0)
#define LIBAGX_WRITE_U32 0
enum { D12_LIST_RECORDING, D12_LIST_ERROR, D12_LIST_CLOSED };
struct d12_cmd_allocator { unsigned recording_lists; };
struct d12_device { uint64_t heap_desc_va; };
struct d12_cmd_list {
   int state;
   bool had_error, uses_heap;
   HRESULT record_error;
   struct d12_device *device;
   struct d12_cmd_allocator *allocator;
};
typedef struct d12_cmd_list ID3D12GraphicsCommandList6;
struct poly_heap { uint32_t bottom; };
struct libagx_write_u32_args { uint64_t address; uint32_t value; };
static bool fail_segment, fail_heap;
static unsigned segment_calls;
static struct d12_cmd_list *
d12_cmd_list_from_iface(ID3D12GraphicsCommandList6 *iface) { return iface; }
static int agx_1d(int count) { return count; }
static void d12_list_close_cdm(struct d12_cmd_list *list)
{
   segment_calls++;
   if (fail_segment) {
      list->state = D12_LIST_ERROR;
      list->record_error = E_OUTOFMEMORY;
   }
}
static void
d12_list_dispatch_precomp(struct d12_cmd_list *list, int kernel,
                         const void *args, size_t size, int grid)
{
   (void)kernel; (void)args; (void)size; (void)grid;
   if (fail_heap) {
      list->state = D12_LIST_ERROR;
      list->record_error = E_OUTOFMEMORY;
   }
}
'''

TEST = r'''
int main(void)
{
   struct d12_device device = {.heap_desc_va = 0x1000};
   struct d12_cmd_allocator allocator;
   for (unsigned mode = 0; mode < 4; mode++) {
      struct d12_cmd_list list = {
         .state = mode == 3 ? D12_LIST_ERROR : D12_LIST_RECORDING,
         .uses_heap = mode == 2,
         .record_error = mode == 3 ? E_FAIL : S_OK,
         .device = &device, .allocator = &allocator,
      };
      allocator.recording_lists = 1;
      segment_calls = 0;
      fail_segment = mode == 1;
      fail_heap = mode == 2;
      HRESULT expected = mode == 0 ? S_OK : mode == 3 ? E_FAIL : E_OUTOFMEMORY;
      assert(d12_list_Close(&list) == expected);
      assert(list.state == D12_LIST_CLOSED);
      assert(list.had_error == (mode != 0));
      assert(allocator.recording_lists == 0);
      assert(segment_calls == (unsigned)(mode < 2));
      assert(d12_list_Close(&list) == E_FAIL);
      assert(allocator.recording_lists == 0);
   }
   puts("command Close fault injection: PASS");
   return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='clang')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'd12_cmd.c')
    args = parser.parse_args()
    source = args.source.read_text()
    start = source.index('static HRESULT STDMETHODCALLTYPE\nd12_list_Close(')
    end = source.index('/* Swap in freshly allocated recording storage.', start)
    args.out.mkdir(parents=True, exist_ok=True)
    test = args.out / 'command_close.c'
    test.write_text(PRELUDE + source[start:end] + TEST)
    binary = args.out / 'command_close'
    subprocess.run([args.cc, '-std=c11', '-O1', '-g', '-Wall', '-Wextra',
                    '-Werror', '-Wpedantic', '-fsanitize=address,undefined',
                    str(test), '-o', str(binary)], check=True)
    subprocess.run([str(binary.resolve())], check=True)


if __name__ == '__main__':
    main()
