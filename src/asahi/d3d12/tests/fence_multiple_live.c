/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Once an armed fence reaches its target, a later CPU rewind must not erase
 * that completion from ALL/ANY event registrations.
 */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <stdio.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { \
   fprintf(stderr, "FAIL fence_multiple_live:%d: %s\n", __LINE__, #expr); \
   goto cleanup; } } while (0)
#define CALL(expr) CHECK(SUCCEEDED(expr))

int
main(void)
{
   int failed = 1, event = -1;
   ID3D12Device *device = NULL;
   ID3D12Device1 *device1 = NULL;
   ID3D12Fence *fences[2] = {0};
   const UINT64 values[2] = {1, 1};
   CALL(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&device));
   CALL(ID3D12Device_QueryInterface(device, &IID_ID3D12Device1, (void **)&device1));
   for (unsigned i = 0; i < 2; i++)
      CALL(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fences[i]));
   event = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
   CHECK(event > 0);
   struct pollfd pfd = {.fd = event, .events = POLLIN};
   HANDLE handle = (HANDLE)(intptr_t)event;
   uint64_t token;

   CALL(ID3D12Device1_SetEventOnMultipleFenceCompletion(device1, fences, values, 2,
      D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, handle));
   CALL(ID3D12Fence_Signal(fences[0], 1));
   CALL(ID3D12Fence_Signal(fences[0], 0));
   CHECK(poll(&pfd, 1, 0) == 0);
   CALL(ID3D12Fence_Signal(fences[1], 1));
   CHECK(poll(&pfd, 1, 1000) == 1 && (pfd.revents & POLLIN));
   CHECK(read(event, &token, sizeof(token)) == sizeof(token));
   puts("ALL preserves completion across rewind: PASS");

   CALL(ID3D12Fence_Signal(fences[1], 0));
   CALL(ID3D12Device1_SetEventOnMultipleFenceCompletion(device1, fences, values, 2,
      D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, handle));
   CALL(ID3D12Fence_Signal(fences[0], 1));
   CALL(ID3D12Fence_Signal(fences[0], 0));
   CHECK(poll(&pfd, 1, 1000) == 1 && (pfd.revents & POLLIN));
   CHECK(read(event, &token, sizeof(token)) == sizeof(token));
   puts("ANY preserves completion across rewind: PASS");
   failed = 0;
cleanup:
   /* Finish a registration even on a failed assertion before closing its
    * eventfd, so a pending asynchronous waiter cannot signal a reused fd. */
   for (unsigned i = 0; i < 2; i++)
      if (fences[i]) ID3D12Fence_Signal(fences[i], 1);
   if (failed && event >= 0) {
      struct pollfd pending = {.fd = event, .events = POLLIN};
      (void)poll(&pending, 1, 1000);
   }
   for (unsigned i = 0; i < 2; i++)
      if (fences[i]) ID3D12Fence_Release(fences[i]);
   if (event >= 0) close(event);
   if (device1) ID3D12Device1_Release(device1);
   if (device) ID3D12Device_Release(device);
   printf("MULTIPLE FENCE EVENTS: %s\n", failed ? "FAILURES" : "ALL PASS");
   return failed;
}
