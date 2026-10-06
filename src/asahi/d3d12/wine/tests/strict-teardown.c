/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Unix render fds must never reach the PE CRT, including probe/error cleanup.
 * Run as x64 under the Wine ARM64EC tool; no GPU commands are submitted.
 */
#define COBJMACROS
#include <windows.h>
#include <directx/d3d12.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void __cdecl
invalid_parameter(const wchar_t *expression, const wchar_t *function,
                  const wchar_t *file, unsigned line, uintptr_t reserved)
{
   (void)expression; (void)function; (void)file; (void)line; (void)reserved;
   fputs("STRICT_TEARDOWN_INVALID_PARAMETER\n", stderr);
   abort();
}

static int failures;
#define CHECK(test) do { \
   if (!(test)) { printf("FAIL line=%d: %s\n", __LINE__, #test); failures++; } \
} while (0)

static void
create_release(void)
{
   ID3D12Device *device = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                &IID_ID3D12Device, (void **)&device);
   printf("CREATE hr=%08lx device=%p\n", (unsigned long)hr, device);
   CHECK(hr == S_OK && device);
   if (device) {
      CHECK(ID3D12Device_GetNodeCount(device) == 1);
      CHECK(ID3D12Device_Release(device) == 0);
   }
}

int
main(void)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   _set_invalid_parameter_handler(invalid_parameter);
   SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
   /* Warm up lazy DLL/CRT/compiler state before the live Unix fd snapshot. */
   create_release();
   puts("STRICT_TEARDOWN_FD_BASELINE");
   Sleep(3000);
   static const GUID unsupported = {0x13041304, 0, 0, {0}};
   for (unsigned i = 0; i < 8; i++) {
      create_release();
      /* UE's adapter probe creates a device without retaining an interface. */
      CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                             &IID_ID3D12Device, NULL) == S_FALSE);
      void *out = (void *)(uintptr_t)1;
      CHECK(D3D12CreateDevice(NULL, (D3D_FEATURE_LEVEL)0xffff,
                             &IID_ID3D12Device, &out) == E_INVALIDARG);
      CHECK(out == NULL);
      out = (void *)(uintptr_t)1;
      CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                             &unsupported, &out) == E_NOINTERFACE);
      CHECK(out == NULL);
   }
   puts("STRICT_TEARDOWN_FD_AFTER");
   Sleep(3000);
   printf("STRICT_TEARDOWN_RESULT failures=%d\n", failures);
   return failures ? 1 : 0;
}
