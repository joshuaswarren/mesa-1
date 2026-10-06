/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * An application SDK fixture which must never be loaded by d3d12agx.
 */
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
   if (reason == DLL_PROCESS_ATTACH)
      ExitProcess(99);
   return TRUE;
}

__declspec(dllexport) HRESULT WINAPI
D3D12GetInterface(REFCLSID clsid, REFIID iid, void **out)
{
   return E_FAIL;
}
