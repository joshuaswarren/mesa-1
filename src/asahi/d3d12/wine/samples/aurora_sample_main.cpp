/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Entry point for Microsoft's D3D12HelloWorld samples (HelloTriangle,
 * HelloTexture, HelloBundles, HelloConstBuffers) built with llvm-mingw, used
 * to compare the d3d12agx Proton tool against the vkd3d-proton one. The sample
 * sources are unmodified; this file replaces Main.cpp/the message loop so the
 * run is bounded and measurable:
 *   --frames N      render N frames then exit (default 600)
 *   --selftest      read the back buffer after the last frame and check the
 *                   clear colour in the corner and that the triangle covered
 *                   a sane share of the frame; prints PASS/FAIL
 * Output: one JSON line with frame-time statistics on stdout.
 * Build with -DSAMPLE_CLASS=D3D12HelloTexture -DSAMPLE_HEADER="D3D12HelloTexture.h"
 * (defaults: D3D12HelloTriangle). All four samples share the member names
 * used below and the clear colour 0.0,0.2,0.4.
 */
#include "stdafx.h"
#ifndef SAMPLE_CLASS
#define SAMPLE_CLASS D3D12HelloTriangle
#endif
#ifndef SAMPLE_HEADER
#define SAMPLE_HEADER "D3D12HelloTriangle.h"
#endif
#define SAMPLE_STR_(x) #x
#define SAMPLE_STR(x) SAMPLE_STR_(x)
#define private public
#define protected public
#include SAMPLE_HEADER
#undef private
#undef protected

#include <d3dcompiler.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static HWND g_hwnd;
static bool g_quit;

static LRESULT CALLBACK
wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
   if (msg == WM_DESTROY) {
      g_quit = true;
      PostQuitMessage(0);
      return 0;
   }
   return DefWindowProcW(hwnd, msg, wp, lp);
}


struct Pixel { unsigned char b, g, r, a; };

#ifdef SAMPLE_PRECOMPILED
/* HelloTexture loads shaders_VSMain.cso / shaders_PSMain.cso that Visual
 * Studio compiles from shaders.hlsl at build time. Produce them next to the
 * exe at start-up with the same compiler the other samples use at run time
 * (d3dcompiler_47), so the sample source stays untouched. */
static bool
precompile_shaders(const std::wstring &assets)
{
   using Microsoft::WRL::ComPtr;
   const std::wstring hlsl = assets + L"shaders.hlsl";
   struct { const char *entry, *target; const wchar_t *out; } stages[] = {
      {"VSMain", "vs_5_0", L"shaders_VSMain.cso"},
      {"PSMain", "ps_5_0", L"shaders_PSMain.cso"},
   };
   for (const auto &st : stages) {
      ComPtr<ID3DBlob> code, err;
      HRESULT hr = D3DCompileFromFile(hlsl.c_str(), nullptr, nullptr, st.entry, st.target,
                                      D3DCOMPILE_ENABLE_STRICTNESS, 0, &code, &err);
      if (FAILED(hr)) {
         printf("{\"status\":\"INIT_FAIL\",\"error\":\"D3DCompileFromFile(%s) hr=0x%08lx %s\"}\n",
                st.entry, (unsigned long)hr,
                err ? (const char *)err->GetBufferPointer() : "");
         return false;
      }
      FILE *f = _wfopen((assets + st.out).c_str(), L"wb");
      if (!f)
         return false;
      fwrite(code->GetBufferPointer(), 1, code->GetBufferSize(), f);
      fclose(f);
   }
   return true;
}
#endif

static bool
read_back(SAMPLE_CLASS &s, std::vector<Pixel> &out, UINT &w, UINT &h)
{
   using Microsoft::WRL::ComPtr;
   UINT idx = s.m_swapChain->GetCurrentBackBufferIndex();
   /* The frame just presented is the previous buffer. */
   idx = (idx + 1) % 2;
   ComPtr<ID3D12Resource> src = s.m_renderTargets[idx];
   D3D12_RESOURCE_DESC desc = src->GetDesc();
   w = (UINT)desc.Width;
   h = desc.Height;
   UINT pitch = (w * 4 + 255) & ~255u;

   ComPtr<ID3D12Resource> rb;
   CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_READBACK);
   CD3DX12_RESOURCE_DESC bdesc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)pitch * h);
   if (FAILED(s.m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bdesc,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                  IID_PPV_ARGS(&rb))))
      return false;

   s.m_commandAllocator->Reset();
   s.m_commandList->Reset(s.m_commandAllocator.Get(), nullptr);
   auto to_copy = CD3DX12_RESOURCE_BARRIER::Transition(src.Get(), D3D12_RESOURCE_STATE_PRESENT,
                                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
   s.m_commandList->ResourceBarrier(1, &to_copy);
   D3D12_TEXTURE_COPY_LOCATION dst = {};
   dst.pResource = rb.Get();
   dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
   dst.PlacedFootprint.Footprint.Format = desc.Format;
   dst.PlacedFootprint.Footprint.Width = w;
   dst.PlacedFootprint.Footprint.Height = h;
   dst.PlacedFootprint.Footprint.Depth = 1;
   dst.PlacedFootprint.Footprint.RowPitch = pitch;
   D3D12_TEXTURE_COPY_LOCATION srcloc = {};
   srcloc.pResource = src.Get();
   srcloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
   s.m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &srcloc, nullptr);
   auto back = CD3DX12_RESOURCE_BARRIER::Transition(src.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                    D3D12_RESOURCE_STATE_PRESENT);
   s.m_commandList->ResourceBarrier(1, &back);
   s.m_commandList->Close();
   ID3D12CommandList *lists[] = {s.m_commandList.Get()};
   s.m_commandQueue->ExecuteCommandLists(1, lists);
   const UINT64 value = ++s.m_fenceValue;
   s.m_commandQueue->Signal(s.m_fence.Get(), value);
   if (s.m_fence->GetCompletedValue() < value) {
      s.m_fence->SetEventOnCompletion(value, s.m_fenceEvent);
      WaitForSingleObject(s.m_fenceEvent, 5000);
   }
   void *map = nullptr;
   D3D12_RANGE range = {0, (SIZE_T)pitch * h};
   if (FAILED(rb->Map(0, &range, &map)))
      return false;
   out.resize((size_t)w * h);
   for (UINT y = 0; y < h; y++) {
      const unsigned char *row = (const unsigned char *)map + (size_t)y * pitch;
      for (UINT x = 0; x < w; x++) {
         /* back buffer is R8G8B8A8_UNORM in this sample */
         out[(size_t)y * w + x] = {row[x * 4 + 2], row[x * 4 + 1], row[x * 4 + 0], row[x * 4 + 3]};
      }
   }
   rb->Unmap(0, nullptr);
   return true;
}

int
wmain(int argc, wchar_t **argv)
{
   int frames = 600;
   bool selftest = false;
   for (int i = 1; i < argc; i++) {
      if (!wcscmp(argv[i], L"--frames") && i + 1 < argc)
         frames = _wtoi(argv[++i]);
      else if (!wcscmp(argv[i], L"--selftest"))
         selftest = true;
   }

   SAMPLE_CLASS sample(1280, 720, L"" SAMPLE_STR(SAMPLE_CLASS) L" (aurora)");
   sample.ParseCommandLineArgs(argv, argc);

   HINSTANCE hinst = GetModuleHandleW(nullptr);
   WNDCLASSEXW wc = {};
   wc.cbSize = sizeof(wc);
   wc.style = CS_HREDRAW | CS_VREDRAW;
   wc.lpfnWndProc = wnd_proc;
   wc.hInstance = hinst;
   wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
   wc.lpszClassName = L"AuroraDXSampleClass";
   RegisterClassExW(&wc);
   RECT rc = {0, 0, (LONG)sample.GetWidth(), (LONG)sample.GetHeight()};
   AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
   g_hwnd = CreateWindowW(wc.lpszClassName, sample.GetTitle(), WS_OVERLAPPEDWINDOW,
                          CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left,
                          rc.bottom - rc.top, nullptr, nullptr, hinst, nullptr);
   Win32Application::m_hwnd = g_hwnd;

   LARGE_INTEGER freq, t0, t1;
   QueryPerformanceFrequency(&freq);
   QueryPerformanceCounter(&t0);
   try {
#ifdef SAMPLE_PRECOMPILED
      if (!precompile_shaders(sample.m_assetsPath))
         return 2;
#endif
      sample.OnInit();
   } catch (const std::exception &e) {
      printf("{\"status\":\"INIT_FAIL\",\"error\":\"%s\"}\n", e.what());
      fflush(stdout);
      return 2;
   }
   QueryPerformanceCounter(&t1);
   double init_ms = (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart;
   ShowWindow(g_hwnd, SW_SHOW);

   std::vector<double> ms;
   ms.reserve(frames);
   MSG msg = {};
   int rendered = 0;
   LARGE_INTEGER prev;
   QueryPerformanceCounter(&prev);
   while (rendered < frames && !g_quit) {
      while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
         TranslateMessage(&msg);
         DispatchMessageW(&msg);
      }
      sample.OnUpdate();
      sample.OnRender();
      LARGE_INTEGER now;
      QueryPerformanceCounter(&now);
      ms.push_back((now.QuadPart - prev.QuadPart) * 1000.0 / freq.QuadPart);
      prev = now;
      rendered++;
   }

   std::string verdict = "SKIPPED";
   int center[3] = {-1, -1, -1}, corner[3] = {-1, -1, -1};
   double covered = -1;
   if (selftest) {
      std::vector<Pixel> px;
      UINT w = 0, h = 0;
      if (read_back(sample, px, w, h) && w && h) {
         Pixel c = px[(size_t)(h / 2) * w + w / 2];
         Pixel k = px[(size_t)10 * w + 10];
         center[0] = c.r; center[1] = c.g; center[2] = c.b;
         corner[0] = k.r; corner[1] = k.g; corner[2] = k.b;
         /* clear colour 0.0,0.2,0.4 -> (0,51,102) in every HelloWorld sample. */
         auto is_clear = [](const Pixel &p) {
            return p.r < 8 && p.g > 40 && p.g < 62 && p.b > 90 && p.b < 114;
         };
         size_t drawn = 0;
         for (const Pixel &p : px)
            drawn += !is_clear(p);
         covered = (double)drawn / px.size();
         /* The triangle spans 0.25*aspect..0.25 of the clip space in each
          * sample: ~5.6% of the frame at 16:9 (HelloConstBuffers slides it
          * sideways, HelloTexture fills it with a checkerboard), so accept a
          * window around that and reject an all-clear or garbage frame. */
         bool corner_ok = is_clear(k);
         bool cover_ok = covered > 0.01 && covered < 0.30;
         verdict = (corner_ok && cover_ok) ? "PASS" : "FAIL";
      } else {
         verdict = "READBACK_FAIL";
      }
   }

   std::vector<double> sorted(ms.begin() + std::min<size_t>(ms.size(), 10), ms.end());
   std::sort(sorted.begin(), sorted.end());
   double sum = 0;
   for (double v : sorted) sum += v;
   double avg = sorted.empty() ? 0 : sum / sorted.size();
   double p50 = sorted.empty() ? 0 : sorted[sorted.size() / 2];
   double p95 = sorted.empty() ? 0 : sorted[(size_t)(sorted.size() * 0.95)];
   double p99 = sorted.empty() ? 0 : sorted[(size_t)(sorted.size() * 0.99)];
   printf("{\"status\":\"OK\",\"frames\":%d,\"init_ms\":%.2f,\"avg_ms\":%.3f,\"p50_ms\":%.3f,"
          "\"p95_ms\":%.3f,\"p99_ms\":%.3f,\"fps\":%.1f,\"selftest\":\"%s\","
          "\"sample\":\"%s\",\"covered\":%.4f,\"center\":[%d,%d,%d],\"corner\":[%d,%d,%d]}\n",
          rendered, init_ms, avg, p50, p95, p99, avg > 0 ? 1000.0 / avg : 0.0,
          verdict.c_str(), SAMPLE_STR(SAMPLE_CLASS), covered, center[0], center[1], center[2],
          corner[0], corner[1], corner[2]);
   fflush(stdout);
   sample.OnDestroy();
   DestroyWindow(g_hwnd);
   return verdict == "FAIL" || verdict == "READBACK_FAIL" ? 1 : 0;
}
