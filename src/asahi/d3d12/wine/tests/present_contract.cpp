/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Exercise DXVK's outer DXGI contract through the public D3D12 interface.
 */
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <stdexcept>
#include <initializer_list>

static void check(HRESULT hr, const char *what)
{
   if (FAILED(hr)) { std::printf("FAIL %s: 0x%08lx\n",what,(unsigned long)hr); throw std::runtime_error(what); }
}
static ID3D12Device *device;
static ID3D12CommandQueue *queue;
static IDXGISwapChain3 *swap;
static ID3D12DescriptorHeap *rtv;
static ID3D12Fence *fence;
static UINT64 serial;
static HANDLE event;
static void drain()
{
   check(queue->Signal(fence,++serial),"Signal");
   check(fence->SetEventOnCompletion(serial,event),"SetEventOnCompletion");
   if (WaitForSingleObject(event,5000) != WAIT_OBJECT_0) throw std::runtime_error("fence timeout");
}
static void frame(UINT interval)
{
   ID3D12Resource *buffer = nullptr;
   check(swap->GetBuffer(swap->GetCurrentBackBufferIndex(),__uuidof(ID3D12Resource),(void **)&buffer),"GetBuffer");
   ID3D12CommandAllocator *alloc = nullptr;
   ID3D12GraphicsCommandList *list = nullptr;
   check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,__uuidof(ID3D12CommandAllocator),(void **)&alloc),"allocator");
   check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,nullptr,__uuidof(ID3D12GraphicsCommandList),(void **)&list),"list");
   auto handle = rtv->GetCPUDescriptorHandleForHeapStart();
   device->CreateRenderTargetView(buffer,nullptr,handle);
   D3D12_RESOURCE_BARRIER b = {};
   b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
   b.Transition = {buffer,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_RENDER_TARGET};
   list->ResourceBarrier(1,&b);
   float color[] = {0.125f,0.25f,0.5f,0.0f};
   list->ClearRenderTargetView(handle,color,0,nullptr);
   b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
   b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
   list->ResourceBarrier(1,&b);
   check(list->Close(),"Close");
   ID3D12CommandList *lists[] = {list}; queue->ExecuteCommandLists(1,lists);
   check(swap->Present(interval,0),"Present"); drain();
   list->Release(); alloc->Release(); buffer->Release();
}
int main()
{
   try {
      WNDCLASSW wc = {}; wc.lpfnWndProc = DefWindowProcW; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"AuroraPresentContract";
      if (!RegisterClassW(&wc)) return 1;
      HWND window = CreateWindowW(wc.lpszClassName,L"Aurora present contract",WS_OVERLAPPEDWINDOW,
                                 CW_USEDEFAULT,CW_USEDEFAULT,640,480,nullptr,nullptr,wc.hInstance,nullptr);
      if (!window) return 1;
      ShowWindow(window,SW_SHOW); SetForegroundWindow(window);
      DWORD until = GetTickCount()+500;
      while ((LONG)(GetTickCount()-until)<0) { MSG msg; while (PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); } Sleep(5); }
      check(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,__uuidof(ID3D12Device),(void **)&device),"device");
      D3D12_COMMAND_QUEUE_DESC q = {}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      check(device->CreateCommandQueue(&q,__uuidof(ID3D12CommandQueue),(void **)&queue),"queue");
      IDXGIFactory2 *factory = nullptr;
      check(CreateDXGIFactory1(__uuidof(IDXGIFactory2),(void **)&factory),"factory");
      DXGI_SWAP_CHAIN_DESC1 desc = {}; desc.Width = 128; desc.Height = 64; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount = 2;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
      IDXGISwapChain1 *s1 = nullptr;
      check(factory->CreateSwapChainForHwnd(queue,window,&desc,nullptr,nullptr,&s1),"swapchain");
      check(s1->QueryInterface(__uuidof(IDXGISwapChain3),(void **)&swap),"swapchain3"); s1->Release();
      D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; hd.NumDescriptors = 1;
      check(device->CreateDescriptorHeap(&hd,__uuidof(ID3D12DescriptorHeap),(void **)&rtv),"RTV heap");
      check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),(void **)&fence),"fence"); event = CreateEventW(nullptr,FALSE,FALSE,nullptr);
      LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
      for (UINT interval : {0u,1u,2u,4u}) {
         frame(interval);
         LARGE_INTEGER start,end; QueryPerformanceCounter(&start);
         for (unsigned i = 0; i < 8; i++) frame(interval);
         QueryPerformanceCounter(&end);
         double ms = (end.QuadPart-start.QuadPart)*1000.0/freq.QuadPart/8;
         std::printf("sync_interval=%u ms_per_frame=%.6f\n",interval,ms);
         if (interval && (ms < interval*12.0 || ms > interval*25.0)) throw std::runtime_error("FIFO pacing");
      }
      check(swap->Present(0,DXGI_PRESENT_TEST),"Present TEST");
      check(swap->ResizeBuffers(2,131,73,DXGI_FORMAT_R8G8B8A8_UNORM,0),"ResizeBuffers");
      frame(0);
      ID3D12Resource *res = nullptr; check(swap->GetBuffer(0,__uuidof(ID3D12Resource),(void **)&res),"resized GetBuffer");
      auto actual = res->GetDesc(); res->Release();
      if (actual.Width != 131 || actual.Height != 73) throw std::runtime_error("resized dimensions");
      HRESULT invalid = swap->ResizeBuffers(17,127,71,DXGI_FORMAT_R8G8B8A8_UNORM,0);
      if (SUCCEEDED(invalid)) throw std::runtime_error("invalid resize accepted");
      frame(0);
      HRESULT full = swap->SetFullscreenState(TRUE,nullptr);
      std::printf("fullscreen_hr=0x%08lx\n",(unsigned long)full);
      if (SUCCEEDED(full)) { frame(1); check(swap->SetFullscreenState(FALSE,nullptr),"leave fullscreen"); frame(0); }
      else if (full != DXGI_ERROR_NOT_CURRENTLY_AVAILABLE && full != DXGI_ERROR_UNSUPPORTED) check(full,"fullscreen");
      drain(); CloseHandle(event); fence->Release(); rtv->Release(); swap->Release(); factory->Release(); queue->Release(); device->Release(); DestroyWindow(window);
      std::puts("PRESENT_CONTRACT_PASS"); return 0;
   } catch (const std::exception &e) { std::printf("FAIL %s\n",e.what()); return 1; }
}
