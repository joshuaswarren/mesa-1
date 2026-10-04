/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Native G17P PSO qualification and optional shim-only recording/submit.
 * Loaded G17P DRM shim is required in every mode. No rendered pixel verdict. */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include "d12_test_abi.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_shader(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f) return NULL;
   if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
   long length = ftell(f);
   if (length <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
   void *bytes = malloc((size_t)length);
   if (!bytes || fread(bytes, 1, (size_t)length, f) != (size_t)length) {
      free(bytes); fclose(f); return NULL;
   }
   fclose(f); *size = (size_t)length; return bytes;
}

static bool shim_loaded(void)
{
   FILE *maps = fopen("/proc/self/maps", "r");
   if (!maps) return false;
   char line[4096]; bool found = false;
   while (fgets(line, sizeof(line), maps))
      if (strstr(line, "libasahi_noop_drm_shim.so")) found = true;
   fclose(maps);
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   return found && chip && !strcmp(chip, "g17p");
}

int main(int argc, char **argv)
{
   unsigned mode = 0, base = 1;
   if (argc == 4 && !strcmp(argv[1], "--record-only")) { mode = 1; base = 2; }
   if (argc == 4 && !strcmp(argv[1], "--shim-submit-only")) { mode = 2; base = 2; }
   if (argc != (int)base + 2 || !shim_loaded()) {
      fprintf(stderr, "usage under loaded G17P DRM shim: %s [--record-only|--shim-submit-only] tri.dxil solid.dxil\n", argv[0]);
      return 2;
   }
   size_t vs_size = 0, fs_size = 0;
   void *vs = read_shader(argv[base], &vs_size);
   void *fs = read_shader(argv[base + 1], &fs_size);
   ID3D12Device *device = NULL;
   ID3DBlob *blob = NULL;
   ID3D12RootSignature *root = NULL;
   ID3D12PipelineState *pso = NULL;
   ID3D12Resource *target = NULL;
   ID3D12DescriptorHeap *rtv_heap = NULL;
   ID3D12CommandAllocator *allocator = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12Fence *fence = NULL;
   const char *phase = "PSO COMPILE ONLY";
   bool phase_complete = false;
   HRESULT hr = E_FAIL;
   if (!vs || !fs) goto done;
   hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                         &IID_ID3D12Device, (void **)&device);
   if (hr != S_OK || !device) goto done;
   const D3D12_ROOT_SIGNATURE_DESC rs = {0};
   hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, NULL);
   if (hr != S_OK || !blob) goto done;
   hr = ID3D12Device_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(blob),
          ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&root);
   if (hr != S_OK || !root) goto done;
   D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {
      .pRootSignature = root,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = fs, .BytecodeLength = fs_size},
      .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID,
                         .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE},
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
      .NumRenderTargets = 1, .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1}, .SampleMask = ~0u,
   };
   desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   hr = ID3D12Device_CreateGraphicsPipelineState(device, &desc,
                                                &IID_ID3D12PipelineState, (void **)&pso);
   if (hr != S_OK || !pso) goto done;
   if (!mode) { phase_complete = true; goto done; }
   phase = mode == 1 ? "RECORD ONLY" : "SHIM SUBMIT ONLY";
   if (mode == 2) {
      const char *validate = getenv("ASAHI_SHIM_VALIDATE_G17P_STREAM");
      const char *expected = getenv("ASAHI_SHIM_G17P_EXPECT_VALID_STREAM");
      if (!validate || strcmp(validate, "1") || !expected || strcmp(expected, "1")) {
         hr = E_FAIL; goto done;
      }
   }
   const D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   const D3D12_RESOURCE_DESC resource = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = 256, .Height = 256, .DepthOrArraySize = 1, .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1},
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
   };
   hr = ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE,
          &resource, D3D12_RESOURCE_STATE_RENDER_TARGET, NULL,
          &IID_ID3D12Resource, (void **)&target);
   if (hr != S_OK || !target) goto done;
   const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1,
   };
   hr = ID3D12Device_CreateDescriptorHeap(device, &rtv_desc,
          &IID_ID3D12DescriptorHeap, (void **)&rtv_heap);
   if (hr != S_OK || !rtv_heap) goto done;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   ID3D12Device_CreateRenderTargetView(device, target, NULL, rtv);
   hr = ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT,
          &IID_ID3D12CommandAllocator, (void **)&allocator);
   if (hr != S_OK || !allocator) goto done;
   hr = ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
          allocator, pso, &IID_ID3D12GraphicsCommandList, (void **)&list);
   if (hr != S_OK || !list) goto done;
   const FLOAT clear[4] = {1, 0, 0, 1};
   const D3D12_VIEWPORT viewport = {.Width = 256, .Height = 256, .MaxDepth = 1};
   const D3D12_RECT scissor = {.right = 256, .bottom = 256};
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &viewport);
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &scissor);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
   hr = ID3D12GraphicsCommandList_Close(list);
   if (hr != S_OK) goto done;
   if (mode == 1) { phase_complete = true; goto done; }
   const D3D12_COMMAND_QUEUE_DESC qdesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(device, &qdesc, &IID_ID3D12CommandQueue,
                                       (void **)&queue);
   if (hr != S_OK || !queue) goto done;
   hr = ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE,
                                &IID_ID3D12Fence, (void **)&fence);
   if (hr != S_OK || !fence) goto done;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12Device_GetDeviceRemovedReason(device);
   if (hr != S_OK) goto done;
   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   if (hr != S_OK) goto done;
   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   if (hr != S_OK) goto done;
   UINT64 completed = ID3D12Fence_GetCompletedValue(fence);
   if (completed == UINT64_MAX || completed < 1) {hr = E_FAIL; goto done;}
   hr = ID3D12Device_GetDeviceRemovedReason(device);

done:
   const bool passed = phase_complete && hr == S_OK && pso;
   printf("G17P %s: %s HRESULT=0x%08x %s\n", phase,
          passed ? "PASS" : "FAIL", (unsigned)hr,
          mode == 2 ? "SHIM_METADATA_ONLY_NO_PIXEL_OR_HARDWARE_VERDICT" : "NO_GPU_SUBMISSION");
   if (fence) ID3D12Fence_Release(fence);
   if (queue) ID3D12CommandQueue_Release(queue);
   if (list) ID3D12GraphicsCommandList_Release(list);
   if (allocator) ID3D12CommandAllocator_Release(allocator);
   if (rtv_heap) ID3D12DescriptorHeap_Release(rtv_heap);
   if (target) ID3D12Resource_Release(target);
   if (pso) ID3D12PipelineState_Release(pso);
   if (root) ID3D12RootSignature_Release(root);
   if (blob) ID3D10Blob_Release(blob);
   if (device) ID3D12Device_Release(device);
   free(vs); free(fs);
   return passed ? 0 : 1;
}
