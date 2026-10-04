/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Native G17P PSO qualification only. Never creates/submits a command queue. */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
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
   if (argc != 3 || !shim_loaded()) {
      fprintf(stderr, "usage under loaded G17P DRM shim: %s tri.dxil solid.dxil\n", argv[0]);
      return 2;
   }
   size_t vs_size = 0, fs_size = 0;
   void *vs = read_shader(argv[1], &vs_size);
   void *fs = read_shader(argv[2], &fs_size);
   ID3D12Device *device = NULL;
   ID3DBlob *blob = NULL;
   ID3D12RootSignature *root = NULL;
   ID3D12PipelineState *pso = NULL;
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
done:
   const bool passed = hr == S_OK && pso;
   printf("G17P PSO COMPILE ONLY: %s HRESULT=0x%08x NO_GPU_SUBMISSION\n",
          passed ? "PASS" : "FAIL", (unsigned)hr);
   if (pso) ID3D12PipelineState_Release(pso);
   if (root) ID3D12RootSignature_Release(root);
   if (blob) ID3D10Blob_Release(blob);
   if (device) ID3D12Device_Release(device);
   free(vs); free(fs);
   return passed ? 0 : 1;
}
