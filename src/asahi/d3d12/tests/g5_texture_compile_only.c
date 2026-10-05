/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Compile the mandatory G5 texture shader without creating a command queue. */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   if (argc != 2) return 2;
   FILE *f = fopen(argv[1], "rb");
   if (!f) return 2;
   if (fseek(f, 0, SEEK_END)) { fclose(f); return 2; }
   long size = ftell(f);
   if (size <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return 2; }
   void *code = malloc((size_t)size);
   if (!code || fread(code, 1, (size_t)size, f) != (size_t)size) {
      free(code); fclose(f); return 2;
   }
   fclose(f);
   ID3D12Device *dev = NULL;
   ID3DBlob *blob = NULL;
   ID3D12RootSignature *root = NULL;
   ID3D12PipelineState *pso = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   if (hr != S_OK || !dev) goto done;
   const D3D12_DESCRIPTOR_RANGE range = {
      .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV, .NumDescriptors = 1,
      .BaseShaderRegister = 0, .RegisterSpace = 0,
      .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
   };
   const D3D12_ROOT_PARAMETER params[2] = {
      {.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
       .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &range},
       .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL},
      {.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
       .Descriptor = {.ShaderRegister = 0},
       .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL},
   };
   const D3D12_ROOT_SIGNATURE_DESC rs = {.NumParameters = 2, .pParameters = params};
   hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, NULL);
   if (hr != S_OK || !blob) goto done;
   hr = ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(blob),
         ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&root);
   if (hr != S_OK || !root) goto done;
   const D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {
      .pRootSignature = root, .CS = {.pShaderBytecode = code, .BytecodeLength = size},
   };
   hr = ID3D12Device_CreateComputePipelineState(dev, &desc, &IID_ID3D12PipelineState,
                                               (void **)&pso);
done:
   ;
   const int succeeded = hr == S_OK && pso != NULL;
   printf("G5 TEXTURE COMPILE ONLY: HRESULT=0x%08x object=%d NO_QUEUE_NO_GPU_SUBMISSION_NOT_SUITE_PARITY\n",
          (unsigned)hr, pso != NULL);
   if (pso) ID3D12PipelineState_Release(pso);
   if (root) ID3D12RootSignature_Release(root);
   if (blob) ID3D10Blob_Release(blob);
   if (dev) ID3D12Device_Release(dev);
   free(code);
   return succeeded ? 0 : 1;
}
