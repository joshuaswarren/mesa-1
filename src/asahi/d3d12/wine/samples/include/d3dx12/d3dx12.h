/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Minimal d3dx12 helpers for building Microsoft's D3D12HelloWorld samples
 * (HelloTriangle, HelloTexture, HelloBundles, HelloConstBuffers) with
 * mingw-w64's d3d12.h (the full d3dx12.h needs newer SDK types). Same names
 * and semantics as the public helpers the samples use; nothing else.
 */
#pragma once
#include <d3d12.h>
#include <cstring>
#include <vector>

struct CD3DX12_RECT : public D3D12_RECT {
   CD3DX12_RECT() = default;
   CD3DX12_RECT(LONG l, LONG t, LONG r, LONG b) { left = l; top = t; right = r; bottom = b; }
};

struct CD3DX12_VIEWPORT : public D3D12_VIEWPORT {
   CD3DX12_VIEWPORT() = default;
   CD3DX12_VIEWPORT(FLOAT x, FLOAT y, FLOAT w, FLOAT h, FLOAT minDepth = D3D12_MIN_DEPTH,
                    FLOAT maxDepth = D3D12_MAX_DEPTH) {
      TopLeftX = x; TopLeftY = y; Width = w; Height = h; MinDepth = minDepth; MaxDepth = maxDepth;
   }
};

struct CD3DX12_HEAP_PROPERTIES : public D3D12_HEAP_PROPERTIES {
   explicit CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE type, UINT creationNodeMask = 1,
                                    UINT nodeMask = 1) {
      Type = type; CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
      MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
      CreationNodeMask = creationNodeMask; VisibleNodeMask = nodeMask;
   }
};

struct CD3DX12_RESOURCE_DESC : public D3D12_RESOURCE_DESC {
   CD3DX12_RESOURCE_DESC() = default;
   static CD3DX12_RESOURCE_DESC Buffer(UINT64 width,
                                       D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
                                       UINT64 alignment = 0) {
      CD3DX12_RESOURCE_DESC d;
      d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Alignment = alignment; d.Width = width;
      d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = DXGI_FORMAT_UNKNOWN;
      d.SampleDesc.Count = 1; d.SampleDesc.Quality = 0;
      d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = flags;
      return d;
   }
   static CD3DX12_RESOURCE_DESC Tex2D(DXGI_FORMAT format, UINT64 width, UINT height,
                                      UINT16 arraySize = 1, UINT16 mipLevels = 0,
                                      UINT sampleCount = 1, UINT sampleQuality = 0,
                                      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                      UINT64 alignment = 0) {
      CD3DX12_RESOURCE_DESC d;
      d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Alignment = alignment; d.Width = width;
      d.Height = height; d.DepthOrArraySize = arraySize; d.MipLevels = mipLevels; d.Format = format;
      d.SampleDesc.Count = sampleCount; d.SampleDesc.Quality = sampleQuality;
      d.Layout = layout; d.Flags = flags;
      return d;
   }
};

struct CD3DX12_RANGE : public D3D12_RANGE {
   CD3DX12_RANGE(SIZE_T begin, SIZE_T end) { Begin = begin; End = end; }
};

struct CD3DX12_ROOT_SIGNATURE_DESC : public D3D12_ROOT_SIGNATURE_DESC {
   CD3DX12_ROOT_SIGNATURE_DESC() = default;
   void Init(UINT numParameters, const D3D12_ROOT_PARAMETER *params, UINT numStaticSamplers = 0,
             const D3D12_STATIC_SAMPLER_DESC *samplers = nullptr,
             D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) {
      NumParameters = numParameters; pParameters = params; NumStaticSamplers = numStaticSamplers;
      pStaticSamplers = samplers; Flags = flags;
   }
};

struct CD3DX12_RASTERIZER_DESC : public D3D12_RASTERIZER_DESC {
   explicit CD3DX12_RASTERIZER_DESC(int) {
      FillMode = D3D12_FILL_MODE_SOLID; CullMode = D3D12_CULL_MODE_BACK;
      FrontCounterClockwise = FALSE; DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
      DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
      SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
      DepthClipEnable = TRUE; MultisampleEnable = FALSE; AntialiasedLineEnable = FALSE;
      ForcedSampleCount = 0; ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
   }
};

struct CD3DX12_BLEND_DESC : public D3D12_BLEND_DESC {
   explicit CD3DX12_BLEND_DESC(int) {
      AlphaToCoverageEnable = FALSE; IndependentBlendEnable = FALSE;
      const D3D12_RENDER_TARGET_BLEND_DESC rt = {
         FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
         D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP,
         D3D12_COLOR_WRITE_ENABLE_ALL,
      };
      for (UINT i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
         RenderTarget[i] = rt;
   }
};

struct CD3DX12_CPU_DESCRIPTOR_HANDLE : public D3D12_CPU_DESCRIPTOR_HANDLE {
   CD3DX12_CPU_DESCRIPTOR_HANDLE() = default;
   CD3DX12_CPU_DESCRIPTOR_HANDLE(const D3D12_CPU_DESCRIPTOR_HANDLE &other) { ptr = other.ptr; }
   CD3DX12_CPU_DESCRIPTOR_HANDLE(D3D12_CPU_DESCRIPTOR_HANDLE other, INT offsetInDescriptors,
                                 UINT descriptorIncrementSize) {
      ptr = other.ptr + SIZE_T(INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
   }
   void Offset(INT offsetInDescriptors, UINT descriptorIncrementSize) {
      ptr += SIZE_T(INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
   }
};

struct CD3DX12_RESOURCE_BARRIER : public D3D12_RESOURCE_BARRIER {
   static CD3DX12_RESOURCE_BARRIER Transition(
      ID3D12Resource *res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
      UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
      D3D12_RESOURCE_BARRIER raw;
      std::memset(&raw, 0, sizeof(raw));
      raw.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; raw.Flags = flags;
      raw.Transition.pResource = res; raw.Transition.StateBefore = before;
      raw.Transition.StateAfter = after; raw.Transition.Subresource = subresource;
      CD3DX12_RESOURCE_BARRIER b;
      static_cast<D3D12_RESOURCE_BARRIER &>(b) = raw;
      return b;
   }
};

struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE {
   CD3DX12_SHADER_BYTECODE() = default;
   explicit CD3DX12_SHADER_BYTECODE(ID3DBlob *blob) {
      pShaderBytecode = blob->GetBufferPointer(); BytecodeLength = blob->GetBufferSize();
   }
   CD3DX12_SHADER_BYTECODE(const void *p, SIZE_T n) { pShaderBytecode = p; BytecodeLength = n; }
};

#define D3D12_DEFAULT 0

/* ---- root signature 1.1 (HelloTexture, HelloConstBuffers) ---------------- */

struct CD3DX12_DESCRIPTOR_RANGE1 : public D3D12_DESCRIPTOR_RANGE1 {
   CD3DX12_DESCRIPTOR_RANGE1() = default;
   void Init(D3D12_DESCRIPTOR_RANGE_TYPE type, UINT numDescriptors, UINT baseShaderRegister,
             UINT registerSpace = 0,
             D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
             UINT offsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) {
      RangeType = type; NumDescriptors = numDescriptors; BaseShaderRegister = baseShaderRegister;
      RegisterSpace = registerSpace; Flags = flags;
      OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
   }
};

struct CD3DX12_ROOT_PARAMETER1 : public D3D12_ROOT_PARAMETER1 {
   CD3DX12_ROOT_PARAMETER1() = default;
   void InitAsDescriptorTable(UINT numDescriptorRanges, const D3D12_DESCRIPTOR_RANGE1 *ranges,
                              D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) {
      ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      ShaderVisibility = visibility;
      DescriptorTable.NumDescriptorRanges = numDescriptorRanges;
      DescriptorTable.pDescriptorRanges = ranges;
   }
   void InitAsConstantBufferView(UINT shaderRegister, UINT registerSpace = 0,
                                 D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                                 D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) {
      ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      ShaderVisibility = visibility;
      Descriptor.ShaderRegister = shaderRegister; Descriptor.RegisterSpace = registerSpace;
      Descriptor.Flags = flags;
   }
};

struct CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC : public D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
   CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC() = default;
   void Init_1_1(UINT numParameters, const D3D12_ROOT_PARAMETER1 *params, UINT numStaticSamplers = 0,
                 const D3D12_STATIC_SAMPLER_DESC *samplers = nullptr,
                 D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) {
      Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
      Desc_1_1.NumParameters = numParameters; Desc_1_1.pParameters = params;
      Desc_1_1.NumStaticSamplers = numStaticSamplers; Desc_1_1.pStaticSamplers = samplers;
      Desc_1_1.Flags = flags;
   }
};

/* Serialize at the highest version the device supports; a 1.1 description is
 * downgraded to 1.0 (descriptor tables and root descriptors only, which is all
 * the samples use) when the device reports 1.0. */
inline HRESULT
D3DX12SerializeVersionedRootSignature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc,
                                      D3D_ROOT_SIGNATURE_VERSION maxVersion, ID3DBlob **blob,
                                      ID3DBlob **error)
{
   if (error)
      *error = nullptr;
   if (maxVersion >= D3D_ROOT_SIGNATURE_VERSION_1_1 || desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_0)
      return D3D12SerializeVersionedRootSignature(desc, blob, error);

   const D3D12_ROOT_SIGNATURE_DESC1 &d1 = desc->Desc_1_1;
   std::vector<D3D12_ROOT_PARAMETER> params(d1.NumParameters);
   std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges(d1.NumParameters);
   for (UINT i = 0; i < d1.NumParameters; i++) {
      const D3D12_ROOT_PARAMETER1 &p1 = d1.pParameters[i];
      params[i].ParameterType = p1.ParameterType;
      params[i].ShaderVisibility = p1.ShaderVisibility;
      switch (p1.ParameterType) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
         ranges[i].resize(p1.DescriptorTable.NumDescriptorRanges);
         for (UINT r = 0; r < p1.DescriptorTable.NumDescriptorRanges; r++) {
            const D3D12_DESCRIPTOR_RANGE1 &s = p1.DescriptorTable.pDescriptorRanges[r];
            ranges[i][r].RangeType = s.RangeType;
            ranges[i][r].NumDescriptors = s.NumDescriptors;
            ranges[i][r].BaseShaderRegister = s.BaseShaderRegister;
            ranges[i][r].RegisterSpace = s.RegisterSpace;
            ranges[i][r].OffsetInDescriptorsFromTableStart = s.OffsetInDescriptorsFromTableStart;
         }
         params[i].DescriptorTable.NumDescriptorRanges = (UINT)ranges[i].size();
         params[i].DescriptorTable.pDescriptorRanges = ranges[i].data();
         break;
      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         params[i].Constants = p1.Constants;
         break;
      default:
         params[i].Descriptor.ShaderRegister = p1.Descriptor.ShaderRegister;
         params[i].Descriptor.RegisterSpace = p1.Descriptor.RegisterSpace;
         break;
      }
   }
   D3D12_ROOT_SIGNATURE_DESC d0 = {};
   d0.NumParameters = d1.NumParameters; d0.pParameters = params.data();
   d0.NumStaticSamplers = d1.NumStaticSamplers; d0.pStaticSamplers = d1.pStaticSamplers;
   d0.Flags = d1.Flags;
   return D3D12SerializeRootSignature(&d0, D3D_ROOT_SIGNATURE_VERSION_1_0, blob, error);
}

/* ---- subresource upload (HelloTexture) ----------------------------------- */

struct CD3DX12_TEXTURE_COPY_LOCATION : public D3D12_TEXTURE_COPY_LOCATION {
   CD3DX12_TEXTURE_COPY_LOCATION(ID3D12Resource *res, UINT sub) {
      pResource = res; Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      PlacedFootprint = {}; SubresourceIndex = sub;
   }
   CD3DX12_TEXTURE_COPY_LOCATION(ID3D12Resource *res, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &fp) {
      pResource = res; Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; PlacedFootprint = fp;
   }
};

inline UINT64
GetRequiredIntermediateSize(ID3D12Resource *dst, UINT firstSubresource, UINT numSubresources)
{
   D3D12_RESOURCE_DESC desc = dst->GetDesc();
   UINT64 required = 0;
   ID3D12Device *device = nullptr;
   dst->GetDevice(__uuidof(ID3D12Device), (void **)&device);
   device->GetCopyableFootprints(&desc, firstSubresource, numSubresources, 0, nullptr, nullptr,
                                 nullptr, &required);
   device->Release();
   return required;
}

inline UINT64
UpdateSubresources(ID3D12GraphicsCommandList *cmdList, ID3D12Resource *dst,
                   ID3D12Resource *intermediate, UINT64 intermediateOffset, UINT firstSubresource,
                   UINT numSubresources, const D3D12_SUBRESOURCE_DATA *srcData)
{
   D3D12_RESOURCE_DESC dstDesc = dst->GetDesc();
   D3D12_RESOURCE_DESC intDesc = intermediate->GetDesc();
   std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(numSubresources);
   std::vector<UINT> rows(numSubresources);
   std::vector<UINT64> rowSizes(numSubresources);
   UINT64 required = 0;
   ID3D12Device *device = nullptr;
   dst->GetDevice(__uuidof(ID3D12Device), (void **)&device);
   device->GetCopyableFootprints(&dstDesc, firstSubresource, numSubresources, intermediateOffset,
                                 layouts.data(), rows.data(), rowSizes.data(), &required);
   device->Release();
   if (intDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       intDesc.Width < required + layouts[0].Offset)
      return 0;

   BYTE *data = nullptr;
   if (FAILED(intermediate->Map(0, nullptr, (void **)&data)))
      return 0;
   for (UINT i = 0; i < numSubresources; i++) {
      if (rowSizes[i] > (SIZE_T)-1)
         return 0;
      for (UINT z = 0; z < layouts[i].Footprint.Depth; z++) {
         BYTE *dstSlice = data + layouts[i].Offset + (SIZE_T)layouts[i].Footprint.RowPitch * rows[i] * z;
         const BYTE *srcSlice = (const BYTE *)srcData[i].pData + srcData[i].SlicePitch * z;
         for (UINT y = 0; y < rows[i]; y++)
            memcpy(dstSlice + (SIZE_T)layouts[i].Footprint.RowPitch * y,
                   srcSlice + srcData[i].RowPitch * y, (SIZE_T)rowSizes[i]);
      }
   }
   intermediate->Unmap(0, nullptr);

   if (dstDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      cmdList->CopyBufferRegion(dst, 0, intermediate, layouts[0].Offset, layouts[0].Footprint.Width);
   } else {
      for (UINT i = 0; i < numSubresources; i++) {
         CD3DX12_TEXTURE_COPY_LOCATION d(dst, i + firstSubresource);
         CD3DX12_TEXTURE_COPY_LOCATION s(intermediate, layouts[i]);
         cmdList->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
      }
   }
   return required;
}
