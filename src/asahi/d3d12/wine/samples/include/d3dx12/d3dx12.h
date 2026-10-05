/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Minimal d3dx12 helpers for building Microsoft's D3D12HelloWorld samples
 * with mingw-w64's d3d12.h (the full d3dx12.h needs newer SDK types). Same
 * names and semantics as the public helpers the samples use; nothing else.
 */
#pragma once
#include <d3d12.h>
#include <cstring>

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
      CD3DX12_RESOURCE_BARRIER b;
      std::memset(&b, 0, sizeof(b));
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Flags = flags;
      b.Transition.pResource = res; b.Transition.StateBefore = before;
      b.Transition.StateAfter = after; b.Transition.Subresource = subresource;
      return b;
   }
};

#define D3D12_DEFAULT 0
