/* SPDX-License-Identifier: MIT */
#define INITGUID
#define COBJMACROS
/* Reused from the 2026-10-06 parity audit. The COM calls here have no
 * aggregate returns, so both native and Proton public headers share the ABI. */
#ifdef D12_CAPS_PROBE_PROTON_HEADERS
#include "vkd3d_windows.h"
#include "vkd3d_d3d12.h"
#else
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
typedef HRESULT (*create_fn)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
static void begin(const char *name, HRESULT hr) {
  printf("{\"query\":\"%s\",\"hr\":\"0x%08x\",\"fields\":{", name, (unsigned)hr);
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  setbuf(stdout, NULL);
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 3; }
  create_fn create = (create_fn)dlsym(lib, "D3D12CreateDevice");
  if (!create) return 4;
  ID3D12Device *dev = NULL;
  HRESULT hr = create(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&dev);
  printf("{\"create_device_hr\":\"0x%08x\",\"library\":\"%s\"}\n", (unsigned)hr, argv[1]);
  if (hr < 0 || !dev) return 5;

  { D3D12_FEATURE_DATA_D3D12_OPTIONS q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS", hr);
    printf("\"DoublePrecisionFloatShaderOps\":%lld", (long long)q.DoublePrecisionFloatShaderOps);
    printf(",\"OutputMergerLogicOp\":%lld", (long long)q.OutputMergerLogicOp);
    printf(",\"MinPrecisionSupport\":%lld", (long long)q.MinPrecisionSupport);
    printf(",\"TiledResourcesTier\":%lld", (long long)q.TiledResourcesTier);
    printf(",\"ResourceBindingTier\":%lld", (long long)q.ResourceBindingTier);
    printf(",\"PSSpecifiedStencilRefSupported\":%lld", (long long)q.PSSpecifiedStencilRefSupported);
    printf(",\"TypedUAVLoadAdditionalFormats\":%lld", (long long)q.TypedUAVLoadAdditionalFormats);
    printf(",\"ROVsSupported\":%lld", (long long)q.ROVsSupported);
    printf(",\"ConservativeRasterizationTier\":%lld", (long long)q.ConservativeRasterizationTier);
    printf(",\"MaxGPUVirtualAddressBitsPerResource\":%lld", (long long)q.MaxGPUVirtualAddressBitsPerResource);
    printf(",\"StandardSwizzle64KBSupported\":%lld", (long long)q.StandardSwizzle64KBSupported);
    printf(",\"CrossNodeSharingTier\":%lld", (long long)q.CrossNodeSharingTier);
    printf(",\"CrossAdapterRowMajorTextureSupported\":%lld", (long long)q.CrossAdapterRowMajorTextureSupported);
    printf(",\"VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation\":%lld", (long long)q.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation);
    printf(",\"ResourceHeapTier\":%lld", (long long)q.ResourceHeapTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_ARCHITECTURE q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ARCHITECTURE, &q, sizeof(q)); begin("D3D12_FEATURE_ARCHITECTURE", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"TileBasedRenderer\":%lld", (long long)q.TileBasedRenderer);
    printf(",\"UMA\":%lld", (long long)q.UMA);
    printf(",\"CacheCoherentUMA\":%lld", (long long)q.CacheCoherentUMA);
    puts("}}"); }
  { D3D12_FEATURE_DATA_FEATURE_LEVELS q = {0}; D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_12_0,D3D_FEATURE_LEVEL_12_1,D3D_FEATURE_LEVEL_12_2}; q.NumFeatureLevels = 5; q.pFeatureLevelsRequested = requested;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_FEATURE_LEVELS, &q, sizeof(q)); begin("D3D12_FEATURE_FEATURE_LEVELS", hr);
    printf("\"NumFeatureLevels\":%lld", (long long)q.NumFeatureLevels);
    printf(",\"MaxSupportedFeatureLevel\":%lld", (long long)q.MaxSupportedFeatureLevel);
    puts("}}"); }
  { D3D12_FEATURE_DATA_FORMAT_SUPPORT q = {0}; q.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_FORMAT_SUPPORT, &q, sizeof(q)); begin("D3D12_FEATURE_FORMAT_SUPPORT", hr);
    printf("\"Format\":%lld", (long long)q.Format);
    printf(",\"Support1\":%lld", (long long)q.Support1);
    printf(",\"Support2\":%lld", (long long)q.Support2);
    puts("}}"); }
  { D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS q = {0}; q.Format = DXGI_FORMAT_R8G8B8A8_UNORM; q.SampleCount = 4;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &q, sizeof(q)); begin("D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS", hr);
    printf("\"Format\":%lld", (long long)q.Format);
    printf(",\"SampleCount\":%lld", (long long)q.SampleCount);
    printf(",\"Flags\":%lld", (long long)q.Flags);
    printf(",\"NumQualityLevels\":%lld", (long long)q.NumQualityLevels);
    puts("}}"); }
  { D3D12_FEATURE_DATA_FORMAT_INFO q = {0}; q.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_FORMAT_INFO, &q, sizeof(q)); begin("D3D12_FEATURE_FORMAT_INFO", hr);
    printf("\"Format\":%lld", (long long)q.Format);
    printf(",\"PlaneCount\":%lld", (long long)q.PlaneCount);
    puts("}}"); }
  { D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &q, sizeof(q)); begin("D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT", hr);
    printf("\"MaxGPUVirtualAddressBitsPerResource\":%lld", (long long)q.MaxGPUVirtualAddressBitsPerResource);
    printf(",\"MaxGPUVirtualAddressBitsPerProcess\":%lld", (long long)q.MaxGPUVirtualAddressBitsPerProcess);
    puts("}}"); }
  { D3D12_FEATURE_DATA_SHADER_MODEL q = {0}; q.HighestShaderModel = D3D_SHADER_MODEL_6_8;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_MODEL, &q, sizeof(q)); begin("D3D12_FEATURE_SHADER_MODEL", hr);
    printf("\"HighestShaderModel\":%lld", (long long)q.HighestShaderModel);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS1 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS1, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS1", hr);
    printf("\"WaveOps\":%lld", (long long)q.WaveOps);
    printf(",\"WaveLaneCountMin\":%lld", (long long)q.WaveLaneCountMin);
    printf(",\"WaveLaneCountMax\":%lld", (long long)q.WaveLaneCountMax);
    printf(",\"TotalLaneCount\":%lld", (long long)q.TotalLaneCount);
    printf(",\"ExpandedComputeResourceStates\":%lld", (long long)q.ExpandedComputeResourceStates);
    printf(",\"Int64ShaderOps\":%lld", (long long)q.Int64ShaderOps);
    puts("}}"); }
  { D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT, &q, sizeof(q)); begin("D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"Support\":%lld", (long long)q.Support);
    puts("}}"); }
  { D3D12_FEATURE_DATA_ROOT_SIGNATURE q = {0}; q.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_2;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ROOT_SIGNATURE, &q, sizeof(q)); begin("D3D12_FEATURE_ROOT_SIGNATURE", hr);
    printf("\"HighestVersion\":%lld", (long long)q.HighestVersion);
    puts("}}"); }
  { D3D12_FEATURE_DATA_ARCHITECTURE1 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ARCHITECTURE1, &q, sizeof(q)); begin("D3D12_FEATURE_ARCHITECTURE1", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"TileBasedRenderer\":%lld", (long long)q.TileBasedRenderer);
    printf(",\"UMA\":%lld", (long long)q.UMA);
    printf(",\"CacheCoherentUMA\":%lld", (long long)q.CacheCoherentUMA);
    printf(",\"IsolatedMMU\":%lld", (long long)q.IsolatedMMU);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS2 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS2, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS2", hr);
    printf("\"DepthBoundsTestSupported\":%lld", (long long)q.DepthBoundsTestSupported);
    printf(",\"ProgrammableSamplePositionsTier\":%lld", (long long)q.ProgrammableSamplePositionsTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_SHADER_CACHE q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_CACHE, &q, sizeof(q)); begin("D3D12_FEATURE_SHADER_CACHE", hr);
    printf("\"SupportFlags\":%lld", (long long)q.SupportFlags);
    puts("}}"); }
  { D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY q = {0}; q.CommandListType = D3D12_COMMAND_LIST_TYPE_DIRECT; q.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &q, sizeof(q)); begin("D3D12_FEATURE_COMMAND_QUEUE_PRIORITY", hr);
    printf("\"CommandListType\":%lld", (long long)q.CommandListType);
    printf(",\"Priority\":%lld", (long long)q.Priority);
    printf(",\"PriorityForTypeIsSupported\":%lld", (long long)q.PriorityForTypeIsSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS3 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS3, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS3", hr);
    printf("\"CopyQueueTimestampQueriesSupported\":%lld", (long long)q.CopyQueueTimestampQueriesSupported);
    printf(",\"CastingFullyTypedFormatSupported\":%lld", (long long)q.CastingFullyTypedFormatSupported);
    printf(",\"WriteBufferImmediateSupportFlags\":%lld", (long long)q.WriteBufferImmediateSupportFlags);
    printf(",\"ViewInstancingTier\":%lld", (long long)q.ViewInstancingTier);
    printf(",\"BarycentricsSupported\":%lld", (long long)q.BarycentricsSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_EXISTING_HEAPS q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_EXISTING_HEAPS, &q, sizeof(q)); begin("D3D12_FEATURE_EXISTING_HEAPS", hr);
    printf("\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS4 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS4, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS4", hr);
    printf("\"MSAA64KBAlignedTextureSupported\":%lld", (long long)q.MSAA64KBAlignedTextureSupported);
    printf(",\"SharedResourceCompatibilityTier\":%lld", (long long)q.SharedResourceCompatibilityTier);
    printf(",\"Native16BitShaderOpsSupported\":%lld", (long long)q.Native16BitShaderOpsSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_SERIALIZATION q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SERIALIZATION, &q, sizeof(q)); begin("D3D12_FEATURE_SERIALIZATION", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"HeapSerializationTier\":%lld", (long long)q.HeapSerializationTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_CROSS_NODE q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_CROSS_NODE, &q, sizeof(q)); begin("D3D12_FEATURE_CROSS_NODE", hr);
    printf("\"SharingTier\":%lld", (long long)q.SharingTier);
    printf(",\"AtomicShaderInstructions\":%lld", (long long)q.AtomicShaderInstructions);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS5 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS5, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS5", hr);
    printf("\"SRVOnlyTiledResourceTier3\":%lld", (long long)q.SRVOnlyTiledResourceTier3);
    printf(",\"RenderPassesTier\":%lld", (long long)q.RenderPassesTier);
    printf(",\"RaytracingTier\":%lld", (long long)q.RaytracingTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS6 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS6, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS6", hr);
    printf("\"AdditionalShadingRatesSupported\":%lld", (long long)q.AdditionalShadingRatesSupported);
    printf(",\"PerPrimitiveShadingRateSupportedWithViewportIndexing\":%lld", (long long)q.PerPrimitiveShadingRateSupportedWithViewportIndexing);
    printf(",\"VariableShadingRateTier\":%lld", (long long)q.VariableShadingRateTier);
    printf(",\"ShadingRateImageTileSize\":%lld", (long long)q.ShadingRateImageTileSize);
    printf(",\"BackgroundProcessingSupported\":%lld", (long long)q.BackgroundProcessingSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_QUERY_META_COMMAND q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_QUERY_META_COMMAND, &q, sizeof(q)); begin("D3D12_FEATURE_QUERY_META_COMMAND", hr);
    printf("\"NodeMask\":%lld", (long long)q.NodeMask);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS7 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS7, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS7", hr);
    printf("\"MeshShaderTier\":%lld", (long long)q.MeshShaderTier);
    printf(",\"SamplerFeedbackTier\":%lld", (long long)q.SamplerFeedbackTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT, &q, sizeof(q)); begin("D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"Count\":%lld", (long long)q.Count);
    puts("}}"); }
  { D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES, &q, sizeof(q)); begin("D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES", hr);
    printf("\"NodeIndex\":%lld", (long long)q.NodeIndex);
    printf(",\"Count\":%lld", (long long)q.Count);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS8 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS8, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS8", hr);
    printf("\"UnalignedBlockTexturesSupported\":%lld", (long long)q.UnalignedBlockTexturesSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS9 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS9, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS9", hr);
    printf("\"MeshShaderPipelineStatsSupported\":%lld", (long long)q.MeshShaderPipelineStatsSupported);
    printf(",\"MeshShaderSupportsFullRangeRenderTargetArrayIndex\":%lld", (long long)q.MeshShaderSupportsFullRangeRenderTargetArrayIndex);
    printf(",\"AtomicInt64OnTypedResourceSupported\":%lld", (long long)q.AtomicInt64OnTypedResourceSupported);
    printf(",\"AtomicInt64OnGroupSharedSupported\":%lld", (long long)q.AtomicInt64OnGroupSharedSupported);
    printf(",\"DerivativesInMeshAndAmplificationShadersSupported\":%lld", (long long)q.DerivativesInMeshAndAmplificationShadersSupported);
    printf(",\"WaveMMATier\":%lld", (long long)q.WaveMMATier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS10 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS10, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS10", hr);
    printf("\"VariableRateShadingSumCombinerSupported\":%lld", (long long)q.VariableRateShadingSumCombinerSupported);
    printf(",\"MeshShaderPerPrimitiveShadingRateSupported\":%lld", (long long)q.MeshShaderPerPrimitiveShadingRateSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS11 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS11, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS11", hr);
    printf("\"AtomicInt64OnDescriptorHeapResourceSupported\":%lld", (long long)q.AtomicInt64OnDescriptorHeapResourceSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS12 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS12, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS12", hr);
    printf("\"MSPrimitivesPipelineStatisticIncludesCulledPrimitives\":%lld", (long long)q.MSPrimitivesPipelineStatisticIncludesCulledPrimitives);
    printf(",\"EnhancedBarriersSupported\":%lld", (long long)q.EnhancedBarriersSupported);
    printf(",\"RelaxedFormatCastingSupported\":%lld", (long long)q.RelaxedFormatCastingSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS13 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS13, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS13", hr);
    printf("\"UnrestrictedBufferTextureCopyPitchSupported\":%lld", (long long)q.UnrestrictedBufferTextureCopyPitchSupported);
    printf(",\"UnrestrictedVertexElementAlignmentSupported\":%lld", (long long)q.UnrestrictedVertexElementAlignmentSupported);
    printf(",\"InvertedViewportHeightFlipsYSupported\":%lld", (long long)q.InvertedViewportHeightFlipsYSupported);
    printf(",\"InvertedViewportDepthFlipsZSupported\":%lld", (long long)q.InvertedViewportDepthFlipsZSupported);
    printf(",\"TextureCopyBetweenDimensionsSupported\":%lld", (long long)q.TextureCopyBetweenDimensionsSupported);
    printf(",\"AlphaBlendFactorSupported\":%lld", (long long)q.AlphaBlendFactorSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS14 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS14, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS14", hr);
    printf("\"AdvancedTextureOpsSupported\":%lld", (long long)q.AdvancedTextureOpsSupported);
    printf(",\"WriteableMSAATexturesSupported\":%lld", (long long)q.WriteableMSAATexturesSupported);
    printf(",\"IndependentFrontAndBackStencilRefMaskSupported\":%lld", (long long)q.IndependentFrontAndBackStencilRefMaskSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS15 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS15, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS15", hr);
    printf("\"TriangleFanSupported\":%lld", (long long)q.TriangleFanSupported);
    printf(",\"DynamicIndexBufferStripCutSupported\":%lld", (long long)q.DynamicIndexBufferStripCutSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS16 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS16, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS16", hr);
    printf("\"DynamicDepthBiasSupported\":%lld", (long long)q.DynamicDepthBiasSupported);
    printf(",\"GPUUploadHeapSupported\":%lld", (long long)q.GPUUploadHeapSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS17 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS17, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS17", hr);
    printf("\"NonNormalizedCoordinateSamplersSupported\":%lld", (long long)q.NonNormalizedCoordinateSamplersSupported);
    printf(",\"ManualWriteTrackingResourceSupported\":%lld", (long long)q.ManualWriteTrackingResourceSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS18 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS18, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS18", hr);
    printf("\"RenderPassesValid\":%lld", (long long)q.RenderPassesValid);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS19 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS19, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS19", hr);
    printf("\"MismatchingOutputDimensionsSupported\":%lld", (long long)q.MismatchingOutputDimensionsSupported);
    printf(",\"SupportedSampleCountsWithNoOutputs\":%lld", (long long)q.SupportedSampleCountsWithNoOutputs);
    printf(",\"PointSamplingAddressesNeverRoundUp\":%lld", (long long)q.PointSamplingAddressesNeverRoundUp);
    printf(",\"RasterizerDesc2Supported\":%lld", (long long)q.RasterizerDesc2Supported);
    printf(",\"NarrowQuadrilateralLinesSupported\":%lld", (long long)q.NarrowQuadrilateralLinesSupported);
    printf(",\"AnisoFilterWithPointMipSupported\":%lld", (long long)q.AnisoFilterWithPointMipSupported);
    printf(",\"MaxSamplerDescriptorHeapSize\":%lld", (long long)q.MaxSamplerDescriptorHeapSize);
    printf(",\"MaxSamplerDescriptorHeapSizeWithStaticSamplers\":%lld", (long long)q.MaxSamplerDescriptorHeapSizeWithStaticSamplers);
    printf(",\"MaxViewDescriptorHeapSize\":%lld", (long long)q.MaxViewDescriptorHeapSize);
    printf(",\"ComputeOnlyCustomHeapSupported\":%lld", (long long)q.ComputeOnlyCustomHeapSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS20 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS20, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS20", hr);
    printf("\"ComputeOnlyWriteWatchSupported\":%lld", (long long)q.ComputeOnlyWriteWatchSupported);
    printf(",\"RecreateAtTier\":%lld", (long long)q.RecreateAtTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_PREDICATION q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PREDICATION, &q, sizeof(q)); begin("D3D12_FEATURE_PREDICATION", hr);
    printf("\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_PLACED_RESOURCE_SUPPORT_INFO q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO, &q, sizeof(q)); begin("D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO", hr);
    printf("\"Format\":%lld", (long long)q.Format);
    printf(",\"Dimension\":%lld", (long long)q.Dimension);
    printf(",\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_HARDWARE_COPY q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_HARDWARE_COPY, &q, sizeof(q)); begin("D3D12_FEATURE_HARDWARE_COPY", hr);
    printf("\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS21 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS21, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS21", hr);
    printf("\"WorkGraphsTier\":%lld", (long long)q.WorkGraphsTier);
    printf(",\"ExecuteIndirectTier\":%lld", (long long)q.ExecuteIndirectTier);
    printf(",\"SampleCmpGradientAndBiasSupported\":%lld", (long long)q.SampleCmpGradientAndBiasSupported);
    printf(",\"ExtendedCommandInfoSupported\":%lld", (long long)q.ExtendedCommandInfoSupported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_TIGHT_ALIGNMENT q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_TIGHT_ALIGNMENT, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_TIGHT_ALIGNMENT", hr);
    printf("\"SupportTier\":%lld", (long long)q.SupportTier);
    puts("}}"); }
  { D3D12_FEATURE_DATA_APPLICATION_SPECIFIC_DRIVER_STATE q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_APPLICATION_SPECIFIC_DRIVER_STATE, &q, sizeof(q)); begin("D3D12_FEATURE_APPLICATION_SPECIFIC_DRIVER_STATE", hr);
    printf("\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_BYTECODE_BYPASS_HASH_SUPPORTED q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_BYTECODE_BYPASS_HASH_SUPPORTED, &q, sizeof(q)); begin("D3D12_FEATURE_BYTECODE_BYPASS_HASH_SUPPORTED", hr);
    printf("\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_SHADERCACHE_ABI_SUPPORT q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_CACHE_ABI_SUPPORT, &q, sizeof(q)); begin("D3D12_FEATURE_SHADER_CACHE_ABI_SUPPORT", hr);
    printf("\"MinimumABISupportVersion\":%lld", (long long)q.MinimumABISupportVersion);
    printf(",\"MaximumABISupportVersion\":%lld", (long long)q.MaximumABISupportVersion);
    puts("}}"); }
  { D3D12_FEATURE_DATA_BARRIER_LAYOUT q = {0}; q.CommandListType = D3D12_COMMAND_LIST_TYPE_DIRECT; q.Layout = D3D12_BARRIER_LAYOUT_RENDER_TARGET;
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_BARRIER_LAYOUT, &q, sizeof(q)); begin("D3D12_FEATURE_BARRIER_LAYOUT", hr);
    printf("\"CommandListType\":%lld", (long long)q.CommandListType);
    printf(",\"Layout\":%lld", (long long)q.Layout);
    printf(",\"Supported\":%lld", (long long)q.Supported);
    puts("}}"); }
  { D3D12_FEATURE_DATA_D3D12_OPTIONS22 q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS22, &q, sizeof(q)); begin("D3D12_FEATURE_D3D12_OPTIONS22", hr);
    printf("\"ShaderExecutionReorderingActuallyReorders\":%lld", (long long)q.ShaderExecutionReorderingActuallyReorders);
    printf(",\"CreateByteOffsetViewsSupported\":%lld", (long long)q.CreateByteOffsetViewsSupported);
    printf(",\"Max1DDispatchSize\":%lld", (long long)q.Max1DDispatchSize);
    printf(",\"Max1DDispatchMeshSize\":%lld", (long long)q.Max1DDispatchMeshSize);
    puts("}}"); }
  for (unsigned sm = 0x60; sm <= 0x68; sm++) {
    D3D12_FEATURE_DATA_SHADER_MODEL q = { (D3D_SHADER_MODEL)sm };
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_MODEL, &q, sizeof(q));
    printf("{\"shader_model_request\":%u,\"hr\":\"0x%08x\",\"returned\":%u}\n",sm,(unsigned)hr,q.HighestShaderModel);
  }
  /* DISPLAYABLE is absent from the pinned Proton IDL, but is in the public
   * Microsoft enum. Its data is BOOL DisplayableTexture; sharing-tier enum. */
  {
    struct { BOOL DisplayableTexture; unsigned SharedResourceCompatibilityTier; } q = {0};
    hr = ID3D12Device_CheckFeatureSupport(dev, (D3D12_FEATURE)28, &q, sizeof(q));
    printf("{\"query\":\"D3D12_FEATURE_DISPLAYABLE\",\"hr\":\"0x%08x\",\"fields\":{\"DisplayableTexture\":%d,\"SharedResourceCompatibilityTier\":%u}}\n",(unsigned)hr,q.DisplayableTexture,q.SharedResourceCompatibilityTier);
  }
  {
    D3D_FEATURE_LEVEL fls[] = {D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_12_0,D3D_FEATURE_LEVEL_12_1,D3D_FEATURE_LEVEL_12_2};
    for (unsigned i=0; i<5; i++) {
      D3D12_FEATURE_DATA_FEATURE_LEVELS q = {.NumFeatureLevels=1,.pFeatureLevelsRequested=&fls[i]};
      hr = ID3D12Device_CheckFeatureSupport(dev,D3D12_FEATURE_FEATURE_LEVELS,&q,sizeof(q));
      printf("{\"feature_level_request\":%u,\"hr\":\"0x%08x\",\"returned\":%u}\n",fls[i],(unsigned)hr,q.MaxSupportedFeatureLevel);
    }
  }
  for (unsigned rs = 1; rs <= 3; rs++) {
    D3D12_FEATURE_DATA_ROOT_SIGNATURE q = { (D3D_ROOT_SIGNATURE_VERSION)rs };
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ROOT_SIGNATURE, &q, sizeof(q));
    printf("{\"root_signature_request\":%u,\"hr\":\"0x%08x\",\"returned\":%u}\n",rs,(unsigned)hr,q.HighestVersion);
  }
  for (unsigned f = 0; f <= 190; f++) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT q = { .Format = (DXGI_FORMAT)f };
    hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_FORMAT_SUPPORT, &q, sizeof(q));
    printf("{\"format\":%u,\"hr\":\"0x%08x\",\"support1\":%u,\"support2\":%u}\n", f, (unsigned)hr, q.Support1, q.Support2);
    for (unsigned n = 1; n <= 8; n *= 2) {
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS m = {.Format=(DXGI_FORMAT)f,.SampleCount=n};
      hr = ID3D12Device_CheckFeatureSupport(dev,D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,&m,sizeof(m));
      printf("{\"format_msaa\":%u,\"samples\":%u,\"hr\":\"0x%08x\",\"quality_levels\":%u}\n",f,n,(unsigned)hr,m.NumQualityLevels);
    }
  }

  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device, &p); printf("{\"interface\":\"ID3D12Device\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device1, &p); printf("{\"interface\":\"ID3D12Device1\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device2, &p); printf("{\"interface\":\"ID3D12Device2\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device3, &p); printf("{\"interface\":\"ID3D12Device3\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device4, &p); printf("{\"interface\":\"ID3D12Device4\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device5, &p); printf("{\"interface\":\"ID3D12Device5\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device6, &p); printf("{\"interface\":\"ID3D12Device6\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device7, &p); printf("{\"interface\":\"ID3D12Device7\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device8, &p); printf("{\"interface\":\"ID3D12Device8\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device9, &p); printf("{\"interface\":\"ID3D12Device9\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device10, &p); printf("{\"interface\":\"ID3D12Device10\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device11, &p); printf("{\"interface\":\"ID3D12Device11\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device12, &p); printf("{\"interface\":\"ID3D12Device12\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device13, &p); printf("{\"interface\":\"ID3D12Device13\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  { void *p = NULL; hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device14, &p); printf("{\"interface\":\"ID3D12Device14\",\"hr\":\"0x%08x\"}\n",(unsigned)hr); if (p) IUnknown_Release((IUnknown *)p); }
  {
    const GUID iid15 = {0x76cff76f,0x1e9b,0x4450,{0x8c,0xdc,0x34,0xf1,0xaf,0x78,0x8e,0x5b}};
    void *p = NULL; hr = ID3D12Device_QueryInterface(dev,&iid15,&p);
    printf("{\"interface\":\"ID3D12Device15\",\"hr\":\"0x%08x\"}\n",(unsigned)hr);
    if (p) IUnknown_Release((IUnknown *)p);
  }

  ID3D12Device_Release(dev); dlclose(lib); return 0; }
