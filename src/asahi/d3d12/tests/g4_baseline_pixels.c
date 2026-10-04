/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Experimental native baseline: legal opaque UMA texture, real fenced pixels.
 * This is not full G4_triangle/CopyTextureRegion or seventeen-suite parity. */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include "d12_test_abi.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W 256u
#define H 256u
#define PITCH (W * 4u)
#define BYTES (PITCH * H)

static void *slurp(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f) return NULL;
   if (fseek(f, 0, SEEK_END)) {fclose(f);return NULL;}
   long n = ftell(f);
   if (n <= 0 || fseek(f, 0, SEEK_SET)) {fclose(f);return NULL;}
   void *p = malloc((size_t)n);
   if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {free(p);fclose(f);return NULL;}
   fclose(f);*size = (size_t)n;return p;
}
static bool noop_loaded(void)
{
   FILE *f = fopen("/proc/self/maps", "r");
   if (!f) return true;
   char line[4096];bool found = false;
   while (fgets(line, sizeof(line), f)) if (strstr(line, "libasahi_noop_drm_shim.so")) found = true;
   fclose(f);return found;
}
static bool covered(unsigned x, unsigned y)
{
   /* tri.dxil: clip (1,-1),(-1,-1),(5,1); D3D pixel centres/top-left rule. */
   const float p[3][2] = {{W,H},{0,H},{3*W,0}};
   for (unsigned e=0;e<3;e++) {
      unsigned a=e,b=(e+1)%3;float dx=p[b][0]-p[a][0],dy=p[b][1]-p[a][1];
      float edge=dx*((float)y+0.5f-p[a][1])-dy*((float)x+0.5f-p[a][0]);
      bool top_left=dy==0 ? dx>0 : dy<0;
      if (edge<0 || (edge==0 && !top_left)) return false;
   }
   return true;
}
int main(int argc, char **argv)
{
   bool cpu_only = argc==4 && !strcmp(argv[1], "--cpu-roundtrip-only");
   bool pixels = argc==4 && !strcmp(argv[1], "--pixels");
   const char *enable=getenv("AURORA_D3D12_PHYSICAL_BASELINE");
   if ((!cpu_only && !pixels) || (pixels && (noop_loaded() || !enable || strcmp(enable,"1")))) {
      fprintf(stderr,"usage: %s --cpu-roundtrip-only|--pixels tri.dxil solid.dxil (physical pixels requires exact opt-in and no noop shim)\n",argv[0]);return 2;
   }
   int result=1;const char *stage="shader input";HRESULT hr=E_FAIL;
   size_t vs_size=0,fs_size=0;void *vs=slurp(argv[2],&vs_size),*fs=slurp(argv[3],&fs_size);
   uint8_t *poison=malloc(BYTES),*readback=malloc(BYTES);
   ID3D12Device *dev=NULL;ID3DBlob *blob=NULL;ID3D12RootSignature *root=NULL;
   ID3D12PipelineState *pso=NULL;ID3D12Resource *rt=NULL;ID3D12DescriptorHeap *heap=NULL;
   ID3D12CommandAllocator *alloc=NULL;ID3D12GraphicsCommandList *list=NULL;
   ID3D12CommandQueue *queue=NULL;ID3D12Fence *fence=NULL;bool mapped=false;
#define REQUIRE(ok,label) do {stage=(label);if (!(ok)) goto cleanup;} while(0)
#define CALL(call,label) do {hr=(call);REQUIRE(hr==S_OK,label);} while(0)
   REQUIRE(vs && fs && poison && readback,"shader/CPU storage");
   CALL(D3D12CreateDevice(NULL,D3D_FEATURE_LEVEL_11_0,&IID_ID3D12Device,(void **)&dev),"CreateDevice");
   REQUIRE(dev,"device object");
   D3D12_FEATURE_DATA_ARCHITECTURE1 arch={.NodeIndex=0};
   CALL(ID3D12Device_CheckFeatureSupport(dev,D3D12_FEATURE_ARCHITECTURE1,&arch,sizeof(arch)),"UMA architecture");
   REQUIRE(arch.UMA,"UMA required");
   printf("ARCH UMA=%u CacheCoherentUMA=%u (driver performs explicit CPU cache maintenance)\n",
          (unsigned)arch.UMA,(unsigned)arch.CacheCoherentUMA);
   D3D12_HEAP_PROPERTIES hp={.Type=D3D12_HEAP_TYPE_CUSTOM,
      .CPUPageProperty=D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,.MemoryPoolPreference=D3D12_MEMORY_POOL_L0,
      .CreationNodeMask=1,.VisibleNodeMask=1};
   D3D12_RESOURCE_DESC desc={.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D,.Width=W,.Height=H,
      .DepthOrArraySize=1,.MipLevels=1,.Format=DXGI_FORMAT_R8G8B8A8_UNORM,.SampleDesc={.Count=1},
      .Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET|D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS};
   CALL(ID3D12Device_CreateCommittedResource(dev,&hp,D3D12_HEAP_FLAG_NONE,&desc,
      D3D12_RESOURCE_STATE_COMMON,NULL,&IID_ID3D12Resource,(void **)&rt),"CPU-visible opaque UMA RT");
   REQUIRE(rt,"RT object");
   for (unsigned i=0;i<BYTES;i++) poison[i]=cpu_only ? (uint8_t)(i*37u+i/PITCH) : 0xa5;
   CALL(ID3D12Resource_Map(rt,0,NULL,NULL),"Map opaque texture without pointer");mapped=true;
   CALL(ID3D12Resource_WriteToSubresource(rt,0,NULL,poison,PITCH,BYTES),"write only poison/CPU pattern");
   memset(readback,0x33,BYTES);
   CALL(ID3D12Resource_ReadFromSubresource(rt,readback,PITCH,BYTES,0,NULL),"pre-execution CPU copy");
   REQUIRE(!memcmp(poison,readback,BYTES),"exact pre-execution poison/CPU roundtrip");
   ID3D12Resource_Unmap(rt,0,NULL);mapped=false;
   if (cpu_only) {result=0;goto cleanup;}
   const D3D12_ROOT_SIGNATURE_DESC rs={0};
   CALL(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1_0,&blob,NULL),"serialize root");
   REQUIRE(blob,"root blob");
   CALL(ID3D12Device_CreateRootSignature(dev,0,ID3D10Blob_GetBufferPointer(blob),ID3D10Blob_GetBufferSize(blob),
      &IID_ID3D12RootSignature,(void **)&root),"CreateRootSignature");REQUIRE(root,"root object");
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd={.pRootSignature=root,.VS={vs,vs_size},.PS={fs,fs_size},
      .RasterizerState={.FillMode=D3D12_FILL_MODE_SOLID,.CullMode=D3D12_CULL_MODE_NONE,.DepthClipEnable=TRUE},
      .PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,.NumRenderTargets=1,
      .RTVFormats={DXGI_FORMAT_R8G8B8A8_UNORM},.SampleDesc={.Count=1},.SampleMask=~0u};
   pd.BlendState.RenderTarget[0].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
   CALL(ID3D12Device_CreateGraphicsPipelineState(dev,&pd,&IID_ID3D12PipelineState,(void **)&pso),"CreateGraphicsPSO");REQUIRE(pso,"PSO object");
   const D3D12_DESCRIPTOR_HEAP_DESC hd={.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV,.NumDescriptors=1};
   CALL(ID3D12Device_CreateDescriptorHeap(dev,&hd,&IID_ID3D12DescriptorHeap,(void **)&heap),"RTV heap");REQUIRE(heap,"heap object");
   D3D12_CPU_DESCRIPTOR_HANDLE rtv=d12_test_descriptor_heap_get_cpu_start(heap);
   ID3D12Device_CreateRenderTargetView(dev,rt,NULL,rtv);
   CALL(ID3D12Device_CreateCommandAllocator(dev,D3D12_COMMAND_LIST_TYPE_DIRECT,&IID_ID3D12CommandAllocator,(void **)&alloc),"allocator");REQUIRE(alloc,"allocator object");
   CALL(ID3D12Device_CreateCommandList(dev,0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,pso,&IID_ID3D12GraphicsCommandList,(void **)&list),"list");REQUIRE(list,"list object");
   D3D12_RESOURCE_BARRIER barrier={.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition={.pResource=rt,.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                   .StateBefore=D3D12_RESOURCE_STATE_COMMON,.StateAfter=D3D12_RESOURCE_STATE_RENDER_TARGET}};
   ID3D12GraphicsCommandList_ResourceBarrier(list,1,&barrier);
   const FLOAT clear[4]={1,0,0,1};ID3D12GraphicsCommandList_ClearRenderTargetView(list,rtv,clear,0,NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list,1,&rtv,FALSE,NULL);
   const D3D12_VIEWPORT vp={.Width=W,.Height=H,.MaxDepth=1};const D3D12_RECT sc={.right=W,.bottom=H};
   ID3D12GraphicsCommandList_RSSetViewports(list,1,&vp);ID3D12GraphicsCommandList_RSSetScissorRects(list,1,&sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list,root);ID3D12GraphicsCommandList_SetPipelineState(list,pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(list,D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_DrawInstanced(list,3,1,0,0);
   barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_RENDER_TARGET;barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;
   ID3D12GraphicsCommandList_ResourceBarrier(list,1,&barrier);
   CALL(ID3D12GraphicsCommandList_Close(list),"Close actual clear/draw/COMMON transition");
   const D3D12_COMMAND_QUEUE_DESC qd={.Type=D3D12_COMMAND_LIST_TYPE_DIRECT};
   CALL(ID3D12Device_CreateCommandQueue(dev,&qd,&IID_ID3D12CommandQueue,(void **)&queue),"queue");REQUIRE(queue,"queue object");
   CALL(ID3D12Device_CreateFence(dev,0,D3D12_FENCE_FLAG_NONE,&IID_ID3D12Fence,(void **)&fence),"fence");REQUIRE(fence,"fence object");
   ID3D12CommandList *lists[]={(ID3D12CommandList *)list};ID3D12CommandQueue_ExecuteCommandLists(queue,1,lists);
   CALL(ID3D12Device_GetDeviceRemovedReason(dev),"device after submit");CALL(ID3D12CommandQueue_Signal(queue,fence,1),"signal");
   CALL(ID3D12Fence_SetEventOnCompletion(fence,1,NULL),"exact render fence");
   UINT64 done=ID3D12Fence_GetCompletedValue(fence);REQUIRE(done!=UINT64_MAX && done>=1,"fence value1 observed");
   CALL(ID3D12Device_GetDeviceRemovedReason(dev),"device after render fence");
   memset(readback,0x5a,BYTES);CALL(ID3D12Resource_Map(rt,0,NULL,NULL),"post-fence Map(NULL)");mapped=true;
   CALL(ID3D12Resource_ReadFromSubresource(rt,readback,PITCH,BYTES,0,NULL),"read actual EOT bytes through CPU API");
   unsigned bad=0,tri=0,first_x=0,first_y=0;
   for(unsigned y=0;y<H;y++) for(unsigned x=0;x<W;x++) {
      const uint8_t expected_tri[4]={0x40,0x80,0xbf,0xff},expected_clear[4]={0xff,0,0,0xff};
      bool in=covered(x,y);tri+=in;
      if(memcmp(readback+y*PITCH+x*4,in?expected_tri:expected_clear,4)) {if(!bad){first_x=x;first_y=y;}bad++;}
   }
   printf("PIXEL ORACLE total=%u model_triangle=%u mismatches=%u first=%u,%u fence=%llu\n",W*H,tri,bad,first_x,first_y,(unsigned long long)done);
   REQUIRE(!bad && tri>0 && tri<W*H,"every real pixel matches independent CPU geometry/color oracle");
   result=0;
cleanup:
   printf("G17P %s: %s stage=%s HRESULT=0x%08x %s\n",cpu_only?"CPU ROUNDTRIP ONLY":"PHYSICAL BASELINE PIXELS",
      result?"FAIL":"PASS",stage,(unsigned)hr,cpu_only?"NO_GPU_SUBMISSION":"NOT_FULL_G4_OR_SUITE_PARITY");
   if(mapped)ID3D12Resource_Unmap(rt,0,NULL);
   if(fence)ID3D12Fence_Release(fence);if(queue)ID3D12CommandQueue_Release(queue);
   if(list)ID3D12GraphicsCommandList_Release(list);if(alloc)ID3D12CommandAllocator_Release(alloc);
   if(heap)ID3D12DescriptorHeap_Release(heap);if(rt)ID3D12Resource_Release(rt);
   if(pso)ID3D12PipelineState_Release(pso);if(root)ID3D12RootSignature_Release(root);
   if(blob)ID3D10Blob_Release(blob);if(dev)ID3D12Device_Release(dev);
   free(vs);free(fs);free(poison);free(readback);return result;
}
