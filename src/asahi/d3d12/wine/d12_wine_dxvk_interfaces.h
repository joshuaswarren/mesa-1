/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * C declarations of DXVK's private presenter interfaces (public header
 * src/dxgi/dxgi_interfaces.h in DXVK, zlib licence). DXVK's dxgi.dll asks the
 * D3D12 command queue for IDXGIVkSwapChainFactory and drives presentation
 * through IDXGIVkSwapChain; the vtable orders below must match DXVK exactly.
 */
#ifndef D12_WINE_DXVK_INTERFACES_H
#define D12_WINE_DXVK_INTERFACES_H

#include <windows.h>
#include <dxgi1_6.h>

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

typedef struct DXGI_VK_HDR_METADATA {
   DXGI_HDR_METADATA_TYPE Type;
   union {
      DXGI_HDR_METADATA_HDR10 HDR10;
   };
} DXGI_VK_HDR_METADATA;

typedef struct DXGI_VK_FRAME_STATISTICS {
   UINT64 PresentCount;
   UINT64 PresentQPCTime;
} DXGI_VK_FRAME_STATISTICS;

typedef struct IDXGIVkSurfaceFactory IDXGIVkSurfaceFactory;
typedef struct IDXGIVkSwapChain3 IDXGIVkSwapChain3;
typedef struct IDXGIVkSwapChainFactory IDXGIVkSwapChainFactory;

/* {1e7895a1-1bc3-4f9c-a670-290a4bc9581a} */
typedef struct IDXGIVkSurfaceFactoryVtbl {
   HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkSurfaceFactory *, REFIID, void **);
   ULONG (STDMETHODCALLTYPE *AddRef)(IDXGIVkSurfaceFactory *);
   ULONG (STDMETHODCALLTYPE *Release)(IDXGIVkSurfaceFactory *);
   VkResult (STDMETHODCALLTYPE *CreateSurface)(IDXGIVkSurfaceFactory *, VkInstance,
                                               VkPhysicalDevice, VkSurfaceKHR *);
} IDXGIVkSurfaceFactoryVtbl;
struct IDXGIVkSurfaceFactory { const IDXGIVkSurfaceFactoryVtbl *lpVtbl; };

/* IDXGIVkSwapChain {e4a9059e-b569-46ab-8de7-501bd2bc7f7a}
 * IDXGIVkSwapChain1 {785326d4-b77b-4826-ae70-8d08308ee6d1}
 * IDXGIVkSwapChain2 {aed91093-e02e-458c-bdef-a97da1a7e6d2}
 * IDXGIVkSwapChain3 {fa25651a-ae62-4ddd-9086-29ea3144820a} */
typedef struct IDXGIVkSwapChain3Vtbl {
   HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkSwapChain3 *, REFIID, void **);
   ULONG (STDMETHODCALLTYPE *AddRef)(IDXGIVkSwapChain3 *);
   ULONG (STDMETHODCALLTYPE *Release)(IDXGIVkSwapChain3 *);
   /* IDXGIVkSwapChain */
   HRESULT (STDMETHODCALLTYPE *GetDesc)(IDXGIVkSwapChain3 *, DXGI_SWAP_CHAIN_DESC1 *);
   HRESULT (STDMETHODCALLTYPE *GetAdapter)(IDXGIVkSwapChain3 *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *GetDevice)(IDXGIVkSwapChain3 *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *GetImage)(IDXGIVkSwapChain3 *, UINT, REFIID, void **);
   UINT (STDMETHODCALLTYPE *GetImageIndex)(IDXGIVkSwapChain3 *);
   UINT (STDMETHODCALLTYPE *GetFrameLatency)(IDXGIVkSwapChain3 *);
   HANDLE (STDMETHODCALLTYPE *GetFrameLatencyEvent)(IDXGIVkSwapChain3 *);
   HRESULT (STDMETHODCALLTYPE *ChangeProperties)(IDXGIVkSwapChain3 *,
                                                 const DXGI_SWAP_CHAIN_DESC1 *,
                                                 const UINT *, IUnknown *const *);
   HRESULT (STDMETHODCALLTYPE *SetPresentRegion)(IDXGIVkSwapChain3 *, const RECT *);
   HRESULT (STDMETHODCALLTYPE *SetGammaControl)(IDXGIVkSwapChain3 *, UINT,
                                                const DXGI_RGB *);
   HRESULT (STDMETHODCALLTYPE *SetFrameLatency)(IDXGIVkSwapChain3 *, UINT);
   HRESULT (STDMETHODCALLTYPE *Present)(IDXGIVkSwapChain3 *, UINT, UINT,
                                        const DXGI_PRESENT_PARAMETERS *);
   UINT (STDMETHODCALLTYPE *CheckColorSpaceSupport)(IDXGIVkSwapChain3 *,
                                                    DXGI_COLOR_SPACE_TYPE);
   HRESULT (STDMETHODCALLTYPE *SetColorSpace)(IDXGIVkSwapChain3 *,
                                              DXGI_COLOR_SPACE_TYPE);
   HRESULT (STDMETHODCALLTYPE *SetHDRMetaData)(IDXGIVkSwapChain3 *,
                                               const DXGI_VK_HDR_METADATA *);
   /* IDXGIVkSwapChain1 */
   void (STDMETHODCALLTYPE *GetLastPresentCount)(IDXGIVkSwapChain3 *, UINT64 *);
   void (STDMETHODCALLTYPE *GetFrameStatistics)(IDXGIVkSwapChain3 *,
                                                DXGI_VK_FRAME_STATISTICS *);
   /* IDXGIVkSwapChain2 */
   void (STDMETHODCALLTYPE *SetTargetFrameRate)(IDXGIVkSwapChain3 *, double);
   /* IDXGIVkSwapChain3 */
   HRESULT (STDMETHODCALLTYPE *SetBackgroundColor)(IDXGIVkSwapChain3 *,
                                                   const DXGI_RGBA *);
   HRESULT (STDMETHODCALLTYPE *SetRotation)(IDXGIVkSwapChain3 *, DXGI_MODE_ROTATION);
} IDXGIVkSwapChain3Vtbl;
struct IDXGIVkSwapChain3 { const IDXGIVkSwapChain3Vtbl *lpVtbl; };

/* {e7d6c3ca-23a0-4e08-9f2f-ea5231df6633} */
typedef struct IDXGIVkSwapChainFactoryVtbl {
   HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkSwapChainFactory *, REFIID, void **);
   ULONG (STDMETHODCALLTYPE *AddRef)(IDXGIVkSwapChainFactory *);
   ULONG (STDMETHODCALLTYPE *Release)(IDXGIVkSwapChainFactory *);
   HRESULT (STDMETHODCALLTYPE *CreateSwapChain)(IDXGIVkSwapChainFactory *,
                                                IDXGIVkSurfaceFactory *,
                                                const DXGI_SWAP_CHAIN_DESC1 *,
                                                IDXGIVkSwapChain3 **);
} IDXGIVkSwapChainFactoryVtbl;
struct IDXGIVkSwapChainFactory { const IDXGIVkSwapChainFactoryVtbl *lpVtbl; };

static const GUID D12_IID_IDXGIVkSurfaceFactory =
   {0x1e7895a1, 0x1bc3, 0x4f9c, {0xa6, 0x70, 0x29, 0x0a, 0x4b, 0xc9, 0x58, 0x1a}};
static const GUID D12_IID_IDXGIVkSwapChain =
   {0xe4a9059e, 0xb569, 0x46ab, {0x8d, 0xe7, 0x50, 0x1b, 0xd2, 0xbc, 0x7f, 0x7a}};
static const GUID D12_IID_IDXGIVkSwapChain1 =
   {0x785326d4, 0xb77b, 0x4826, {0xae, 0x70, 0x8d, 0x08, 0x30, 0x8e, 0xe6, 0xd1}};
static const GUID D12_IID_IDXGIVkSwapChain2 =
   {0xaed91093, 0xe02e, 0x458c, {0xbd, 0xef, 0xa9, 0x7d, 0xa1, 0xa7, 0xe6, 0xd2}};
static const GUID D12_IID_IDXGIVkSwapChain3 =
   {0xfa25651a, 0xae62, 0x4ddd, {0x90, 0x86, 0x29, 0xea, 0x31, 0x44, 0x82, 0x0a}};
static const GUID D12_IID_IDXGIVkSwapChainFactory =
   {0xe7d6c3ca, 0x23a0, 0x4e08, {0x9f, 0x2f, 0xea, 0x52, 0x31, 0xdf, 0x66, 0x33}};

#endif
