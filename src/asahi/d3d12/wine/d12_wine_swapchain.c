/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Presentation for the Wine build. DXVK's dxgi.dll owns the swap chain object
 * and the window; it asks our ID3D12CommandQueue for IDXGIVkSwapChainFactory
 * and hands us an IDXGIVkSurfaceFactory that creates a VkSurfaceKHR for the
 * HWND. We render into ordinary d3d12agx render-target textures, copy the
 * presented one back to the CPU on the application's queue, and push it to
 * the window through a small Vulkan (winevulkan -> Honeykrisp) swapchain:
 * staging buffer -> VkImage -> blit -> present. The CPU copy of one frame is
 * the known cost of this first version; see DESIGN.md.
 *
 * Everything here uses the public D3D12 API of our own device, so it does not
 * depend on d3d12agx internals beyond the QueryInterface hook in d12_queue.c.
 */

#include "d12_private.h"
#include "d12_wine_dxvk_interfaces.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

IDXGIAdapter *d12_wine_device_adapter(struct d12_device *device);
bool d12_wine_queue_query_interface(ID3D12CommandQueue *queue, REFIID riid,
                                    void **out);

#define D12_WINE_MAX_IMAGES 16

/* ---- Vulkan presenter ---------------------------------------------------- */

struct d12_wine_vk {
   HMODULE lib;
   PFN_vkGetInstanceProcAddr gipa;
   PFN_vkGetDeviceProcAddr gdpa;
   VkInstance instance;
   VkPhysicalDevice phys;
   VkDevice device;
   VkQueue queue;
   uint32_t queue_family;
   VkSurfaceKHR surface;
   VkSwapchainKHR swapchain;
   VkFormat swap_format;
   VkExtent2D swap_extent;
   VkPresentModeKHR present_mode;
   bool vsync;
   uint32_t swap_image_count;
   VkImage swap_images[D12_WINE_MAX_IMAGES];
   VkCommandPool pool;
   VkCommandBuffer cmd;
   VkFence fence;
   VkSemaphore acquire_sem, render_sem;
   VkBuffer staging;
   VkDeviceMemory staging_mem;
   void *staging_map;
   VkDeviceSize staging_size;
   VkImage src_image;
   VkDeviceMemory src_mem;
   VkFormat src_format;
   uint32_t src_w, src_h;
   bool out_of_date;

#define VKFN(name) PFN_vk##name name;
   VKFN(CreateInstance) VKFN(DestroyInstance) VKFN(EnumeratePhysicalDevices)
   VKFN(GetPhysicalDeviceQueueFamilyProperties) VKFN(CreateDevice)
   VKFN(DestroyDevice) VKFN(GetDeviceQueue) VKFN(DestroySurfaceKHR)
   VKFN(GetPhysicalDeviceSurfaceSupportKHR)
   VKFN(GetPhysicalDeviceSurfaceCapabilitiesKHR)
   VKFN(GetPhysicalDeviceSurfaceFormatsKHR)
   VKFN(GetPhysicalDeviceSurfacePresentModesKHR)
   VKFN(GetPhysicalDeviceMemoryProperties) VKFN(GetPhysicalDeviceProperties)
   VKFN(CreateSwapchainKHR) VKFN(DestroySwapchainKHR) VKFN(GetSwapchainImagesKHR)
   VKFN(AcquireNextImageKHR) VKFN(QueuePresentKHR) VKFN(CreateCommandPool)
   VKFN(DestroyCommandPool) VKFN(AllocateCommandBuffers) VKFN(BeginCommandBuffer)
   VKFN(EndCommandBuffer) VKFN(ResetCommandBuffer) VKFN(QueueSubmit)
   VKFN(CreateFence) VKFN(DestroyFence) VKFN(WaitForFences) VKFN(ResetFences)
   VKFN(CreateSemaphore) VKFN(DestroySemaphore) VKFN(CreateBuffer)
   VKFN(DestroyBuffer) VKFN(CreateImage) VKFN(DestroyImage)
   VKFN(GetBufferMemoryRequirements) VKFN(GetImageMemoryRequirements)
   VKFN(AllocateMemory) VKFN(FreeMemory) VKFN(BindBufferMemory)
   VKFN(BindImageMemory) VKFN(MapMemory) VKFN(UnmapMemory)
   VKFN(CmdPipelineBarrier) VKFN(CmdCopyBufferToImage) VKFN(CmdBlitImage)
   VKFN(DeviceWaitIdle) VKFN(QueueWaitIdle)
#undef VKFN
};

static VkFormat
d12_wine_vk_format(DXGI_FORMAT f)
{
   switch (f) {
   case DXGI_FORMAT_R8G8B8A8_UNORM:
   case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
   case DXGI_FORMAT_R8G8B8A8_TYPELESS:
      return VK_FORMAT_R8G8B8A8_UNORM;
   case DXGI_FORMAT_B8G8R8A8_UNORM:
   case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
   case DXGI_FORMAT_B8G8R8A8_TYPELESS:
      return VK_FORMAT_B8G8R8A8_UNORM;
   case DXGI_FORMAT_R10G10B10A2_UNORM:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
   case DXGI_FORMAT_R16G16B16A16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

static uint32_t
d12_wine_format_bpp(DXGI_FORMAT f)
{
   return f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4;
}

static bool
d12_wine_vk_load(struct d12_wine_vk *vk)
{
   vk->lib = LoadLibraryW(L"vulkan-1.dll");
   if (!vk->lib) {
      mesa_loge("wine present: vulkan-1.dll not available");
      return false;
   }
   vk->gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(vk->lib,
                                                        "vkGetInstanceProcAddr");
   if (!vk->gipa)
      return false;
#define LOAD_I(name) vk->name = (PFN_vk##name)vk->gipa(vk->instance, "vk" #name)
   LOAD_I(CreateInstance);
   return vk->CreateInstance != NULL;
}

static bool
d12_wine_vk_load_instance_fns(struct d12_wine_vk *vk)
{
   LOAD_I(DestroyInstance);
   LOAD_I(EnumeratePhysicalDevices);
   LOAD_I(GetPhysicalDeviceQueueFamilyProperties);
   LOAD_I(CreateDevice);
   LOAD_I(DestroySurfaceKHR);
   LOAD_I(GetPhysicalDeviceSurfaceSupportKHR);
   LOAD_I(GetPhysicalDeviceSurfaceCapabilitiesKHR);
   LOAD_I(GetPhysicalDeviceSurfaceFormatsKHR);
   LOAD_I(GetPhysicalDeviceSurfacePresentModesKHR);
   LOAD_I(GetPhysicalDeviceMemoryProperties);
   LOAD_I(GetPhysicalDeviceProperties);
   vk->gdpa = (PFN_vkGetDeviceProcAddr)vk->gipa(vk->instance, "vkGetDeviceProcAddr");
   return vk->gdpa && vk->CreateDevice && vk->GetPhysicalDeviceSurfaceFormatsKHR;
#undef LOAD_I
}

static bool
d12_wine_vk_load_device_fns(struct d12_wine_vk *vk)
{
#define LOAD_D(name) vk->name = (PFN_vk##name)vk->gdpa(vk->device, "vk" #name)
   LOAD_D(DestroyDevice); LOAD_D(GetDeviceQueue); LOAD_D(CreateSwapchainKHR);
   LOAD_D(DestroySwapchainKHR); LOAD_D(GetSwapchainImagesKHR);
   LOAD_D(AcquireNextImageKHR); LOAD_D(QueuePresentKHR); LOAD_D(CreateCommandPool);
   LOAD_D(DestroyCommandPool); LOAD_D(AllocateCommandBuffers); LOAD_D(BeginCommandBuffer);
   LOAD_D(EndCommandBuffer); LOAD_D(ResetCommandBuffer); LOAD_D(QueueSubmit);
   LOAD_D(CreateFence); LOAD_D(DestroyFence); LOAD_D(WaitForFences); LOAD_D(ResetFences);
   LOAD_D(CreateSemaphore); LOAD_D(DestroySemaphore); LOAD_D(CreateBuffer);
   LOAD_D(DestroyBuffer); LOAD_D(CreateImage); LOAD_D(DestroyImage);
   LOAD_D(GetBufferMemoryRequirements); LOAD_D(GetImageMemoryRequirements);
   LOAD_D(AllocateMemory); LOAD_D(FreeMemory); LOAD_D(BindBufferMemory);
   LOAD_D(BindImageMemory); LOAD_D(MapMemory); LOAD_D(UnmapMemory);
   LOAD_D(CmdPipelineBarrier); LOAD_D(CmdCopyBufferToImage); LOAD_D(CmdBlitImage);
   LOAD_D(DeviceWaitIdle); LOAD_D(QueueWaitIdle);
#undef LOAD_D
   return vk->CreateSwapchainKHR && vk->CmdBlitImage && vk->QueuePresentKHR;
}

static uint32_t
d12_wine_vk_memory_type(struct d12_wine_vk *vk, uint32_t allowed,
                        VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties props;
   vk->GetPhysicalDeviceMemoryProperties(vk->phys, &props);
   for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
      if ((allowed & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   return UINT32_MAX;
}

static void
d12_wine_vk_destroy_swapchain(struct d12_wine_vk *vk)
{
   if (vk->swapchain) {
      vk->DeviceWaitIdle(vk->device);
      vk->DestroySwapchainKHR(vk->device, vk->swapchain, NULL);
      vk->swapchain = VK_NULL_HANDLE;
   }
   vk->swap_image_count = 0;
}

static bool
d12_wine_vk_create_swapchain(struct d12_wine_vk *vk, VkFormat preferred,
                             bool vsync)
{
   VkSurfaceCapabilitiesKHR caps;
   if (vk->GetPhysicalDeviceSurfaceCapabilitiesKHR(vk->phys, vk->surface, &caps))
      return false;

   uint32_t nfmt = 0;
   vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->phys, vk->surface, &nfmt, NULL);
   VkSurfaceFormatKHR fmts[32];
   if (nfmt > 32)
      nfmt = 32;
   vk->GetPhysicalDeviceSurfaceFormatsKHR(vk->phys, vk->surface, &nfmt, fmts);
   if (!nfmt)
      return false;
   VkSurfaceFormatKHR chosen = fmts[0];
   for (uint32_t i = 0; i < nfmt; i++) {
      if (fmts[i].format == preferred) {
         chosen = fmts[i];
         break;
      }
   }
   /* Prefer a UNORM 8-bit surface over sRGB variants so the blit is a plain copy. */
   if (chosen.format != preferred) {
      for (uint32_t i = 0; i < nfmt; i++) {
         if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM ||
             fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM) {
            chosen = fmts[i];
            break;
         }
      }
   }

   uint32_t nmodes = 0;
   vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->phys, vk->surface, &nmodes, NULL);
   VkPresentModeKHR modes[8];
   if (nmodes > 8)
      nmodes = 8;
   vk->GetPhysicalDeviceSurfacePresentModesKHR(vk->phys, vk->surface, &nmodes, modes);
   VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
   if (!vsync) {
      for (uint32_t i = 0; i < nmodes; i++)
         if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
            mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      if (mode == VK_PRESENT_MODE_FIFO_KHR)
         for (uint32_t i = 0; i < nmodes; i++)
            if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
               mode = VK_PRESENT_MODE_MAILBOX_KHR;
   }

   VkExtent2D extent = caps.currentExtent;
   if (extent.width == UINT32_MAX) {
      extent.width = vk->src_w;
      extent.height = vk->src_h;
   }
   if (extent.width == 0 || extent.height == 0)
      return false;

   uint32_t min_images = caps.minImageCount > 2 ? caps.minImageCount : 2;
   if (caps.maxImageCount && min_images > caps.maxImageCount)
      min_images = caps.maxImageCount;

   VkSwapchainCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = vk->surface,
      .minImageCount = min_images,
      .imageFormat = chosen.format,
      .imageColorSpace = chosen.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha = (caps.supportedCompositeAlpha &
                         VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                           ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                           : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode = mode,
      .clipped = VK_TRUE,
      .oldSwapchain = vk->swapchain,
   };
   VkSwapchainKHR sc;
   VkResult res = vk->CreateSwapchainKHR(vk->device, &info, NULL, &sc);
   if (vk->swapchain) {
      vk->DeviceWaitIdle(vk->device);
      vk->DestroySwapchainKHR(vk->device, vk->swapchain, NULL);
      vk->swapchain = VK_NULL_HANDLE;
   }
   if (res != VK_SUCCESS) {
      mesa_loge("wine present: vkCreateSwapchainKHR failed (%d)", res);
      return false;
   }
   vk->swapchain = sc;
   vk->swap_format = chosen.format;
   vk->swap_extent = extent;
   vk->present_mode = mode;
   vk->vsync = vsync;
   vk->swap_image_count = D12_WINE_MAX_IMAGES;
   vk->GetSwapchainImagesKHR(vk->device, sc, &vk->swap_image_count,
                             vk->swap_images);
   vk->out_of_date = false;
   return true;
}

static void
d12_wine_vk_destroy_source(struct d12_wine_vk *vk)
{
   if (!vk->device)
      return;
   if (vk->src_image) {
      vk->DestroyImage(vk->device, vk->src_image, NULL);
      vk->src_image = VK_NULL_HANDLE;
   }
   if (vk->src_mem) {
      vk->FreeMemory(vk->device, vk->src_mem, NULL);
      vk->src_mem = VK_NULL_HANDLE;
   }
   if (vk->staging_map) {
      vk->UnmapMemory(vk->device, vk->staging_mem);
      vk->staging_map = NULL;
   }
   if (vk->staging) {
      vk->DestroyBuffer(vk->device, vk->staging, NULL);
      vk->staging = VK_NULL_HANDLE;
   }
   if (vk->staging_mem) {
      vk->FreeMemory(vk->device, vk->staging_mem, NULL);
      vk->staging_mem = VK_NULL_HANDLE;
   }
}

static bool
d12_wine_vk_create_source(struct d12_wine_vk *vk, uint32_t w, uint32_t h,
                          VkFormat format, uint32_t bpp)
{
   d12_wine_vk_destroy_source(vk);
   vk->src_w = w;
   vk->src_h = h;
   vk->src_format = format;
   vk->staging_size = (VkDeviceSize)w * h * bpp;

   VkBufferCreateInfo binfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = vk->staging_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
   };
   if (vk->CreateBuffer(vk->device, &binfo, NULL, &vk->staging))
      return false;
   VkMemoryRequirements req;
   vk->GetBufferMemoryRequirements(vk->device, vk->staging, &req);
   VkMemoryAllocateInfo ainfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = d12_wine_vk_memory_type(
         vk, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   if (ainfo.memoryTypeIndex == UINT32_MAX ||
       vk->AllocateMemory(vk->device, &ainfo, NULL, &vk->staging_mem) ||
       vk->BindBufferMemory(vk->device, vk->staging, vk->staging_mem, 0) ||
       vk->MapMemory(vk->device, vk->staging_mem, 0, VK_WHOLE_SIZE, 0,
                     &vk->staging_map))
      return false;

   VkImageCreateInfo iinfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   if (vk->CreateImage(vk->device, &iinfo, NULL, &vk->src_image))
      return false;
   vk->GetImageMemoryRequirements(vk->device, vk->src_image, &req);
   ainfo.allocationSize = req.size;
   ainfo.memoryTypeIndex = d12_wine_vk_memory_type(
      vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (ainfo.memoryTypeIndex == UINT32_MAX)
      ainfo.memoryTypeIndex = d12_wine_vk_memory_type(vk, req.memoryTypeBits, 0);
   if (vk->AllocateMemory(vk->device, &ainfo, NULL, &vk->src_mem) ||
       vk->BindImageMemory(vk->device, vk->src_image, vk->src_mem, 0))
      return false;
   return true;
}

static void
d12_wine_vk_fini(struct d12_wine_vk *vk)
{
   if (vk->device) {
      vk->DeviceWaitIdle(vk->device);
      d12_wine_vk_destroy_source(vk);
      d12_wine_vk_destroy_swapchain(vk);
      if (vk->acquire_sem)
         vk->DestroySemaphore(vk->device, vk->acquire_sem, NULL);
      if (vk->render_sem)
         vk->DestroySemaphore(vk->device, vk->render_sem, NULL);
      if (vk->fence)
         vk->DestroyFence(vk->device, vk->fence, NULL);
      if (vk->pool)
         vk->DestroyCommandPool(vk->device, vk->pool, NULL);
      vk->DestroyDevice(vk->device, NULL);
   }
   if (vk->instance) {
      if (vk->surface)
         vk->DestroySurfaceKHR(vk->instance, vk->surface, NULL);
      vk->DestroyInstance(vk->instance, NULL);
   }
   if (vk->lib)
      FreeLibrary(vk->lib);
   memset(vk, 0, sizeof(*vk));
}

static bool
d12_wine_vk_init(struct d12_wine_vk *vk, IDXGIVkSurfaceFactory *surface_factory)
{
   memset(vk, 0, sizeof(*vk));
   if (!d12_wine_vk_load(vk))
      goto fail;

   const char *inst_exts[] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "d3d12agx-present",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo iinfo = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = ARRAY_SIZE(inst_exts),
      .ppEnabledExtensionNames = inst_exts,
   };
   if (vk->CreateInstance(&iinfo, NULL, &vk->instance)) {
      mesa_loge("wine present: vkCreateInstance failed");
      goto fail;
   }
   if (!d12_wine_vk_load_instance_fns(vk))
      goto fail;

   uint32_t ndev = 1;
   if (vk->EnumeratePhysicalDevices(vk->instance, &ndev, &vk->phys) < 0 || !ndev) {
      mesa_loge("wine present: no Vulkan physical device");
      goto fail;
   }

   VkResult sres = surface_factory->lpVtbl->CreateSurface(
      surface_factory, vk->instance, vk->phys, &vk->surface);
   if (sres != VK_SUCCESS) {
      mesa_loge("wine present: surface creation failed (%d)", sres);
      goto fail;
   }

   uint32_t nfam = 0;
   vk->GetPhysicalDeviceQueueFamilyProperties(vk->phys, &nfam, NULL);
   VkQueueFamilyProperties fams[16];
   if (nfam > 16)
      nfam = 16;
   vk->GetPhysicalDeviceQueueFamilyProperties(vk->phys, &nfam, fams);
   vk->queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < nfam; i++) {
      VkBool32 present = VK_FALSE;
      vk->GetPhysicalDeviceSurfaceSupportKHR(vk->phys, i, vk->surface, &present);
      if (present && (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
         vk->queue_family = i;
         break;
      }
   }
   if (vk->queue_family == UINT32_MAX) {
      mesa_loge("wine present: no present-capable graphics queue");
      goto fail;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = vk->queue_family,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   const char *dev_exts[] = {"VK_KHR_swapchain"};
   VkDeviceCreateInfo dinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qinfo,
      .enabledExtensionCount = ARRAY_SIZE(dev_exts),
      .ppEnabledExtensionNames = dev_exts,
   };
   if (vk->CreateDevice(vk->phys, &dinfo, NULL, &vk->device)) {
      mesa_loge("wine present: vkCreateDevice failed");
      goto fail;
   }
   if (!d12_wine_vk_load_device_fns(vk))
      goto fail;
   vk->GetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

   VkCommandPoolCreateInfo pinfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = vk->queue_family,
   };
   if (vk->CreateCommandPool(vk->device, &pinfo, NULL, &vk->pool))
      goto fail;
   VkCommandBufferAllocateInfo cinfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = vk->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (vk->AllocateCommandBuffers(vk->device, &cinfo, &vk->cmd))
      goto fail;
   VkFenceCreateInfo finfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
   };
   if (vk->CreateFence(vk->device, &finfo, NULL, &vk->fence))
      goto fail;
   VkSemaphoreCreateInfo sinfo = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   if (vk->CreateSemaphore(vk->device, &sinfo, NULL, &vk->acquire_sem) ||
       vk->CreateSemaphore(vk->device, &sinfo, NULL, &vk->render_sem))
      goto fail;
   return true;

fail:
   d12_wine_vk_fini(vk);
   return false;
}

static void
d12_wine_vk_image_barrier(struct d12_wine_vk *vk, VkImage image,
                          VkImageLayout from, VkImageLayout to,
                          VkAccessFlags src_access, VkAccessFlags dst_access)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vk->CmdPipelineBarrier(vk->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                          1, &b);
}

/* Pixels are already in vk->staging_map. Upload, blit, present. */
static HRESULT
d12_wine_vk_present(struct d12_wine_vk *vk, bool vsync)
{
   if (!vk->swapchain || vk->out_of_date || vk->vsync != vsync) {
      if (!d12_wine_vk_create_swapchain(vk, vk->src_format, vsync))
         return DXGI_ERROR_DEVICE_RESET;
   }

   for (int attempt = 0; attempt < 2; attempt++) {
      vk->WaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);

      uint32_t idx = 0;
      VkResult res = vk->AcquireNextImageKHR(vk->device, vk->swapchain, UINT64_MAX,
                                             vk->acquire_sem, VK_NULL_HANDLE, &idx);
      if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
         if (!d12_wine_vk_create_swapchain(vk, vk->src_format, vsync))
            return DXGI_ERROR_DEVICE_RESET;
         continue;
      }
      if (res != VK_SUCCESS)
         return DXGI_ERROR_DEVICE_RESET;

      vk->ResetFences(vk->device, 1, &vk->fence);
      vk->ResetCommandBuffer(vk->cmd, 0);
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      vk->BeginCommandBuffer(vk->cmd, &begin);

      d12_wine_vk_image_barrier(vk, vk->src_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                                VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferImageCopy copy = {
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {vk->src_w, vk->src_h, 1},
      };
      vk->CmdCopyBufferToImage(vk->cmd, vk->staging, vk->src_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      d12_wine_vk_image_barrier(vk, vk->src_image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_TRANSFER_READ_BIT);
      d12_wine_vk_image_barrier(vk, vk->swap_images[idx],
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                                VK_ACCESS_TRANSFER_WRITE_BIT);
      VkImageBlit blit = {
         .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .srcOffsets = {{0, 0, 0}, {(int32_t)vk->src_w, (int32_t)vk->src_h, 1}},
         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .dstOffsets = {{0, 0, 0},
                        {(int32_t)vk->swap_extent.width,
                         (int32_t)vk->swap_extent.height, 1}},
      };
      vk->CmdBlitImage(vk->cmd, vk->src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       vk->swap_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);
      d12_wine_vk_image_barrier(vk, vk->swap_images[idx],
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                VK_ACCESS_TRANSFER_WRITE_BIT, 0);
      vk->EndCommandBuffer(vk->cmd);

      VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &vk->acquire_sem,
         .pWaitDstStageMask = &wait_stage,
         .commandBufferCount = 1,
         .pCommandBuffers = &vk->cmd,
         .signalSemaphoreCount = 1,
         .pSignalSemaphores = &vk->render_sem,
      };
      if (vk->QueueSubmit(vk->queue, 1, &submit, vk->fence))
         return DXGI_ERROR_DEVICE_RESET;

      VkPresentInfoKHR present = {
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &vk->render_sem,
         .swapchainCount = 1,
         .pSwapchains = &vk->swapchain,
         .pImageIndices = &idx,
      };
      res = vk->QueuePresentKHR(vk->queue, &present);
      if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
         vk->out_of_date = true;
      else if (res != VK_SUCCESS)
         return DXGI_ERROR_DEVICE_RESET;
      return S_OK;
   }
   return DXGI_ERROR_DEVICE_RESET;
}

/* ---- presenter object ---------------------------------------------------- */

struct d12_wine_presenter {
   IDXGIVkSwapChain3 iface;
   int refcount;

   ID3D12CommandQueue *queue;
   ID3D12Device *device;
   struct d12_device *d12dev;
   IDXGIVkSurfaceFactory *surface_factory;

   DXGI_SWAP_CHAIN_DESC1 desc;
   ID3D12Resource *images[D12_WINE_MAX_IMAGES];
   UINT image_index;

   /* GPU->CPU readback of the presented image on the app's queue. */
   ID3D12CommandAllocator *alloc;
   ID3D12GraphicsCommandList *list;
   ID3D12Fence *fence;
   UINT64 fence_value;
   HANDLE fence_event;
   ID3D12Resource *readback;
   UINT readback_pitch;

   struct d12_wine_vk vk;
   bool vk_ready;

   HANDLE latency_sem;
   UINT frame_latency;
   UINT64 present_count;
   UINT64 last_present_qpc;
   DXGI_COLOR_SPACE_TYPE color_space;
   simple_mtx_t lock;
};

static inline struct d12_wine_presenter *
presenter_from(IDXGIVkSwapChain3 *iface)
{
   return (struct d12_wine_presenter *)iface;
}

static void
presenter_destroy_images(struct d12_wine_presenter *p)
{
   for (unsigned i = 0; i < D12_WINE_MAX_IMAGES; i++) {
      if (p->images[i]) {
         ID3D12Resource_Release(p->images[i]);
         p->images[i] = NULL;
      }
   }
   if (p->readback) {
      ID3D12Resource_Release(p->readback);
      p->readback = NULL;
   }
}

static HRESULT
presenter_create_images(struct d12_wine_presenter *p)
{
   presenter_destroy_images(p);

   if (p->desc.BufferCount < 1 || p->desc.BufferCount > D12_WINE_MAX_IMAGES ||
       !p->desc.Width || !p->desc.Height)
      return DXGI_ERROR_INVALID_CALL;
   if (d12_wine_vk_format(p->desc.Format) == VK_FORMAT_UNDEFINED) {
      mesa_loge("wine present: unsupported swapchain format %u", p->desc.Format);
      return DXGI_ERROR_INVALID_CALL;
   }

   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = p->desc.Width,
      .Height = p->desc.Height,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = p->desc.Format,
      .SampleDesc = {1, 0},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      /* SIMULTANEOUS_ACCESS selects the GPU-tiled, renderable layout
       * (d12_resource_compute_texture_layout: interop_layout); a plain
       * one-level 2D render target would be laid out linear, which the
       * native G17P background/end-of-tile path rejects ("attachment-shape
       * ... tiling=0"). Swapchain images are shared-access images anyway. */
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
               D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
   };
   if (p->desc.BufferUsage & DXGI_USAGE_UNORDERED_ACCESS)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

   for (UINT i = 0; i < p->desc.BufferCount; i++) {
      HRESULT hr = ID3D12Device_CreateCommittedResource(
         p->device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
         D3D12_RESOURCE_STATE_PRESENT, NULL, &IID_ID3D12Resource,
         (void **)&p->images[i]);
      if (FAILED(hr)) {
         mesa_loge("wine present: back buffer %u creation failed 0x%08x", i,
                   (unsigned)hr);
         presenter_destroy_images(p);
         return hr;
      }
   }

   UINT bpp = d12_wine_format_bpp(p->desc.Format);
   p->readback_pitch = ALIGN_POT(p->desc.Width * bpp,
                                 D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
   D3D12_HEAP_PROPERTIES rb_heap = {.Type = D3D12_HEAP_TYPE_READBACK};
   D3D12_RESOURCE_DESC rb_desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = (UINT64)p->readback_pitch * p->desc.Height,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_UNKNOWN,
      .SampleDesc = {1, 0},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   HRESULT hr = ID3D12Device_CreateCommittedResource(
      p->device, &rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
      D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
      (void **)&p->readback);
   if (FAILED(hr)) {
      presenter_destroy_images(p);
      return hr;
   }

   if (p->vk_ready &&
       !d12_wine_vk_create_source(&p->vk, p->desc.Width, p->desc.Height,
                                  d12_wine_vk_format(p->desc.Format), bpp)) {
      mesa_loge("wine present: Vulkan source image creation failed");
      return E_FAIL;
   }
   p->image_index = 0;
   return S_OK;
}

static HRESULT
presenter_init_readback(struct d12_wine_presenter *p)
{
   HRESULT hr = ID3D12Device_CreateCommandAllocator(
      p->device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
      (void **)&p->alloc);
   if (FAILED(hr))
      return hr;
   hr = ID3D12Device_CreateCommandList(p->device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       p->alloc, NULL,
                                       &IID_ID3D12GraphicsCommandList,
                                       (void **)&p->list);
   if (FAILED(hr))
      return hr;
   ID3D12GraphicsCommandList_Close(p->list);
   hr = ID3D12Device_CreateFence(p->device, 0, D3D12_FENCE_FLAG_NONE,
                                 &IID_ID3D12Fence, (void **)&p->fence);
   if (FAILED(hr))
      return hr;
   p->fence_event = CreateEventW(NULL, FALSE, FALSE, NULL);
   return p->fence_event ? S_OK : E_FAIL;
}

/* Copy back buffer `index` to the readback buffer and wait for it. */
static HRESULT
presenter_readback(struct d12_wine_presenter *p, UINT index)
{
   ID3D12Resource *src = p->images[index];
   HRESULT hr = ID3D12CommandAllocator_Reset(p->alloc);
   if (FAILED(hr))
      return hr;
   hr = ID3D12GraphicsCommandList_Reset(p->list, p->alloc, NULL);
   if (FAILED(hr))
      return hr;

   D3D12_RESOURCE_BARRIER to_copy = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {
         .pResource = src,
         .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
         .StateBefore = D3D12_RESOURCE_STATE_PRESENT,
         .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE,
      },
   };
   ID3D12GraphicsCommandList_ResourceBarrier(p->list, 1, &to_copy);

   D3D12_TEXTURE_COPY_LOCATION dst = {
      .pResource = p->readback,
      .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
      .PlacedFootprint = {
         .Offset = 0,
         .Footprint = {
            .Format = p->desc.Format,
            .Width = p->desc.Width,
            .Height = p->desc.Height,
            .Depth = 1,
            .RowPitch = p->readback_pitch,
         },
      },
   };
   D3D12_TEXTURE_COPY_LOCATION srcloc = {
      .pResource = src,
      .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
      .SubresourceIndex = 0,
   };
   ID3D12GraphicsCommandList_CopyTextureRegion(p->list, &dst, 0, 0, 0, &srcloc,
                                               NULL);

   D3D12_RESOURCE_BARRIER back = to_copy;
   back.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
   back.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
   ID3D12GraphicsCommandList_ResourceBarrier(p->list, 1, &back);
   hr = ID3D12GraphicsCommandList_Close(p->list);
   if (FAILED(hr))
      return hr;

   ID3D12CommandList *lists[] = {(ID3D12CommandList *)p->list};
   ID3D12CommandQueue_ExecuteCommandLists(p->queue, 1, lists);
   UINT64 value = ++p->fence_value;
   hr = ID3D12CommandQueue_Signal(p->queue, p->fence, value);
   if (FAILED(hr))
      return hr;
   if (ID3D12Fence_GetCompletedValue(p->fence) < value) {
      hr = ID3D12Fence_SetEventOnCompletion(p->fence, value, p->fence_event);
      if (FAILED(hr))
         return hr;
      if (WaitForSingleObject(p->fence_event, 5000) != WAIT_OBJECT_0) {
         mesa_loge("wine present: GPU readback timed out");
         return DXGI_ERROR_DEVICE_HUNG;
      }
   }
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_QueryInterface(IDXGIVkSwapChain3 *iface, REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChain) ||
       d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChain1) ||
       d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChain2) ||
       d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChain3)) {
      p_atomic_inc(&presenter_from(iface)->refcount);
      *out = iface;
      return S_OK;
   }
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
presenter_AddRef(IDXGIVkSwapChain3 *iface)
{
   return p_atomic_inc_return(&presenter_from(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
presenter_Release(IDXGIVkSwapChain3 *iface)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   ULONG ref = p_atomic_dec_return(&p->refcount);
   if (ref)
      return ref;

   if (p->vk_ready)
      d12_wine_vk_fini(&p->vk);
   presenter_destroy_images(p);
   if (p->list)
      ID3D12GraphicsCommandList_Release(p->list);
   if (p->alloc)
      ID3D12CommandAllocator_Release(p->alloc);
   if (p->fence)
      ID3D12Fence_Release(p->fence);
   if (p->fence_event)
      CloseHandle(p->fence_event);
   if (p->latency_sem)
      CloseHandle(p->latency_sem);
   if (p->surface_factory)
      p->surface_factory->lpVtbl->Release(p->surface_factory);
   if (p->device)
      ID3D12Device_Release(p->device);
   if (p->queue)
      ID3D12CommandQueue_Release(p->queue);
   simple_mtx_destroy(&p->lock);
   free(p);
   return 0;
}

static HRESULT STDMETHODCALLTYPE
presenter_GetDesc(IDXGIVkSwapChain3 *iface, DXGI_SWAP_CHAIN_DESC1 *desc)
{
   if (!desc)
      return E_INVALIDARG;
   *desc = presenter_from(iface)->desc;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_GetAdapter(IDXGIVkSwapChain3 *iface, REFIID riid, void **out)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   if (!out)
      return E_POINTER;
   *out = NULL;
   IDXGIAdapter *adapter = d12_wine_device_adapter(p->d12dev);
   if (!adapter) {
      /* The application created the device without an adapter: hand back the
       * first one DXGI enumerates, which is the only GPU on this machine. */
      HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
      HRESULT (WINAPI *create)(REFIID, void **) =
         dxgi ? (void *)GetProcAddress(dxgi, "CreateDXGIFactory1") : NULL;
      IDXGIFactory1 *factory = NULL;
      if (!create || FAILED(create(&IID_IDXGIFactory1, (void **)&factory)))
         return DXGI_ERROR_NOT_FOUND;
      IDXGIAdapter1 *a1 = NULL;
      HRESULT hr = IDXGIFactory1_EnumAdapters1(factory, 0, &a1);
      IDXGIFactory1_Release(factory);
      if (FAILED(hr))
         return hr;
      adapter = (IDXGIAdapter *)a1;
   }
   HRESULT hr = IDXGIAdapter_QueryInterface(adapter, riid, out);
   IDXGIAdapter_Release(adapter);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
presenter_GetDevice(IDXGIVkSwapChain3 *iface, REFIID riid, void **out)
{
   return ID3D12Device_QueryInterface(presenter_from(iface)->device, riid, out);
}

static HRESULT STDMETHODCALLTYPE
presenter_GetImage(IDXGIVkSwapChain3 *iface, UINT index, REFIID riid, void **out)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   if (!out)
      return E_POINTER;
   if (index >= p->desc.BufferCount || !p->images[index]) {
      *out = NULL;
      return DXGI_ERROR_INVALID_CALL;
   }
   return ID3D12Resource_QueryInterface(p->images[index], riid, out);
}

static UINT STDMETHODCALLTYPE
presenter_GetImageIndex(IDXGIVkSwapChain3 *iface)
{
   return presenter_from(iface)->image_index;
}

static UINT STDMETHODCALLTYPE
presenter_GetFrameLatency(IDXGIVkSwapChain3 *iface)
{
   return presenter_from(iface)->frame_latency;
}

static HANDLE STDMETHODCALLTYPE
presenter_GetFrameLatencyEvent(IDXGIVkSwapChain3 *iface)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   HANDLE dup = NULL;
   if (p->latency_sem &&
       !DuplicateHandle(GetCurrentProcess(), p->latency_sem, GetCurrentProcess(),
                        &dup, 0, FALSE, DUPLICATE_SAME_ACCESS))
      return NULL;
   return dup;
}

static HRESULT STDMETHODCALLTYPE
presenter_ChangeProperties(IDXGIVkSwapChain3 *iface,
                           const DXGI_SWAP_CHAIN_DESC1 *desc,
                           const UINT *node_masks, IUnknown *const *queues)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   (void)node_masks;
   (void)queues;
   simple_mtx_lock(&p->lock);
   /* Drain the queue so no in-flight work references the old buffers. */
   UINT64 value = ++p->fence_value;
   if (SUCCEEDED(ID3D12CommandQueue_Signal(p->queue, p->fence, value)) &&
       ID3D12Fence_GetCompletedValue(p->fence) < value) {
      ID3D12Fence_SetEventOnCompletion(p->fence, value, p->fence_event);
      WaitForSingleObject(p->fence_event, 5000);
   }
   p->desc = *desc;
   HRESULT hr = presenter_create_images(p);
   if (p->vk_ready)
      p->vk.out_of_date = true;
   simple_mtx_unlock(&p->lock);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetPresentRegion(IDXGIVkSwapChain3 *iface, const RECT *region)
{
   (void)iface; (void)region;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetGammaControl(IDXGIVkSwapChain3 *iface, UINT count,
                          const DXGI_RGB *points)
{
   (void)iface; (void)count; (void)points;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetFrameLatency(IDXGIVkSwapChain3 *iface, UINT latency)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   if (!latency || latency > DXGI_MAX_SWAP_CHAIN_BUFFERS)
      return DXGI_ERROR_INVALID_CALL;
   p->frame_latency = latency;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_Present(IDXGIVkSwapChain3 *iface, UINT sync_interval, UINT flags,
                  const DXGI_PRESENT_PARAMETERS *params)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   (void)params;
   if (flags & DXGI_PRESENT_TEST)
      return S_OK;

   simple_mtx_lock(&p->lock);
   UINT index = p->image_index;
   HRESULT hr = presenter_readback(p, index);
   if (SUCCEEDED(hr) && p->vk_ready) {
      D3D12_RANGE range = {0, (SIZE_T)p->readback_pitch * p->desc.Height};
      void *src = NULL;
      hr = ID3D12Resource_Map(p->readback, 0, &range, &src);
      if (SUCCEEDED(hr)) {
         UINT bpp = d12_wine_format_bpp(p->desc.Format);
         size_t row = (size_t)p->desc.Width * bpp;
         uint8_t *dst = p->vk.staging_map;
         for (UINT y = 0; y < p->desc.Height; y++)
            memcpy(dst + y * row, (uint8_t *)src + (size_t)y * p->readback_pitch,
                   row);
         D3D12_RANGE none = {0, 0};
         ID3D12Resource_Unmap(p->readback, 0, &none);
         hr = d12_wine_vk_present(&p->vk, sync_interval != 0);
      }
   }
   if (SUCCEEDED(hr)) {
      p->image_index = (p->image_index + 1) % p->desc.BufferCount;
      p->present_count++;
      LARGE_INTEGER qpc;
      QueryPerformanceCounter(&qpc);
      p->last_present_qpc = qpc.QuadPart;
      if (p->latency_sem)
         ReleaseSemaphore(p->latency_sem, 1, NULL);
   }
   simple_mtx_unlock(&p->lock);
   return hr;
}

static UINT STDMETHODCALLTYPE
presenter_CheckColorSpaceSupport(IDXGIVkSwapChain3 *iface,
                                 DXGI_COLOR_SPACE_TYPE space)
{
   (void)iface;
   return space == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
             ? DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT
             : 0;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetColorSpace(IDXGIVkSwapChain3 *iface, DXGI_COLOR_SPACE_TYPE space)
{
   if (space != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)
      return E_INVALIDARG;
   presenter_from(iface)->color_space = space;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetHDRMetaData(IDXGIVkSwapChain3 *iface,
                         const DXGI_VK_HDR_METADATA *metadata)
{
   (void)iface; (void)metadata;
   return S_OK;
}

static void STDMETHODCALLTYPE
presenter_GetLastPresentCount(IDXGIVkSwapChain3 *iface, UINT64 *count)
{
   *count = presenter_from(iface)->present_count;
}

static void STDMETHODCALLTYPE
presenter_GetFrameStatistics(IDXGIVkSwapChain3 *iface,
                             DXGI_VK_FRAME_STATISTICS *stats)
{
   struct d12_wine_presenter *p = presenter_from(iface);
   stats->PresentCount = p->present_count;
   stats->PresentQPCTime = p->last_present_qpc;
}

static void STDMETHODCALLTYPE
presenter_SetTargetFrameRate(IDXGIVkSwapChain3 *iface, double rate)
{
   (void)iface; (void)rate;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetBackgroundColor(IDXGIVkSwapChain3 *iface, const DXGI_RGBA *color)
{
   (void)iface; (void)color;
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
presenter_SetRotation(IDXGIVkSwapChain3 *iface, DXGI_MODE_ROTATION rotation)
{
   (void)iface;
   return rotation == DXGI_MODE_ROTATION_IDENTITY ? S_OK : E_INVALIDARG;
}

static const IDXGIVkSwapChain3Vtbl presenter_vtbl = {
   presenter_QueryInterface,
   presenter_AddRef,
   presenter_Release,
   presenter_GetDesc,
   presenter_GetAdapter,
   presenter_GetDevice,
   presenter_GetImage,
   presenter_GetImageIndex,
   presenter_GetFrameLatency,
   presenter_GetFrameLatencyEvent,
   presenter_ChangeProperties,
   presenter_SetPresentRegion,
   presenter_SetGammaControl,
   presenter_SetFrameLatency,
   presenter_Present,
   presenter_CheckColorSpaceSupport,
   presenter_SetColorSpace,
   presenter_SetHDRMetaData,
   presenter_GetLastPresentCount,
   presenter_GetFrameStatistics,
   presenter_SetTargetFrameRate,
   presenter_SetBackgroundColor,
   presenter_SetRotation,
};

/* ---- factory tear-off on the command queue -------------------------------- */

struct d12_wine_swapchain_factory {
   IDXGIVkSwapChainFactory iface;
   int refcount;
   ID3D12CommandQueue *queue;
};

static HRESULT STDMETHODCALLTYPE
factory_QueryInterface(IDXGIVkSwapChainFactory *iface, REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChainFactory)) {
      p_atomic_inc(&((struct d12_wine_swapchain_factory *)iface)->refcount);
      *out = iface;
      return S_OK;
   }
   /* Anything else is the queue's business. */
   return ID3D12CommandQueue_QueryInterface(
      ((struct d12_wine_swapchain_factory *)iface)->queue, riid, out);
}

static ULONG STDMETHODCALLTYPE
factory_AddRef(IDXGIVkSwapChainFactory *iface)
{
   return p_atomic_inc_return(&((struct d12_wine_swapchain_factory *)iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
factory_Release(IDXGIVkSwapChainFactory *iface)
{
   struct d12_wine_swapchain_factory *f = (struct d12_wine_swapchain_factory *)iface;
   ULONG ref = p_atomic_dec_return(&f->refcount);
   if (!ref) {
      ID3D12CommandQueue_Release(f->queue);
      free(f);
   }
   return ref;
}

static HRESULT STDMETHODCALLTYPE
factory_CreateSwapChain(IDXGIVkSwapChainFactory *iface,
                        IDXGIVkSurfaceFactory *surface_factory,
                        const DXGI_SWAP_CHAIN_DESC1 *desc,
                        IDXGIVkSwapChain3 **out)
{
   struct d12_wine_swapchain_factory *f = (struct d12_wine_swapchain_factory *)iface;
   if (!out || !desc)
      return E_INVALIDARG;
   *out = NULL;

   struct d12_wine_presenter *p = calloc(1, sizeof(*p));
   if (!p)
      return E_OUTOFMEMORY;
   p->iface.lpVtbl = &presenter_vtbl;
   p->refcount = 1;
   simple_mtx_init(&p->lock, mtx_plain);
   p->queue = f->queue;
   ID3D12CommandQueue_AddRef(p->queue);
   p->desc = *desc;
   p->frame_latency = 1;
   p->color_space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;

   HRESULT hr = ID3D12CommandQueue_GetDevice(p->queue, &IID_ID3D12Device,
                                             (void **)&p->device);
   if (FAILED(hr))
      goto fail;
   p->d12dev = d12_device_from_iface((ID3D12Device2 *)p->device);

   if (surface_factory) {
      p->surface_factory = surface_factory;
      surface_factory->lpVtbl->AddRef(surface_factory);
      p->vk_ready = d12_wine_vk_init(&p->vk, surface_factory);
      if (!p->vk_ready) {
         mesa_loge("wine present: Vulkan presenter unavailable; frames will "
                   "be rendered but not shown");
      }
   }

   hr = presenter_init_readback(p);
   if (FAILED(hr))
      goto fail;
   hr = presenter_create_images(p);
   if (FAILED(hr))
      goto fail;

   if (desc->Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)
      p->latency_sem = CreateSemaphoreW(NULL, p->frame_latency,
                                        DXGI_MAX_SWAP_CHAIN_BUFFERS, NULL);

   *out = &p->iface;
   return S_OK;

fail:
   presenter_Release(&p->iface);
   return hr;
}

static const IDXGIVkSwapChainFactoryVtbl factory_vtbl = {
   factory_QueryInterface,
   factory_AddRef,
   factory_Release,
   factory_CreateSwapChain,
};

bool
d12_wine_queue_query_interface(ID3D12CommandQueue *queue, REFIID riid,
                               void **out)
{
   if (!d12_guid_equal(riid, &D12_IID_IDXGIVkSwapChainFactory))
      return false;
   struct d12_wine_swapchain_factory *f = calloc(1, sizeof(*f));
   if (!f) {
      *out = NULL;
      return true;
   }
   f->iface.lpVtbl = &factory_vtbl;
   f->refcount = 1;
   f->queue = queue;
   ID3D12CommandQueue_AddRef(queue);
   *out = &f->iface;
   return true;
}
