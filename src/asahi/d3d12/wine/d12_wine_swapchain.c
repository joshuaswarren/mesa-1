/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Presentation for the Wine build. DXVK's dxgi.dll owns the swap chain object
 * and the window; it asks our ID3D12CommandQueue for IDXGIVkSwapChainFactory
 * and hands us an IDXGIVkSurfaceFactory that creates a VkSurfaceKHR for the
 * HWND. Vulkan owns uncompressed shared render-target images. Their exported
 * Win32 handles are opened through Wine's D3DKMT bridge and imported by Asahi
 * as dma-bufs, so D3D12 and Vulkan address the same image storage. Present
 * samples that GPU image into the WSI image; it never maps or copies pixels.
 * The image layout is the existing shared Asahi AIL layout in both drivers.
 */

#include "d12_private.h"
#define VK_USE_PLATFORM_WIN32_KHR
#include "d12_wine_dxvk_interfaces.h"
#include "agx_wine_unixlib.h"
#include "util/os_time.h"
#include "util/u_debug.h"
#include "present_opaque_spv.h"


#include <stdio.h>
#include <errno.h>
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
   VkSemaphore acquire_sem;
   VkSemaphore render_sem[D12_WINE_MAX_IMAGES];
   VkImage src_images[D12_WINE_MAX_IMAGES];
   VkImageView src_views[D12_WINE_MAX_IMAGES];
   VkSampler source_sampler;
   VkDeviceMemory src_mem[D12_WINE_MAX_IMAGES];
   VkFormat src_format;
   uint32_t src_w, src_h;
   bool out_of_date;
   bool opaque;
   uint64_t present_id;
   VkDescriptorSetLayout opaque_set_layout;
   VkPipelineLayout opaque_layout;
   VkPipeline opaque_pipeline;
   VkDescriptorPool opaque_pool;
   VkImageView swap_views[D12_WINE_MAX_IMAGES];
   VkDescriptorSet opaque_sets[D12_WINE_MAX_IMAGES];

#define VKFN(name) PFN_vk##name name;
   VKFN(CreateInstance) VKFN(DestroyInstance) VKFN(EnumeratePhysicalDevices)
   VKFN(GetPhysicalDeviceQueueFamilyProperties) VKFN(CreateDevice)
   VKFN(DestroyDevice) VKFN(GetDeviceQueue) VKFN(DestroySurfaceKHR)
   VKFN(GetPhysicalDeviceSurfaceSupportKHR)
   VKFN(GetPhysicalDeviceSurfaceCapabilitiesKHR)
   VKFN(GetPhysicalDeviceSurfaceFormatsKHR)
   VKFN(GetPhysicalDeviceSurfacePresentModesKHR)
   VKFN(GetPhysicalDeviceMemoryProperties) VKFN(GetPhysicalDeviceProperties)
   VKFN(GetPhysicalDeviceProperties2)
   VKFN(GetPhysicalDeviceFeatures)
   VKFN(CreateSwapchainKHR) VKFN(DestroySwapchainKHR) VKFN(GetSwapchainImagesKHR)
   VKFN(AcquireNextImageKHR) VKFN(QueuePresentKHR) VKFN(CreateCommandPool)
   VKFN(DestroyCommandPool) VKFN(AllocateCommandBuffers) VKFN(BeginCommandBuffer)
   VKFN(EndCommandBuffer) VKFN(ResetCommandBuffer) VKFN(QueueSubmit)
   VKFN(CreateFence) VKFN(DestroyFence) VKFN(WaitForFences) VKFN(ResetFences)
   VKFN(CreateSemaphore) VKFN(DestroySemaphore)
   VKFN(CreateImage) VKFN(DestroyImage) VKFN(GetImageMemoryRequirements)
   VKFN(GetMemoryWin32HandleKHR)
   VKFN(AllocateMemory) VKFN(FreeMemory) VKFN(BindImageMemory)
   VKFN(CmdPipelineBarrier) VKFN(CmdBlitImage)
   VKFN(DeviceWaitIdle) VKFN(QueueWaitIdle)
   VKFN(CreateImageView) VKFN(DestroyImageView) VKFN(CreateSampler) VKFN(DestroySampler)
   VKFN(CreateShaderModule) VKFN(DestroyShaderModule)
   VKFN(CreateDescriptorSetLayout) VKFN(DestroyDescriptorSetLayout)
   VKFN(CreatePipelineLayout) VKFN(DestroyPipelineLayout)
   VKFN(CreateComputePipelines) VKFN(DestroyPipeline)
   VKFN(CreateDescriptorPool) VKFN(DestroyDescriptorPool)
   VKFN(AllocateDescriptorSets) VKFN(UpdateDescriptorSets)
   VKFN(CmdBindPipeline) VKFN(CmdBindDescriptorSets) VKFN(CmdDispatch)
   VKFN(GetPhysicalDeviceFeatures2) VKFN(WaitForPresentKHR)
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
   LOAD_I(GetPhysicalDeviceProperties2);
   LOAD_I(GetPhysicalDeviceFeatures);
   LOAD_I(GetPhysicalDeviceFeatures2);
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
   LOAD_D(CreateSemaphore); LOAD_D(DestroySemaphore);
   LOAD_D(CreateImage); LOAD_D(DestroyImage); LOAD_D(GetImageMemoryRequirements);
   LOAD_D(GetMemoryWin32HandleKHR);
   LOAD_D(AllocateMemory); LOAD_D(FreeMemory); LOAD_D(BindImageMemory);
   LOAD_D(CmdPipelineBarrier); LOAD_D(CmdBlitImage);
   LOAD_D(DeviceWaitIdle); LOAD_D(QueueWaitIdle);
   LOAD_D(CreateImageView); LOAD_D(DestroyImageView); LOAD_D(CreateSampler); LOAD_D(DestroySampler);
   LOAD_D(CreateShaderModule); LOAD_D(DestroyShaderModule);
   LOAD_D(CreateDescriptorSetLayout); LOAD_D(DestroyDescriptorSetLayout);
   LOAD_D(CreatePipelineLayout); LOAD_D(DestroyPipelineLayout);
   LOAD_D(CreateComputePipelines); LOAD_D(DestroyPipeline);
   LOAD_D(CreateDescriptorPool); LOAD_D(DestroyDescriptorPool);
   LOAD_D(AllocateDescriptorSets); LOAD_D(UpdateDescriptorSets);
   LOAD_D(CmdBindPipeline); LOAD_D(CmdBindDescriptorSets); LOAD_D(CmdDispatch);
   LOAD_D(WaitForPresentKHR);
#undef LOAD_D
   return vk->CreateSwapchainKHR && vk->CmdBlitImage && vk->QueuePresentKHR &&
          vk->GetMemoryWin32HandleKHR && vk->WaitForPresentKHR;
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
      for (unsigned i = 0; i < vk->swap_image_count; i++) {
         if (vk->swap_views[i])
            vk->DestroyImageView(vk->device, vk->swap_views[i], NULL);
         vk->swap_views[i] = VK_NULL_HANDLE;
      }
      if (vk->opaque_pool)
         vk->DestroyDescriptorPool(vk->device, vk->opaque_pool, NULL);
      vk->opaque_pool = VK_NULL_HANDLE;
      vk->DestroySwapchainKHR(vk->device, vk->swapchain, NULL);
      vk->swapchain = VK_NULL_HANDLE;
   }
   for (unsigned i = 0; i < D12_WINE_MAX_IMAGES; i++) {
      if (vk->render_sem[i])
         vk->DestroySemaphore(vk->device, vk->render_sem[i], NULL);
      vk->render_sem[i] = VK_NULL_HANDLE;
   }
   vk->swap_image_count = 0;
}

static bool
d12_wine_vk_init_opaque(struct d12_wine_vk *vk)
{
   VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   VkSamplerCreateInfo sampler = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   if (vk->CreateSampler(vk->device, &sampler, NULL, &vk->source_sampler))
      return false;
   VkDescriptorSetLayoutCreateInfo set = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = ARRAY_SIZE(bindings), .pBindings = bindings,
   };
   if (vk->CreateDescriptorSetLayout(vk->device, &set, NULL, &vk->opaque_set_layout))
      return false;
   VkPipelineLayoutCreateInfo layout = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &vk->opaque_set_layout,
   };
   if (vk->CreatePipelineLayout(vk->device, &layout, NULL, &vk->opaque_layout))
      return false;
   VkShaderModuleCreateInfo code = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(d12_present_opaque_spv), .pCode = d12_present_opaque_spv,
   };
   VkShaderModule module;
   if (vk->CreateShaderModule(vk->device, &code, NULL, &module))
      return false;
   VkComputePipelineCreateInfo pipeline = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main",
      },
      .layout = vk->opaque_layout,
   };
   VkResult result = vk->CreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
                                               &pipeline, NULL, &vk->opaque_pipeline);
   vk->DestroyShaderModule(vk->device, module, NULL);
   return result == VK_SUCCESS;
}

static bool
d12_wine_vk_create_opaque_views(struct d12_wine_vk *vk)
{
   VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, vk->swap_image_count},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, vk->swap_image_count},
   };
   VkDescriptorPoolCreateInfo pool = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = vk->swap_image_count, .poolSizeCount = ARRAY_SIZE(sizes), .pPoolSizes = sizes,
   };
   if (vk->CreateDescriptorPool(vk->device, &pool, NULL, &vk->opaque_pool))
      return false;
   for (unsigned i = 0; i < vk->swap_image_count; i++) {
      VkImageViewCreateInfo view = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = vk->swap_images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = vk->swap_format,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      if (vk->CreateImageView(vk->device, &view, NULL, &vk->swap_views[i]))
         return false;
      VkDescriptorSetAllocateInfo alloc = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = vk->opaque_pool, .descriptorSetCount = 1,
         .pSetLayouts = &vk->opaque_set_layout,
      };
      if (vk->AllocateDescriptorSets(vk->device, &alloc, &vk->opaque_sets[i]))
         return false;
      VkDescriptorImageInfo image = {
         .imageView = vk->swap_views[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      VkWriteDescriptorSet write = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = vk->opaque_sets[i], .dstBinding = 0,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &image,
      };
      vk->UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);
   }
   return true;
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

   VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      (vk->opaque ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
   if ((caps.supportedUsageFlags & usage) != usage)
      return false;
   extent.width = CLAMP(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
   extent.height = CLAMP(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);

   VkSwapchainCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = vk->surface,
      .minImageCount = min_images,
      .imageFormat = chosen.format,
      .imageColorSpace = chosen.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = usage,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = caps.currentTransform,
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
      d12_wine_vk_destroy_swapchain(vk);
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
   if (vk->GetSwapchainImagesKHR(vk->device, sc, &vk->swap_image_count,
                                  vk->swap_images) != VK_SUCCESS)
      return false;
   VkSemaphoreCreateInfo sinfo = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   for (unsigned i = 0; i < vk->swap_image_count; i++) {
      if (vk->CreateSemaphore(vk->device, &sinfo, NULL, &vk->render_sem[i]))
         return false;
   }
   if (vk->opaque && !d12_wine_vk_create_opaque_views(vk))
      return false;
   vk->out_of_date = false;
   return true;
}

static void
d12_wine_vk_destroy_source(struct d12_wine_vk *vk)
{
   if (!vk->device)
      return;
   vk->DeviceWaitIdle(vk->device);
   for (unsigned i = 0; i < D12_WINE_MAX_IMAGES; i++) {
      if (vk->src_views[i])
         vk->DestroyImageView(vk->device, vk->src_views[i], NULL);
      if (vk->src_images[i])
         vk->DestroyImage(vk->device, vk->src_images[i], NULL);
      if (vk->src_mem[i])
         vk->FreeMemory(vk->device, vk->src_mem[i], NULL);
      vk->src_images[i] = VK_NULL_HANDLE;
      vk->src_views[i] = VK_NULL_HANDLE;
      vk->src_mem[i] = VK_NULL_HANDLE;
   }
}

static bool
d12_wine_vk_create_source(struct d12_wine_vk *vk, struct d12_resource *res,
                          unsigned index)
{
   vk->src_w = res->desc.Width;
   vk->src_h = res->desc.Height;
   vk->src_format = d12_wine_vk_format(res->desc.Format);

   /* STORAGE prevents compression: both drivers use the same renderable,
    * writable AIL_TILING_GPU layout. No opaque optimal layout is guessed
    * across vendors: reject any Vulkan physical device other than Asahi. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
   };
   VkImageCreateInfo iinfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = vk->src_format,
      .extent = {vk->src_w, vk->src_h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult result = vk->CreateImage(vk->device, &iinfo, NULL, &vk->src_images[index]);
   if (result) {
      mesa_loge("wine present: shared vkCreateImage failed (%d)", result);
      return false;
   }
   VkMemoryRequirements req;
   vk->GetImageMemoryRequirements(vk->device, vk->src_images[index], &req);
   VkExportMemoryAllocateInfo export = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
   };
   VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = &export,
      .image = vk->src_images[index],
   };
   VkMemoryAllocateInfo ainfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicated,
      .allocationSize = req.size,
      .memoryTypeIndex = d12_wine_vk_memory_type(vk, req.memoryTypeBits, 0),
   };
   if (ainfo.memoryTypeIndex == UINT32_MAX || res->layout.compressed ||
       res->layout.tiling != AIL_TILING_GPU || res->layout.size_B > req.size) {
      mesa_loge("wine present: incompatible shared layout (tiling=%u compressed=%u D3D=%llu Vulkan=%llu types=0x%x)",
                res->layout.tiling, res->layout.compressed,
                (unsigned long long)res->layout.size_B, (unsigned long long)req.size,
                req.memoryTypeBits);
      return false;
   }
   result = vk->AllocateMemory(vk->device, &ainfo, NULL, &vk->src_mem[index]);
   if (result) {
      mesa_loge("wine present: shared vkAllocateMemory failed (%d)", result);
      return false;
   }
   result = vk->BindImageMemory(vk->device, vk->src_images[index], vk->src_mem[index], 0);
   if (result) {
      mesa_loge("wine present: shared vkBindImageMemory failed (%d)", result);
      return false;
   }

   VkImageViewCreateInfo view = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = vk->src_images[index], .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = vk->src_format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   if (vk->CreateImageView(vk->device, &view, NULL, &vk->src_views[index]))
      return false;
   VkMemoryGetWin32HandleInfoKHR hinfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
      .memory = vk->src_mem[index],
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
   };
   HANDLE handle = NULL;
   result = vk->GetMemoryWin32HandleKHR(vk->device, &hinfo, &handle);
   if (result) {
      mesa_loge("wine present: vkGetMemoryWin32HandleKHR failed (%d)", result);
      return false;
   }
   int fd = d12_wine_resource_fd((uintptr_t)handle);
   CloseHandle(handle);
   if (fd < 0) {
      mesa_loge("wine present: shared Win32 handle to dma-buf failed (errno=%d)", errno);
      return false;
   }
   struct agx_bo *bo = agx_bo_import(&res->device->dev, fd);
   d12_wine_close_fd(fd);
   if (!bo) {
      mesa_loge("wine present: Asahi dma-buf import failed");
      return false;
   }
   if (bo->size < req.size) {
      agx_bo_unreference(&res->device->dev, bo);
      return false;
   }
   agx_bo_unreference(&res->device->dev, res->bo);
   res->bo = bo;
   res->addr = bo->va->addr;
   res->offset_B = 0;

   /* Establish GENERAL and external ownership before publishing the image
    * to D3D12. Later acquire barriers preserve externally rendered pixels. */
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (vk->ResetCommandBuffer(vk->cmd, 0) ||
       vk->BeginCommandBuffer(vk->cmd, &begin))
      return false;
   VkImageMemoryBarrier release = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = vk->queue_family,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
      .image = vk->src_images[index],
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vk->CmdPipelineBarrier(vk->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                          0, NULL, 0, NULL, 1, &release);
   if (vk->EndCommandBuffer(vk->cmd) || vk->ResetFences(vk->device, 1, &vk->fence))
      return false;
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &vk->cmd,
   };
   if (vk->QueueSubmit(vk->queue, 1, &submit, vk->fence) ||
       vk->WaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX))
      return false;
   mesa_logi("wine present: shared GPU image %u, %ux%u, size=%llu", index,
             vk->src_w, vk->src_h, (unsigned long long)bo->size);
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
      if (vk->fence)
         vk->DestroyFence(vk->device, vk->fence, NULL);
      if (vk->pool)
         vk->DestroyCommandPool(vk->device, vk->pool, NULL);
      if (vk->source_sampler)
         vk->DestroySampler(vk->device, vk->source_sampler, NULL);
      if (vk->opaque_pipeline)
         vk->DestroyPipeline(vk->device, vk->opaque_pipeline, NULL);
      if (vk->opaque_layout)
         vk->DestroyPipelineLayout(vk->device, vk->opaque_layout, NULL);
      if (vk->opaque_set_layout)
         vk->DestroyDescriptorSetLayout(vk->device, vk->opaque_set_layout, NULL);
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
d12_wine_vk_init(struct d12_wine_vk *vk, IDXGIVkSurfaceFactory *surface_factory,
                  struct d12_device *d12dev)
{
   memset(vk, 0, sizeof(*vk));
   if (!d12_wine_vk_load(vk))
      goto fail;

   const char *inst_exts[] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "d3d12agx-present",
      .apiVersion = VK_API_VERSION_1_2,
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

   VkPhysicalDeviceIDProperties ids = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
   };
   VkPhysicalDeviceDriverProperties driver = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      .pNext = &ids,
   };
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &driver,
   };
   vk->GetPhysicalDeviceProperties2(vk->phys, &props);
   uint8_t expected_uuid[VK_UUID_SIZE];
   agx_get_device_uuid(&d12dev->dev, expected_uuid);
   if (driver.driverID != VK_DRIVER_ID_MESA_HONEYKRISP ||
       memcmp(ids.deviceUUID, expected_uuid, sizeof(expected_uuid))) {
      mesa_loge("wine present: shared AIL images require Honeykrisp on the D3D12 GPU");
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
   const char *dev_exts[] = {"VK_KHR_swapchain", "VK_KHR_external_memory_win32",
                            "VK_KHR_present_id", "VK_KHR_present_wait"};
   VkPhysicalDevicePresentWaitFeaturesKHR wait_feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
   };
   VkPhysicalDevicePresentIdFeaturesKHR id_feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
      .pNext = &wait_feature,
   };
   VkPhysicalDeviceFeatures2 supported = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &id_feature,
   };
   vk->GetPhysicalDeviceFeatures2(vk->phys, &supported);
   if (!id_feature.presentId || !wait_feature.presentWait) {
      mesa_loge("wine present: FIFO pacing requires present-id/wait support");
      goto fail;
   }
   VkPhysicalDeviceFeatures features;
   vk->GetPhysicalDeviceFeatures(vk->phys, &features);
   if (!features.shaderStorageImageWriteWithoutFormat) {
      mesa_loge("wine present: opaque image storage features unavailable");
      goto fail;
   }
   VkPhysicalDeviceFeatures enabled = {
      .shaderStorageImageWriteWithoutFormat = VK_TRUE,
   };
   VkDeviceCreateInfo dinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &id_feature,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qinfo,
      .enabledExtensionCount = ARRAY_SIZE(dev_exts),
      .ppEnabledExtensionNames = dev_exts,
      .pEnabledFeatures = &enabled,
   };
   if (vk->CreateDevice(vk->phys, &dinfo, NULL, &vk->device)) {
      mesa_loge("wine present: vkCreateDevice failed");
      goto fail;
   }
   if (!d12_wine_vk_load_device_fns(vk))
      goto fail;
   vk->GetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);
   if (!d12_wine_vk_init_opaque(vk))
      goto fail;

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
   if (vk->CreateSemaphore(vk->device, &sinfo, NULL, &vk->acquire_sem))
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

/* The D3D12 queue has released the shared source after rendering. */
static HRESULT
d12_wine_vk_present(struct d12_wine_vk *vk, unsigned source, bool vsync)
{
   if (!vk->swapchain || vk->out_of_date || vk->vsync != vsync) {
      if (!d12_wine_vk_create_swapchain(vk, vk->src_format, vsync))
         return DXGI_ERROR_DEVICE_RESET;
   }

   for (int attempt = 0; attempt < 2; attempt++) {
      if (vk->WaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX))
         return DXGI_ERROR_DEVICE_RESET;

      uint32_t idx = 0;
      VkResult res = vk->AcquireNextImageKHR(vk->device, vk->swapchain, UINT64_MAX,
                                             vk->acquire_sem, VK_NULL_HANDLE, &idx);
      if (res == VK_ERROR_OUT_OF_DATE_KHR) {
         if (!d12_wine_vk_create_swapchain(vk, vk->src_format, vsync))
            return DXGI_ERROR_DEVICE_RESET;
         continue;
      }
      if (res == VK_SUBOPTIMAL_KHR)
         vk->out_of_date = true;
      else if (res != VK_SUCCESS)
         return DXGI_ERROR_DEVICE_RESET;

      vk->ResetFences(vk->device, 1, &vk->fence);
      vk->ResetCommandBuffer(vk->cmd, 0);
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      vk->BeginCommandBuffer(vk->cmd, &begin);

      /* Imported pixels must be preserved: UNDEFINED would discard them.
       * AIL has no layout-dependent representation for this uncompressed image.
       * The external queue ownership barrier makes its device writes visible. */
      VkImageMemoryBarrier acquire = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = vk->opaque ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = vk->opaque ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
         .dstQueueFamilyIndex = vk->queue_family,
         .image = vk->src_images[source],
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      vk->CmdPipelineBarrier(vk->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                             1, &acquire);
      d12_wine_vk_image_barrier(vk, vk->swap_images[idx],
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                vk->opaque ? VK_IMAGE_LAYOUT_GENERAL
                                           : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                                vk->opaque ? VK_ACCESS_SHADER_WRITE_BIT
                                           : VK_ACCESS_TRANSFER_WRITE_BIT);
      if (vk->opaque) {
         VkDescriptorImageInfo src = {
            .sampler = vk->source_sampler,
            .imageView = vk->src_views[source],
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         };
         VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk->opaque_sets[idx], .dstBinding = 1,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &src,
         };
         vk->UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);
         vk->CmdBindPipeline(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->opaque_pipeline);
         vk->CmdBindDescriptorSets(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   vk->opaque_layout, 0, 1, &vk->opaque_sets[idx], 0, NULL);
         vk->CmdDispatch(vk->cmd, DIV_ROUND_UP(vk->swap_extent.width, 8),
                          DIV_ROUND_UP(vk->swap_extent.height, 8), 1);
      } else {
         VkImageBlit blit = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {(int32_t)vk->src_w, (int32_t)vk->src_h, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0},
                           {(int32_t)vk->swap_extent.width,
                            (int32_t)vk->swap_extent.height, 1}},
         };
         vk->CmdBlitImage(vk->cmd, vk->src_images[source], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          vk->swap_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &blit, VK_FILTER_LINEAR);
      }
      d12_wine_vk_image_barrier(vk, vk->swap_images[idx],
                                vk->opaque ? VK_IMAGE_LAYOUT_GENERAL
                                           : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                vk->opaque ? VK_ACCESS_SHADER_WRITE_BIT
                                           : VK_ACCESS_TRANSFER_WRITE_BIT, 0);
      VkImageMemoryBarrier release = acquire;
      release.srcAccessMask = acquire.dstAccessMask;
      release.dstAccessMask = 0;
      release.oldLayout = acquire.newLayout;
      release.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      release.srcQueueFamilyIndex = vk->queue_family;
      release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
      vk->CmdPipelineBarrier(vk->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                             0, NULL, 0, NULL, 1, &release);
      vk->EndCommandBuffer(vk->cmd);

      VkPipelineStageFlags wait_stage = vk->opaque ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                    : VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &vk->acquire_sem,
         .pWaitDstStageMask = &wait_stage,
         .commandBufferCount = 1,
         .pCommandBuffers = &vk->cmd,
         .signalSemaphoreCount = 1,
         .pSignalSemaphores = &vk->render_sem[idx],
      };
      if (vk->QueueSubmit(vk->queue, 1, &submit, vk->fence))
         return DXGI_ERROR_DEVICE_RESET;

      uint64_t id = ++vk->present_id;
      VkPresentIdKHR present_id = {
         .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
         .swapchainCount = 1, .pPresentIds = &id,
      };
      VkPresentInfoKHR present = {
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
         .pNext = &present_id,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &vk->render_sem[idx],
         .swapchainCount = 1,
         .pSwapchains = &vk->swapchain,
         .pImageIndices = &idx,
      };
      res = vk->QueuePresentKHR(vk->queue, &present);
      if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
         vk->out_of_date = true;
      else if (res != VK_SUCCESS)
         return DXGI_ERROR_DEVICE_RESET;
      if (vk->WaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX))
         return DXGI_ERROR_DEVICE_RESET;
      if (vsync) {
         res = vk->WaitForPresentKHR(vk->device, vk->swapchain, id, 2000000000ull);
         if (res == VK_ERROR_OUT_OF_DATE_KHR)
            vk->out_of_date = true;
         else if (res == VK_TIMEOUT)
            return DXGI_STATUS_OCCLUDED;
         else if (res != VK_SUCCESS)
            return DXGI_ERROR_DEVICE_RESET;
      }
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

   /* Order the shared-image handoff after the application queue. */
   ID3D12Fence *fence;
   UINT64 fence_value;
   HANDLE fence_event;

   struct d12_wine_vk vk;
   bool vk_ready;

   HANDLE latency_sem;
   UINT frame_latency;
   UINT64 present_count;
   UINT64 last_present_qpc;
   uint64_t render_wait_ns, wsi_ns;
   bool profile;
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

}

static HRESULT
presenter_create_images(struct d12_wine_presenter *p)
{
   if (p->desc.BufferCount < 1 || p->desc.BufferCount > D12_WINE_MAX_IMAGES ||
       !p->desc.Width || !p->desc.Height)
      return DXGI_ERROR_INVALID_CALL;
   if (d12_wine_vk_format(p->desc.Format) == VK_FORMAT_UNDEFINED) {
      mesa_loge("wine present: unsupported swapchain format %u", p->desc.Format);
      return DXGI_ERROR_INVALID_CALL;
   }

   ID3D12Resource *created[D12_WINE_MAX_IMAGES] = {0};
   struct d12_wine_vk replacement = p->vk;
   memset(replacement.src_images, 0, sizeof(replacement.src_images));
   memset(replacement.src_mem, 0, sizeof(replacement.src_mem));
   memset(replacement.src_views, 0, sizeof(replacement.src_views));
   HRESULT hr = S_OK;
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
      hr = ID3D12Device_CreateCommittedResource(
         p->device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
         D3D12_RESOURCE_STATE_PRESENT, NULL, &IID_ID3D12Resource,
         (void **)&created[i]);
      if (FAILED(hr)) {
         mesa_loge("wine present: back buffer %u creation failed 0x%08x", i,
                   (unsigned)hr);
         goto fail;
      }
      if (!d12_wine_vk_create_source(&replacement, d12_resource_from_iface(created[i]), i)) {
         mesa_loge("wine present: shared image import failed");
         hr = E_FAIL;
         goto fail;
      }
   }

   /* Commit the replacement only after every image is ready. A failed
    * ResizeBuffers keeps the existing resources and source image handles. */
   d12_wine_vk_destroy_source(&p->vk);
   presenter_destroy_images(p);
   memcpy(p->images, created, sizeof(created));
   memcpy(p->vk.src_images, replacement.src_images, sizeof(replacement.src_images));
   memcpy(p->vk.src_views, replacement.src_views, sizeof(replacement.src_views));
   memcpy(p->vk.src_mem, replacement.src_mem, sizeof(replacement.src_mem));
   p->vk.src_w = replacement.src_w;
   p->vk.src_h = replacement.src_h;
   p->vk.src_format = replacement.src_format;
   p->image_index = 0;
   return S_OK;

fail:
   d12_wine_vk_destroy_source(&replacement);
   for (unsigned i = 0; i < D12_WINE_MAX_IMAGES; i++) {
      if (created[i])
         ID3D12Resource_Release(created[i]);
   }
   return hr;
}

static HRESULT
presenter_init_sync(struct d12_wine_presenter *p)
{
   HRESULT hr = ID3D12Device_CreateFence(p->device, 0, D3D12_FENCE_FLAG_NONE,
                                         &IID_ID3D12Fence, (void **)&p->fence);
   if (FAILED(hr))
      return hr;
   p->fence_event = CreateEventW(NULL, FALSE, FALSE, NULL);
   return p->fence_event ? S_OK : E_FAIL;
}

static HRESULT
presenter_wait_render(struct d12_wine_presenter *p)
{
   UINT64 value = ++p->fence_value;
   HRESULT hr = ID3D12CommandQueue_Signal(p->queue, p->fence, value);
   if (FAILED(hr))
      return hr;
   if (ID3D12Fence_GetCompletedValue(p->fence) < value) {
      hr = ID3D12Fence_SetEventOnCompletion(p->fence, value, p->fence_event);
      if (FAILED(hr))
         return hr;
      if (WaitForSingleObject(p->fence_event, 5000) != WAIT_OBJECT_0)
         return DXGI_ERROR_DEVICE_HUNG;
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

   if (p->profile)
      mesa_logi("wine present profile: frames=%llu render-wait-ns=%llu wsi-ns=%llu",
                (unsigned long long)p->present_count,
                (unsigned long long)p->render_wait_ns,
                (unsigned long long)p->wsi_ns);
   if (p->vk_ready)
      d12_wine_vk_fini(&p->vk);
   presenter_destroy_images(p);
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
   /* A timeout must not free images still referenced by the application GPU. */
   HRESULT hr = presenter_wait_render(p);
   if (FAILED(hr)) {
      simple_mtx_unlock(&p->lock);
      return hr;
   }
   DXGI_SWAP_CHAIN_DESC1 previous = p->desc;
   p->desc = *desc;
   p->vk.opaque = desc->AlphaMode == DXGI_ALPHA_MODE_UNSPECIFIED ||
                  desc->AlphaMode == DXGI_ALPHA_MODE_IGNORE;
   hr = presenter_create_images(p);
   if (FAILED(hr)) {
      p->desc = previous;
      p->vk.opaque = previous.AlphaMode == DXGI_ALPHA_MODE_UNSPECIFIED ||
                     previous.AlphaMode == DXGI_ALPHA_MODE_IGNORE;
   } else {
      p->vk.out_of_date = true;
   }
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
   if (sync_interval > 4)
      return DXGI_ERROR_INVALID_CALL;

   simple_mtx_lock(&p->lock);
   UINT index = p->image_index;
   uint64_t start = p->profile ? os_time_get_nano() : 0;
   HRESULT hr = presenter_wait_render(p);
   uint64_t ready = p->profile ? os_time_get_nano() : 0;
   /* FIFO + present completion displays the same GPU image for N refreshes.
    * The application back buffer is advanced once per DXGI Present, even
    * when SyncInterval requests more than one native presentation. */
   for (unsigned refresh = 0; SUCCEEDED(hr) && refresh < MAX2(sync_interval, 1); refresh++) {
      hr = d12_wine_vk_present(&p->vk, index, sync_interval != 0);
      if (hr == DXGI_STATUS_OCCLUDED)
         break;
   }
   if (p->profile) {
      p->render_wait_ns += ready - start;
      p->wsi_ns += os_time_get_nano() - ready;
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
   p->profile = debug_get_bool_option("D3D12AGX_PRESENT_PROFILE", false);
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
      p->vk_ready = d12_wine_vk_init(&p->vk, surface_factory, p->d12dev);
      if (!p->vk_ready) {
         hr = DXGI_ERROR_UNSUPPORTED;
         goto fail;
      }
   }

   if (!p->vk_ready) {
      hr = DXGI_ERROR_UNSUPPORTED;
      goto fail;
   }
   hr = presenter_init_sync(p);
   if (FAILED(hr))
      goto fail;
   p->vk.opaque = desc->AlphaMode == DXGI_ALPHA_MODE_UNSPECIFIED ||
                  desc->AlphaMode == DXGI_ALPHA_MODE_IGNORE;
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
