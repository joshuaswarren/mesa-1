/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Execute the present shader on GPU images, then check every channel bit.
 * Test NPOT edges as well as all supported swapchain storage formats.
 */
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../present_opaque_spv.h"
#define VK(expr) do { VkResult r_ = (expr); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d\n", #expr, r_); exit(1); } } while (0)
#define W 31
#define H 17
static VkPhysicalDevice phys;
static VkDevice dev;
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties m;
   vkGetPhysicalDeviceMemoryProperties(phys, &m);
   for (unsigned i = 0; i < m.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags) return i;
   fprintf(stderr, "missing memory type\n"); exit(1);
}
static void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                    VkAccessFlags read, VkAccessFlags write)
{
   VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = read, .dstAccessMask = write, .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,0,NULL,0,NULL,1,&b);
}
int main(void)
{
   VkInstance instance;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VK(vkCreateInstance(&ii,NULL,&instance));
   uint32_t count = 1;
   VK(vkEnumeratePhysicalDevices(instance,&count,&phys));
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(phys,&props);
   printf("renderer: %s\n",props.deviceName);
   uint32_t nf = 16, family = UINT32_MAX;
   VkQueueFamilyProperties fam[16];
   vkGetPhysicalDeviceQueueFamilyProperties(phys,&nf,fam);
   for (unsigned i = 0; i < nf; i++) if (fam[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
   if (family == UINT32_MAX) return 1;
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
   VkPhysicalDeviceFeatures features = {.shaderStorageImageReadWithoutFormat = 1,
                                      .shaderStorageImageWriteWithoutFormat = 1};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .pEnabledFeatures = &features};
   VK(vkCreateDevice(phys,&di,NULL,&dev));
   VkQueue queue; vkGetDeviceQueue(dev,family,0,&queue);
   VkDescriptorSetLayoutBinding bindings[] = {
      {0,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},
      {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},
   };
   VkSamplerCreateInfo sm = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
   VkSampler sampler; VK(vkCreateSampler(dev,&sm,NULL,&sampler));
   VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = bindings};
   VkDescriptorSetLayout set_layout; VK(vkCreateDescriptorSetLayout(dev,&si,NULL,&set_layout));
   VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout};
   VkPipelineLayout layout; VK(vkCreatePipelineLayout(dev,&pi,NULL,&layout));
   VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(d12_present_opaque_spv), .pCode = d12_present_opaque_spv};
   VkShaderModule module; VK(vkCreateShaderModule(dev,&mi,NULL,&module));
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}, .layout = layout};
   VkPipeline pipeline; VK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&ci,NULL,&pipeline));
   VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
   VkCommandPool pool; VK(vkCreateCommandPool(dev,&pool_info,NULL,&pool));
   VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd; VK(vkAllocateCommandBuffers(dev,&ai,&cmd));
   VkFormat formats[] = {VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_B8G8R8A8_UNORM,
                        VK_FORMAT_A2B10G10R10_UNORM_PACK32,VK_FORMAT_R16G16B16A16_SFLOAT};
   for (unsigned f = 0; f < 4; f++) {
      unsigned bpp = f == 3 ? 8 : 4, bytes = W*H*bpp;
      VkImageCreateInfo image_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = formats[f], .extent = {W,H,1}, .mipLevels = 1,
         .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT};
      VkImage image; VK(vkCreateImage(dev,&image_info,NULL,&image));
      VkMemoryRequirements req; vkGetImageMemoryRequirements(dev,image,&req);
      VkMemoryAllocateInfo mem_info = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size, .memoryTypeIndex = memory_type(req.memoryTypeBits,0)};
      VkDeviceMemory mem; VK(vkAllocateMemory(dev,&mem_info,NULL,&mem)); VK(vkBindImageMemory(dev,image,mem,0));
      VkImage source; VK(vkCreateImage(dev,&image_info,NULL,&source));
      VkDeviceMemory source_mem; VK(vkAllocateMemory(dev,&mem_info,NULL,&source_mem)); VK(vkBindImageMemory(dev,source,source_mem,0));
      VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = formats[f],
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
      VkImageView view; VK(vkCreateImageView(dev,&vi,NULL,&view));
      vi.image = source; VkImageView source_view; VK(vkCreateImageView(dev,&vi,NULL,&source_view));
      VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT};
      VkBuffer buffer; VK(vkCreateBuffer(dev,&bi,NULL,&buffer)); vkGetBufferMemoryRequirements(dev,buffer,&req);
      mem_info.allocationSize = req.size;
      mem_info.memoryTypeIndex = memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VkDeviceMemory bm; VK(vkAllocateMemory(dev,&mem_info,NULL,&bm)); VK(vkBindBufferMemory(dev,buffer,bm,0));
      uint8_t *map; VK(vkMapMemory(dev,bm,0,bytes,0,(void **)&map));
      uint8_t expected[W*H*8];
      for (unsigned i = 0; i < W*H; i++) {
         if (f < 2) {
            map[i*4] = (i*17)&255; map[i*4+1] = (i*31)&255; map[i*4+2] = (i*7)&255; map[i*4+3] = i&255;
            memcpy(expected+i*4,map+i*4,4); expected[i*4+3] = 255;
         } else if (f == 2) {
            uint32_t value = (i&1023) | (((i*17)&1023)<<10) | (((i*31)&1023)<<20) | ((i&3)<<30);
            memcpy(map+i*4,&value,4); value |= 0xc0000000u; memcpy(expected+i*4,&value,4);
         } else {
            uint16_t value[] = {0x3400,0x3800,0x3c00,0}; memcpy(map+i*8,value,8);
            value[3] = 0x3c00; memcpy(expected+i*8,value,8);
         }
      }
      VkDescriptorPoolSize ds[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                                   {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1}};
      VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ds};
      VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev,&dpi,NULL,&dp));
      VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
      VkDescriptorSet set; VK(vkAllocateDescriptorSets(dev,&sai,&set));
      VkDescriptorImageInfo img = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &img};
      vkUpdateDescriptorSets(dev,1,&write,0,NULL);
      VkDescriptorImageInfo src = {.sampler = sampler, .imageView = source_view,
                                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      write.dstBinding = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &src;
      vkUpdateDescriptorSets(dev,1,&write,0,NULL);
      VK(vkResetCommandBuffer(cmd,0));
      VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK(vkBeginCommandBuffer(cmd,&begin));
      barrier(cmd,source,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}, .imageExtent = {W,H,1}};
      vkCmdCopyBufferToImage(cmd,buffer,source,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
      barrier(cmd,source,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
      barrier(cmd,image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,0,VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
      vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,NULL);
      vkCmdDispatch(cmd,(W+7)/8,(H+7)/8,1);
      barrier(cmd,image,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
      VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,NULL,0,NULL);
      VK(vkEndCommandBuffer(cmd));
      VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK(vkCreateFence(dev,&fi,NULL,&fence));
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount = 1,.pCommandBuffers = &cmd};
      VK(vkQueueSubmit(queue,1,&submit,fence)); VK(vkWaitForFences(dev,1,&fence,1,5000000000ull));
      if (memcmp(map,expected,bytes)) { fprintf(stderr,"format %u: wrong pixel bits\n",formats[f]); return 1; }
      printf("format %u: %u pixels RGB preserved, alpha opaque\n",formats[f],W*H);
      vkUnmapMemory(dev,bm); vkDestroyFence(dev,fence,NULL); vkDestroyDescriptorPool(dev,dp,NULL);
      vkDestroyBuffer(dev,buffer,NULL); vkFreeMemory(dev,bm,NULL); vkDestroyImageView(dev,view,NULL);
      vkDestroyImage(dev,image,NULL); vkFreeMemory(dev,mem,NULL);
      vkDestroyImageView(dev,source_view,NULL); vkDestroyImage(dev,source,NULL); vkFreeMemory(dev,source_mem,NULL);
   }
   vkDestroyCommandPool(dev,pool,NULL); vkDestroyPipeline(dev,pipeline,NULL); vkDestroyShaderModule(dev,module,NULL);
   vkDestroyPipelineLayout(dev,layout,NULL); vkDestroyDescriptorSetLayout(dev,set_layout,NULL);
   vkDestroySampler(dev,sampler,NULL); vkDestroyDevice(dev,NULL); vkDestroyInstance(instance,NULL);
   puts("PRESENT_OPAQUE_GPU_PASS"); return 0;
}
