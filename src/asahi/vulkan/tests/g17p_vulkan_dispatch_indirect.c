/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Record direct -> indirect transitions and inspect the real G17P launch
 * programs. The direct record must run the geometry helper, followed by a
 * compact indirect application record consuming its six-word geometry.
 *
 * The sequential variant reproduces the CTS atomic counter: DispatchBase
 * (2,3,4; 1,1,1) then DispatchIndirect (3,3,1), with only the final shader-to-host
 * barrier. --execute checks counter 10 and guards; --submit-only validates
 * submission under the shim. --cpu-reference runs the same API workload on
 * a CPU Vulkan implementation, without claiming G17P execution.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g17p_vulkan_compute_spv.h"

#define T8140_DEVICE_ID 0x8140
#define ELEMENT_COUNT   32

/* 44-byte direct geometry helper + 28-byte indirect application record. */
#define HELPER_RECORD_SIZE 44u
#define MAIN_RECORD_SIZE   28u
#define PAIR_SIZE          (MAIN_RECORD_SIZE + HELPER_RECORD_SIZE)

/* CDM mode 2 reads global and local sizes from the geometry address. */
#define INDIRECT_MODE 2u
#define RECORD_TAIL_WORD      UINT32_C(0x60000160)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateComputePipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdDispatch);
VK_FN(vkCmdDispatchBase);
VK_FN(vkCmdDispatchIndirect);
VK_FN(vkEndCommandBuffer);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static const char *const NAME = "g17p-vulkan-dispatch-indirect";

static void
die(const char *what)
{
   fprintf(stderr, "%s: %s\n", NAME, what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "%s: %s returned %d\n", NAME, what, result);
      exit(EXIT_FAILURE);
   }
}

static unsigned failures;

#define REQUIRE(cond, ...)                                                     \
   do {                                                                        \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         fprintf(stderr, "%s: FAIL " #cond " -- ", NAME);                      \
         fprintf(stderr, __VA_ARGS__);                                         \
         fprintf(stderr, "\n");                                                \
      }                                                                        \
   } while (0)

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                      \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                    \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                      \
   } while (0)

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

/* One recorded indirect dispatch, as the driver's own launch trace reports it.
 * `pair` is the exact command-stream span the encoder appended. */
struct recorded_indirect {
   uint64_t main_program_va;
   uint64_t args_va;
   uint64_t geometry_va;
   uint64_t helper_program_va;
   uint32_t local[3];
   uint8_t pair[PAIR_SIZE];
   size_t pair_length;
   bool have_launch;
   bool have_pair;
};

static uint32_t
le32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

/*
 * Reverse of the shared native program-pointer encoding: `low = va >> 6` and
 * `control = 0x40000000 | va[47:40]`.  Bits 39:38 of the address are not in
 * this pair -- they live in the record's own word 1 -- so the caller supplies
 * them.  Returning the reconstruction lets the probe check that the pointer
 * the GPU will follow is the pointer the driver said it emitted, rather than
 * trusting that the two 32-bit halves were assembled in the intended order.
 */
static uint64_t
decode_program_pointer(const uint8_t *record, uint32_t word1)
{
   const uint64_t low = (uint64_t)le32(record + 8) << 6;
   const uint64_t high = (uint64_t)(le32(record + 12) & 0xff) << 40;
   const uint64_t mid = (uint64_t)((word1 >> 22) & 0x3ff) << 38;
   return (low & UINT64_C(0x3fffffffff)) | mid | high;
}

/*
 * Parse the driver's launch trace.  Two line shapes matter, both emitted by
 * hk_dispatch_g17p_native():
 *
 *   G17P indirect launch: args=<hex> geometry=<hex> helper_program=<hex> ...
 *   G17P indirect-pair: base=<hex> len=<n> bytes=<hex...>
 *
 * They are emitted adjacently and in that order for each dispatch, so pair
 * them positionally.
 */
static size_t
parse_trace(const char *path, struct recorded_indirect *out, size_t max)
{
   FILE *f = fopen(path, "r");
   if (!f)
      die("cannot reopen the captured launch trace");

   size_t count = 0;
   uint64_t main_program_va = 0;
   char line[8192];
   while (fgets(line, sizeof(line), f)) {
      const char *main_program = strstr(line, "G17P program[main]:");
      if (main_program) {
         unsigned kind;
         if (sscanf(main_program, "G17P program[main]: kind=%u va=%" SCNx64,
                    &kind, &main_program_va) != 2)
            die("main program trace did not parse");
      }
      const char *launch = strstr(line, "G17P indirect launch: ");
      if (launch) {
         if (count == max)
            die("more indirect dispatches were traced than expected");
         struct recorded_indirect *r = &out[count];
         memset(r, 0, sizeof(*r));
         r->main_program_va = main_program_va;
         if (sscanf(launch,
                    "G17P indirect launch: args=%" SCNx64 " geometry=%" SCNx64
                    " helper_program=%" SCNx64 " local=%ux%ux%u",
                    &r->args_va, &r->geometry_va, &r->helper_program_va,
                    &r->local[0], &r->local[1], &r->local[2]) != 6)
            die("the indirect launch trace line did not parse");
         r->have_launch = true;
         count++;
         continue;
      }

      const char *pair = strstr(line, "G17P indirect-pair: ");
      if (!pair)
         continue;
      if (count == 0)
         die("an indirect-pair dump arrived before its launch line");

      struct recorded_indirect *r = &out[count - 1];
      const char *bytes = strstr(pair, "bytes=");
      unsigned long long base;
      unsigned long long len;
      if (!bytes ||
          sscanf(pair, "G17P indirect-pair: base=%llx len=%llu", &base,
                 &len) != 2)
         die("the indirect-pair trace line did not parse");
      bytes += strlen("bytes=");

      r->pair_length = (size_t)len;
      if (r->pair_length != PAIR_SIZE)
         die("the recorded pair is not the witnessed size");
      for (size_t i = 0; i < r->pair_length; ++i) {
         unsigned value;
         if (sscanf(bytes + i * 2, "%2x", &value) != 1)
            die("the indirect-pair hex body is truncated");
         r->pair[i] = (uint8_t)value;
      }
      r->have_pair = true;
   }

   fclose(f);
   return count;
}

static void
verify(const struct recorded_indirect *r, unsigned index,
       uint64_t expected_args, const uint32_t expected_local[3])
{
   REQUIRE(r->have_launch && r->have_pair,
           "dispatch %u produced no complete trace", index);
   if (!r->have_launch || !r->have_pair)
      return;

   REQUIRE(r->args_va == expected_args,
           "dispatch %u args %" PRIx64 " != the bound indirect buffer %" PRIx64,
           index, r->args_va, expected_args);
   for (unsigned axis = 0; axis < 3; ++axis) {
      REQUIRE(r->local[axis] == expected_local[axis],
              "dispatch %u local[%u] is %u, the pipeline declares %u", index,
              axis, r->local[axis], expected_local[axis]);
   }

   const uint8_t *helper = r->pair;
   const uint8_t *main_record = r->pair + HELPER_RECORD_SIZE;

   REQUIRE(((le32(helper) >> 27) & 3) == 0,
           "dispatch %u geometry producer is not a direct launch", index);
   REQUIRE(((le32(main_record) >> 27) & 3) == INDIRECT_MODE,
           "dispatch %u application is not an indirect-local launch", index);
   for (unsigned word = 4; word < 10; ++word)
      REQUIRE(le32(helper + 4 * word) == 1,
              "dispatch %u helper dimensions must be 1x1x1", index);
   REQUIRE(le32(main_record + MAIN_RECORD_SIZE - 4) == RECORD_TAIL_WORD &&
              le32(helper + HELPER_RECORD_SIZE - 4) == RECORD_TAIL_WORD,
           "dispatch %u changed existing barrier framing", index);

   /* The geometry address is stored high word first.  Assembling it the other
    * way round would still look like a plausible 64-bit value, so check it
    * against the address the driver reported allocating. */
   const uint64_t geometry =
      ((uint64_t)le32(main_record + 16) << 32) | le32(main_record + 20);
   REQUIRE(geometry == r->geometry_va,
           "dispatch %u application geometry %" PRIx64 " != allocated %" PRIx64,
           index, geometry, r->geometry_va);
   REQUIRE(geometry != 0 && (geometry & 3) == 0,
           "dispatch %u geometry %" PRIx64 " is not a usable u32 array", index,
           geometry);

   const uint64_t helper_program =
      decode_program_pointer(helper, le32(helper + 4));
   REQUIRE(helper_program == r->helper_program_va,
           "dispatch %u helper program pointer decodes to %" PRIx64
           ", driver emitted %" PRIx64,
           index, helper_program, r->helper_program_va);

   const uint64_t main_program =
      decode_program_pointer(main_record, le32(main_record + 4));
   REQUIRE(main_program == r->main_program_va,
           "dispatch %u compact record does not name the application program", index);
   REQUIRE(main_program != helper_program,
           "dispatch %u main and helper share program %" PRIx64, index,
           main_program);
}

int
main(int argc, char **argv)
{
   bool execute = false, submit_only = false, cpu_reference = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--execute"))
         execute = true;
      else if (!strcmp(argv[i], "--submit-only"))
         execute = submit_only = true;
      else if (!strcmp(argv[i], "--cpu-reference"))
         execute = cpu_reference = true;
      else
         die("usage: probe [--execute|--submit-only|--cpu-reference]");
   }

   /* Turn the launch trace on before the driver caches the option. */
   setenv("HK_G17P_LAUNCH_TRACE", "1", 1);

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("Vulkan loader found no physical device");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties properties;
   for (uint32_t i = 0; i < physical_count; ++i) {
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (cpu_reference ? properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
                        : properties.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   uint32_t queue_family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            NULL);
   VkQueueFamilyProperties *queue_families =
      calloc(queue_family_count, sizeof(*queue_families));
   if (!queue_families)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            queue_families);

   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_family_count; ++i) {
      if (queue_families[i].queueCount != 0) {
         queue_family = i;
         break;
      }
   }
   if (queue_family == UINT32_MAX)
      die("device has no usable queue family");
   free(queue_families);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkDestroyDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateComputePipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdDispatch);
   LOAD_DEVICE(device, vkCmdDispatchBase);
   LOAD_DEVICE(device, vkCmdDispatchIndirect);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   /* Storage buffer the compute shader writes. */
   const VkDeviceSize buffer_size = ELEMENT_COUNT * sizeof(uint32_t);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer;
   check(vkCreateBuffer(device, &buffer_info, NULL, &buffer),
         "vkCreateBuffer");
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &memory_info, NULL, &memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

   /* Two separate indirect argument buffers, so the probe can tell the two
    * recorded dispatches apart by the address each one bound. */
   const VkBufferCreateInfo args_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(uint32_t) * 3,
      .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   /* The first two argument buffers exercise independent recorded launches;
    * args[2] supplies the executable workload for the selected shader. */
   VkBuffer args[3];
   VkDeviceMemory args_memory[3];
   static const uint32_t args_values[3][3] = {{2, 3, 1}, {5, 1, 1}, {1, 1, 1}};
   for (unsigned i = 0; i < 3; ++i) {
      check(vkCreateBuffer(device, &args_info, NULL, &args[i]),
            "vkCreateBuffer(indirect)");
      VkMemoryRequirements args_requirements;
      vkGetBufferMemoryRequirements(device, args[i], &args_requirements);
      const VkMemoryAllocateInfo args_allocate = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = args_requirements.size,
         .memoryTypeIndex = find_memory_type(
            &memory_properties, args_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      check(vkAllocateMemory(device, &args_allocate, NULL, &args_memory[i]),
            "vkAllocateMemory(indirect)");
      check(vkBindBufferMemory(device, args[i], args_memory[i], 0),
            "vkBindBufferMemory(indirect)");
      uint32_t *mapped_args;
      check(vkMapMemory(device, args_memory[i], 0, sizeof(args_values[i]), 0,
                        (void **)&mapped_args),
            "vkMapMemory(indirect)");
      memcpy(mapped_args, args_values[i], sizeof(args_values[i]));
      vkUnmapMemory(device, args_memory[i]);
   }

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");

   const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &set_info, &descriptor_set),
         "vkAllocateDescriptorSets");
   const VkDescriptorBufferInfo descriptor_buffer = {
      .buffer = buffer,
      .offset = 0,
      .range = buffer_size,
   };
   const VkWriteDescriptorSet descriptor_write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor_buffer,
   };
   vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, NULL);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_compute_spv),
      .pCode = g17p_vulkan_compute_spv,
   };
   VkShaderModule shader;
   check(vkCreateShaderModule(device, &shader_info, NULL, &shader),
         "vkCreateShaderModule");
   const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                &pipeline_layout),
         "vkCreatePipelineLayout");
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
         },
      .layout = pipeline_layout,
   };
   VkPipeline pipeline;
   check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                  NULL, &pipeline),
         "vkCreateComputePipelines");

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_buffer_info,
                                  &command_buffer),
         "vkAllocateCommandBuffers");

   /* Capture the driver's launch trace for the recording window only, so the
    * parse cannot pick up device-creation chatter. */
   char trace_path[] = "/tmp/g17p-indirect-trace-XXXXXX";
   const int trace_fd = mkstemp(trace_path);
   if (trace_fd < 0)
      die("cannot create the launch-trace capture file");
   const int saved_stderr = dup(STDERR_FILENO);
   if (saved_stderr < 0)
      die("cannot save stderr");
   fflush(stderr);
   if (dup2(trace_fd, STDERR_FILENO) < 0)
      die("cannot redirect stderr");

   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   /* A direct dispatch first: it must keep recording exactly as before, so a
    * regression in the shared program builder shows up here and not only on
    * hardware. */
   vkCmdDispatchBase(command_buffer, 2, 3, 4, 1, 1, 1);
   vkCmdDispatchIndirect(command_buffer, args[0], 0);
   vkCmdDispatchIndirect(command_buffer, args[1], 0);
   const VkResult end_result = vkEndCommandBuffer(command_buffer);

   fflush(stderr);
   if (dup2(saved_stderr, STDERR_FILENO) < 0)
      die("cannot restore stderr");
   close(saved_stderr);
   close(trace_fd);

   check(begin_result, "vkBeginCommandBuffer");

   /* This is the load-bearing assertion.  Every refusal inside
    * hk_dispatch_g17p_native() sets the command-buffer error, and that error
    * is what vkEndCommandBuffer returns. */
   if (end_result != VK_SUCCESS) {
      fprintf(stderr,
              "%s: FAIL vkEndCommandBuffer returned %d -- the indirect "
              "dispatch was refused, not recorded\n",
              NAME, end_result);
      failures++;
   }

   struct recorded_indirect recorded[4];
   const size_t count =
      parse_trace(trace_path, recorded, sizeof(recorded) / sizeof(*recorded));
   unlink(trace_path);

   if (!cpu_reference)
      REQUIRE(count == 2, "traced %zu indirect dispatches, recorded 2", count);

   /* Match each variant's shader local size. */
   static const uint32_t expected_local[3] = {32, 1, 1};
   for (unsigned i = 0; i < count && i < 2; ++i)
      verify(&recorded[i], i, recorded[i].args_va, expected_local);

   if (count == 2) {
      REQUIRE(recorded[0].args_va != recorded[1].args_va,
              "both dispatches bound the same argument buffer %" PRIx64,
              recorded[0].args_va);
      /* Two dispatches in flight must not publish into one block: the second
       * helper would overwrite the geometry the first main record consumes. */
      REQUIRE(recorded[0].geometry_va != recorded[1].geometry_va,
              "both dispatches share geometry block %" PRIx64,
              recorded[0].geometry_va);
      REQUIRE(memcmp(recorded[0].pair + HELPER_RECORD_SIZE + 16,
                     recorded[1].pair + HELPER_RECORD_SIZE + 16, 8) != 0,
              "both application records name the same geometry address");
   }

   if (execute) {
      /* Submit the executable workload. The shim verifies the stream only;
       * hardware and CPU-reference runs also check results and guard words. */
      uint32_t *mapped;
      check(vkMapMemory(device, memory, 0, buffer_size, 0, (void **)&mapped),
            "vkMapMemory");
      for (uint32_t i = 0; i < ELEMENT_COUNT; ++i)
         mapped[i] = UINT32_C(0xA5A50000) | i;

      VkCommandBuffer execute_cb;
      check(vkAllocateCommandBuffers(device, &command_buffer_info,
                                     &execute_cb),
            "vkAllocateCommandBuffers(execute)");
      check(vkBeginCommandBuffer(execute_cb, &begin_info),
            "vkBeginCommandBuffer(execute)");
      vkCmdBindPipeline(execute_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(execute_cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
      vkCmdDispatchIndirect(execute_cb, args[2], 0);
      const VkBufferMemoryBarrier host_barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = buffer,
         .offset = 0,
         .size = buffer_size,
      };
      vkCmdPipelineBarrier(execute_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                           &host_barrier, 0, NULL);
      check(vkEndCommandBuffer(execute_cb), "vkEndCommandBuffer(execute)");

      VkQueue queue;
      vkGetDeviceQueue(device, queue_family, 0, &queue);
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      VkFence fence;
      check(vkCreateFence(device, &fence_info, NULL, &fence),
            "vkCreateFence");
      const VkSubmitInfo submit_info = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &execute_cb,
      };
      check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, true, UINT64_C(5000000000)),
            "vkWaitForFences");

      if (!submit_only) {
         for (uint32_t i = 0; i < ELEMENT_COUNT; ++i)
            REQUIRE(mapped[i] == UINT32_C(0x12340000) + (2 * i) + 3,
                    "element %u got %08x", i, mapped[i]);
      } else {
         printf("%s: one primary buffer submitted; counter unchecked\n", NAME);
      }

      vkDestroyFence(device, fence, NULL);
      vkUnmapMemory(device, memory);
   }

   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, shader, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   for (unsigned i = 0; i < 3; ++i) {
      vkDestroyBuffer(device, args[i], NULL);
      vkFreeMemory(device, args_memory[i], NULL);
   }
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   if (failures) {
      fprintf(stderr, "%s: FAIL %u checks\n", NAME, failures);
      return EXIT_FAILURE;
   }

   for (size_t i = 0; i < count; ++i) {
      printf("%s: pair[%zu] args=%" PRIx64 " geometry=%" PRIx64
             " helper_program=%" PRIx64 " bytes=",
             NAME, i, recorded[i].args_va, recorded[i].geometry_va,
             recorded[i].helper_program_va);
      for (size_t b = 0; b < recorded[i].pair_length; ++b)
         printf("%02x", recorded[i].pair[b]);
      printf("\n");
   }

   printf("%s: %s PASS\n", NAME, cpu_reference ? "CPU reference"
          : execute ? "submission" : "native recording");
   return EXIT_SUCCESS;
}
