/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * v48 VM-status mirror probe for the G17P DRM shim. Every vkWaitForFences
 * and vkGetFenceStatus runs the device-lost check (hk_check_status). The
 * shim reports how many GET_PARAMS VM_STATUS queries that took and whether a
 * mirror was registered; an injected sticky error must surface as
 * VK_ERROR_DEVICE_LOST through either path.
 *
 * usage: g17p-vm-status-mirror <expect-success|expect-lost> [waits]
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T8140_DEVICE_ID 0x8140
#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkWaitForFences);
VK_FN(vkGetFenceStatus);

static void
die(const char *what)
{
   fprintf(stderr, "g17p-vm-status-mirror: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-vm-status-mirror: %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD(scope, name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(scope, #name);                  \
      if (!name)                                                               \
         die("missing " #name);                                               \
   } while (0)

int
main(int argc, char **argv)
{
   if (argc < 2 || (strcmp(argv[1], "expect-success") &&
                    strcmp(argv[1], "expect-lost")))
      die("usage: g17p-vm-status-mirror <expect-success|expect-lost> [waits]");
   const bool expect_lost = !strcmp(argv[1], "expect-lost");
   const unsigned waits = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 64;

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("no vkGetInstanceProcAddr");
   LOAD(VK_NULL_HANDLE, vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-vm-status-mirror",
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance");
   LOAD(instance, vkGetDeviceProcAddr);
   LOAD(instance, vkDestroyInstance);
   LOAD(instance, vkEnumeratePhysicalDevices);
   LOAD(instance, vkGetPhysicalDeviceProperties);
   LOAD(instance, vkCreateDevice);

   VkPhysicalDevice physicals[8];
   uint32_t count = 8;
   check(vkEnumeratePhysicalDevices(instance, &count, physicals),
         "vkEnumeratePhysicalDevices");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(physicals[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID)
         physical = physicals[i];
   }
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device), "vkCreateDevice");
#define LOAD_DEVICE(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("missing " #name);                                               \
   } while (0)
   LOAD_DEVICE(vkDestroyDevice);
   LOAD_DEVICE(vkCreateFence);
   LOAD_DEVICE(vkDestroyFence);
   LOAD_DEVICE(vkWaitForFences);
   LOAD_DEVICE(vkGetFenceStatus);

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");

   unsigned ok = 0, lost = 0, other = 0;
   for (unsigned i = 0; i < waits; ++i) {
      VkResult r = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      ok += r == VK_SUCCESS;
      lost += r == VK_ERROR_DEVICE_LOST;
      other += r != VK_SUCCESS && r != VK_ERROR_DEVICE_LOST;
   }
   VkResult status = vkGetFenceStatus(device, fence);
   printf("g17p-vm-status-mirror: waits=%u success=%u lost=%u other=%u "
          "fence-status=%d\n", waits, ok, lost, other, status);

   vkDestroyFence(device, fence, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);

   if (other)
      return EXIT_FAILURE;
   if (expect_lost)
      return lost == waits && status == VK_ERROR_DEVICE_LOST ? EXIT_SUCCESS
                                                             : EXIT_FAILURE;
   return ok == waits && status == VK_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
