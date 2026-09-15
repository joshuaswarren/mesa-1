/*
 * Prints the device timestamp domain as seen through
 * vkGetCalibratedTimestampsKHR (device + monotonic), a timestamp query
 * written by vkCmdWriteTimestamp, timestampPeriod and timestampValidBits,
 * so the units of the two device clocks can be compared.
 * Build: clang --target=aarch64-linux-gnu ... tools/calibrated-timestamp-probe.c -lvulkan
 * On the M4 run only through gpu-experiment.py and run-m4-gl.py.
 */
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

static uint64_t mono_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000ull + ts.tv_nsec; }

int main(void) {
   const char *ext[] = {"VK_KHR_calibrated_timestamps"};
   VkInstance instance; VkInstanceCreateInfo ic = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vkCreateInstance(&ic, NULL, &instance));
   uint32_t n = 1; VkPhysicalDevice physical; CHECK(vkEnumeratePhysicalDevices(instance, &n, &physical));
   VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(physical, &props);
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, NULL);
   VkQueueFamilyProperties *fam = calloc(n, sizeof(*fam)); vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, fam);
   unsigned family = 0; while (family < n && !(fam[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) family++;
   printf("device %s timestampPeriod=%g timestampValidBits=%u\n", props.deviceName, props.limits.timestampPeriod, fam[family].timestampValidBits);
   float prio = 1; VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext};
   VkDevice dev; CHECK(vkCreateDevice(physical, &di, NULL, &dev)); VkQueue queue; vkGetDeviceQueue(dev, family, 0, &queue);
   PFN_vkGetCalibratedTimestampsKHR getcal = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(dev, "vkGetCalibratedTimestampsKHR");
   PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR getdom = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
   uint32_t nd = 0; getdom(physical, &nd, NULL); VkTimeDomainKHR *doms = calloc(nd, sizeof(*doms)); getdom(physical, &nd, doms);
   printf("time domains:"); for (uint32_t i = 0; i < nd; ++i) printf(" %u", doms[i]); printf("\n");
   VkCalibratedTimestampInfoKHR infos[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, NULL, VK_TIME_DOMAIN_DEVICE_KHR}, {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, NULL, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
   uint64_t ts0[2], ts1[2], dev0[2], dev1[2];
   CHECK(getcal(dev, 2, infos, ts0, dev0));
   uint64_t m0 = mono_ns(); usleep(200000); uint64_t m1 = mono_ns();
   CHECK(getcal(dev, 2, infos, ts1, dev1));
   printf("calibrated device: %llu -> %llu (delta %llu) monotonic: %llu -> %llu (delta %llu) wall delta %llu ns maxdev %llu/%llu\n",
          (unsigned long long)ts0[0], (unsigned long long)ts1[0], (unsigned long long)(ts1[0] - ts0[0]),
          (unsigned long long)ts0[1], (unsigned long long)ts1[1], (unsigned long long)(ts1[1] - ts0[1]),
          (unsigned long long)(m1 - m0), (unsigned long long)dev0[0], (unsigned long long)dev1[0]);
   /* Two GPU timestamps 200 ms apart, compared with the calibrated device clock read in between. */
   VkQueryPoolCreateInfo qp = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
   VkQueryPool pool; CHECK(vkCreateQueryPool(dev, &qp, NULL, &pool));
   VkCommandPool cp; VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
   VkCommandBuffer cmd; VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   CHECK(vkAllocateCommandBuffers(dev, &ca, &cmd));
   uint64_t q[2], cal[2];
   for (unsigned i = 0; i < 2; ++i) {
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &bi));
      if (i == 0) vkCmdResetQueryPool(cmd, pool, 0, 2);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, i);
      CHECK(vkEndCommandBuffer(cmd));
      uint64_t c[2]; CHECK(getcal(dev, 2, infos, c, dev0)); cal[i] = c[0];
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue));
      if (i == 0) usleep(200000);
   }
   CHECK(vkGetQueryPoolResults(dev, pool, 0, 2, sizeof(q), q, sizeof(q[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   printf("query timestamps: %llu %llu (delta %llu); calibrated device before each submit: %llu %llu (delta %llu)\n",
          (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)(q[1] - q[0]),
          (unsigned long long)cal[0], (unsigned long long)cal[1], (unsigned long long)(cal[1] - cal[0]));
   printf("ratio query/calibrated delta = %.6f; query - calibrated offset = %lld\n", (double)(q[1] - q[0]) / (double)(cal[1] - cal[0]), (long long)(q[0] - cal[0]));
   printf("CALIBRATED_PROBE_DONE\n");
   return 0;
}
