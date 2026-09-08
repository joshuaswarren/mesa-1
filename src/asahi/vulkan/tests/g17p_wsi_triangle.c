/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * One-queue XCB/Wayland smoke test for the bounded G17P graphics path. It uses one
 * 24-byte vertex buffer and no descriptors. The render pass has one
 * single-sample colour attachment and no depth or stencil attachment.
 */

#define VK_NO_PROTOTYPES
#ifdef G17P_WSI_WAYLAND
#define VK_USE_PLATFORM_WAYLAND_KHR
#else
#define VK_USE_PLATFORM_XCB_KHR
#endif
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef G17P_WSI_WAYLAND
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#else
#include <xcb/xcb.h>
#endif

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceSurfaceSupportKHR);
VK_FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
VK_FN(vkGetPhysicalDeviceSurfaceFormatsKHR);
#ifdef G17P_WSI_WAYLAND
VK_FN(vkCreateWaylandSurfaceKHR);
#else
VK_FN(vkCreateXcbSurfaceKHR);
#endif
VK_FN(vkCreateDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateSwapchainKHR);
VK_FN(vkGetSwapchainImagesKHR);
VK_FN(vkCreateImageView);
VK_FN(vkCreateRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkCreateFramebuffer);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkCreateBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkMapMemory);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateSemaphore);
VK_FN(vkCreateFence);
VK_FN(vkWaitForFences);
VK_FN(vkResetFences);
VK_FN(vkAcquireNextImageKHR);
VK_FN(vkQueueSubmit);
VK_FN(vkQueuePresentKHR);
VK_FN(vkQueueWaitIdle);

static void
die(const char *what)
{
   fprintf(stderr, "g17p-wsi-triangle: %s\n", what);
   exit(1);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-wsi-triangle: %s returned %d\n", what, result);
      exit(1);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                     \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                   \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                     \
   } while (0)

#ifdef G17P_WSI_WAYLAND
struct wayland_window {
   struct wl_display *display;
   struct wl_compositor *compositor;
   struct xdg_wm_base *wm_base;
   struct wl_surface *surface;
   struct xdg_surface *xdg_surface;
   struct xdg_toplevel *toplevel;
   bool configured;
   bool closed;
};

static void
registry_global(void *data, struct wl_registry *registry, uint32_t name,
                const char *interface, uint32_t version)
{
   struct wayland_window *window = data;
   if (!strcmp(interface, wl_compositor_interface.name)) {
      window->compositor = wl_registry_bind(
         registry, name, &wl_compositor_interface, version < 4 ? version : 4);
   } else if (!strcmp(interface, xdg_wm_base_interface.name)) {
      window->wm_base =
         wl_registry_bind(registry, name, &xdg_wm_base_interface, 1);
   }
}

static void
registry_remove(void *data, struct wl_registry *registry, uint32_t name)
{
   (void)data;
   (void)registry;
   (void)name;
}

static const struct wl_registry_listener registry_listener = {
   .global = registry_global,
   .global_remove = registry_remove,
};

static void
wm_base_ping(void *data, struct xdg_wm_base *wm_base, uint32_t serial)
{
   (void)data;
   xdg_wm_base_pong(wm_base, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
   .ping = wm_base_ping,
};

static void
xdg_surface_configure(void *data, struct xdg_surface *surface,
                      uint32_t serial)
{
   struct wayland_window *window = data;
   xdg_surface_ack_configure(surface, serial);
   window->configured = true;
}

static const struct xdg_surface_listener xdg_surface_listener = {
   .configure = xdg_surface_configure,
};

static void
toplevel_configure(void *data, struct xdg_toplevel *toplevel, int32_t width,
                   int32_t height, struct wl_array *states)
{
   (void)data;
   (void)toplevel;
   (void)width;
   (void)height;
   (void)states;
}

static void
toplevel_close(void *data, struct xdg_toplevel *toplevel)
{
   (void)toplevel;
   ((struct wayland_window *)data)->closed = true;
}

static const struct xdg_toplevel_listener toplevel_listener = {
   .configure = toplevel_configure,
   .close = toplevel_close,
};
#else
static xcb_screen_t *
get_screen(xcb_connection_t *connection, int screen_number)
{
   const xcb_setup_t *setup = xcb_get_setup(connection);
   xcb_screen_iterator_t iter = xcb_setup_roots_iterator(setup);

   while (screen_number-- > 0)
      xcb_screen_next(&iter);

   return iter.data;
}
#endif

static VkSurfaceFormatKHR
choose_surface_format(const VkSurfaceFormatKHR *formats, uint32_t count)
{
   if (count == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
      return (VkSurfaceFormatKHR){
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .colorSpace = formats[0].colorSpace,
      };
   }

   const VkFormat preferred[] = {
      VK_FORMAT_B8G8R8A8_UNORM,
      VK_FORMAT_R8G8B8A8_UNORM,
   };

   for (unsigned p = 0; p < 2; ++p) {
      for (uint32_t i = 0; i < count; ++i) {
         if (formats[i].format == preferred[p])
            return formats[i];
      }
   }

   die("surface has neither BGRA8 nor RGBA8 UNORM");
   return (VkSurfaceFormatKHR){0};
}

static VkCompositeAlphaFlagBitsKHR
choose_composite_alpha(VkCompositeAlphaFlagsKHR supported)
{
   const VkCompositeAlphaFlagBitsKHR choices[] = {
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
   };

   for (unsigned i = 0; i < 4; ++i) {
      if (supported & choices[i])
         return choices[i];
   }

   die("surface has no composite-alpha mode");
   return 0;
}

static uint32_t
clamp_u32(uint32_t value, uint32_t min, uint32_t max)
{
   if (value < min)
      return min;
   if (value > max)
      return max;
   return value;
}

int
main(int argc, char **argv)
{
   unsigned frame_count = 300;
   if (argc > 2)
      die("usage: g17p-wsi-triangle [frame-count]");
   if (argc == 2) {
      char *end = NULL;
      unsigned long value = strtoul(argv[1], &end, 10);
      if (!end || *end || value == 0 || value > 100000)
         die("frame-count must be 1..100000");
      frame_count = value;
   }

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");

   LOAD_GLOBAL(vkCreateInstance);

#ifdef G17P_WSI_WAYLAND
   struct wayland_window window = {0};
   window.display = wl_display_connect(NULL);
   if (!window.display)
      die("cannot connect to WAYLAND_DISPLAY");
   struct wl_registry *registry = wl_display_get_registry(window.display);
   if (!registry)
      die("Wayland registry is missing");
   wl_registry_add_listener(registry, &registry_listener, &window);
   if (wl_display_roundtrip(window.display) < 0 || !window.compositor ||
       !window.wm_base)
      die("Wayland compositor has no wl_compositor/xdg_wm_base");
   xdg_wm_base_add_listener(window.wm_base, &wm_base_listener, &window);
   window.surface = wl_compositor_create_surface(window.compositor);
   window.xdg_surface =
      xdg_wm_base_get_xdg_surface(window.wm_base, window.surface);
   window.toplevel = xdg_surface_get_toplevel(window.xdg_surface);
   if (!window.surface || !window.xdg_surface || !window.toplevel)
      die("cannot create Wayland xdg-toplevel");
   xdg_surface_add_listener(window.xdg_surface, &xdg_surface_listener,
                            &window);
   xdg_toplevel_add_listener(window.toplevel, &toplevel_listener, &window);
   xdg_toplevel_set_title(window.toplevel, "G17P Vulkan RT0 triangle");
   wl_surface_commit(window.surface);
   while (!window.configured) {
      if (wl_display_dispatch(window.display) < 0)
         die("Wayland initial configure failed");
   }
#else
   int screen_number = 0;
   xcb_connection_t *xcb = xcb_connect(NULL, &screen_number);
   if (!xcb || xcb_connection_has_error(xcb))
      die("cannot connect to DISPLAY through XCB");
   xcb_screen_t *screen = get_screen(xcb, screen_number);
   if (!screen)
      die("XCB screen is missing");

   xcb_window_t window = xcb_generate_id(xcb);
   const uint32_t event_mask = XCB_EVENT_MASK_EXPOSURE |
                               XCB_EVENT_MASK_STRUCTURE_NOTIFY;
   xcb_create_window(xcb, XCB_COPY_FROM_PARENT, window, screen->root, 0, 0,
                     640, 480, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                     screen->root_visual, XCB_CW_EVENT_MASK, &event_mask);
   static const char title[] = "G17P Vulkan RT0 triangle";
   xcb_change_property(xcb, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME,
                       XCB_ATOM_STRING, 8, sizeof(title) - 1, title);
   xcb_map_window(xcb, window);
   xcb_flush(xcb);
#endif

   const char *instance_extensions[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef G17P_WSI_WAYLAND
      VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#else
      VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
   };
   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-wsi-triangle",
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = instance_extensions,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceSupportKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceFormatsKHR);
#ifdef G17P_WSI_WAYLAND
   LOAD_INSTANCE(instance, vkCreateWaylandSurfaceKHR);
#else
   LOAD_INSTANCE(instance, vkCreateXcbSurfaceKHR);
#endif
   LOAD_INSTANCE(instance, vkCreateDevice);

#ifdef G17P_WSI_WAYLAND
   const VkWaylandSurfaceCreateInfoKHR surface_info = {
      .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
      .display = window.display,
      .surface = window.surface,
   };
#else
   const VkXcbSurfaceCreateInfoKHR surface_info = {
      .sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
      .connection = xcb,
      .window = window,
   };
#endif
   VkSurfaceKHR surface;
#ifdef G17P_WSI_WAYLAND
   check(vkCreateWaylandSurfaceKHR(instance, &surface_info, NULL, &surface),
         "vkCreateWaylandSurfaceKHR");
#else
   check(vkCreateXcbSurfaceKHR(instance, &surface_info, NULL, &surface),
         "vkCreateXcbSurfaceKHR");
#endif

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (physical_count == 0)
      die("Vulkan loader found no physical device");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         VkBool32 present = false;
         check(vkGetPhysicalDeviceSurfaceSupportKHR(
                  physical_devices[p], q, surface, &present),
               "vkGetPhysicalDeviceSurfaceSupportKHR");
         if ((queues[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            physical = physical_devices[p];
            queue_family = q;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("no single queue family supports graphics and presentation");

   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical, &properties);
   fprintf(stderr, "g17p-wsi-triangle: device %s, api %u.%u.%u, queue %u\n",
           properties.deviceName, VK_API_VERSION_MAJOR(properties.apiVersion),
           VK_API_VERSION_MINOR(properties.apiVersion),
           VK_API_VERSION_PATCH(properties.apiVersion), queue_family);

   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateSwapchainKHR);
   LOAD_DEVICE(device, vkGetSwapchainImagesKHR);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdEndRenderPass);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateSemaphore);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkWaitForFences);
   LOAD_DEVICE(device, vkResetFences);
   LOAD_DEVICE(device, vkAcquireNextImageKHR);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkQueuePresentKHR);
   LOAD_DEVICE(device, vkQueueWaitIdle);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   static const float vertices[][2] = {
      {-0.75f, 0.75f},
      {0.75f, 0.75f},
      {0.0f, -0.75f},
   };
   const VkBufferCreateInfo vertex_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_buffer_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");

   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   uint32_t vertex_memory_type = UINT32_MAX;
   const VkMemoryPropertyFlags required_memory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((vertex_requirements.memoryTypeBits & (1u << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & required_memory) ==
             required_memory) {
         vertex_memory_type = i;
         break;
      }
   }
   if (vertex_memory_type == UINT32_MAX)
      die("vertex buffer has no host-visible coherent memory type");

   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = vertex_memory_type,
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));

   VkSurfaceCapabilitiesKHR capabilities;
   check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface,
                                                    &capabilities),
         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
   if (!(capabilities.supportedUsageFlags &
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      die("surface images cannot be colour attachments");

   uint32_t format_count = 0;
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count,
                                              NULL),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
   if (format_count == 0)
      die("surface has no formats");
   VkSurfaceFormatKHR *formats = calloc(format_count, sizeof(*formats));
   if (!formats)
      die("out of host memory");
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count,
                                              formats),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(list)");
   VkSurfaceFormatKHR surface_format =
      choose_surface_format(formats, format_count);
   free(formats);

   VkExtent2D extent = capabilities.currentExtent;
   if (extent.width == UINT32_MAX) {
      extent.width = clamp_u32(640, capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width);
      extent.height = clamp_u32(480, capabilities.minImageExtent.height,
                                capabilities.maxImageExtent.height);
   }

   uint32_t swapchain_image_count = capabilities.minImageCount + 1;
   if (capabilities.maxImageCount &&
       swapchain_image_count > capabilities.maxImageCount)
      swapchain_image_count = capabilities.maxImageCount;

   const VkSwapchainCreateInfoKHR swapchain_info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = swapchain_image_count,
      .imageFormat = surface_format.format,
      .imageColorSpace = surface_format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = capabilities.currentTransform,
      .compositeAlpha =
         choose_composite_alpha(capabilities.supportedCompositeAlpha),
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = true,
   };
   VkSwapchainKHR swapchain;
   check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &swapchain),
         "vkCreateSwapchainKHR");

   check(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count,
                                 NULL),
         "vkGetSwapchainImagesKHR(count)");
   VkImage *images = calloc(swapchain_image_count, sizeof(*images));
   VkImageView *views = calloc(swapchain_image_count, sizeof(*views));
   VkFramebuffer *framebuffers =
      calloc(swapchain_image_count, sizeof(*framebuffers));
   VkCommandBuffer *command_buffers =
      calloc(swapchain_image_count, sizeof(*command_buffers));
   if (!images || !views || !framebuffers || !command_buffers)
      die("out of host memory");
   check(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count,
                                 images),
         "vkGetSwapchainImagesKHR(list)");

   for (uint32_t i = 0; i < swapchain_image_count; ++i) {
      const VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = images[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = surface_format.format,
         .components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
         },
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
         },
      };
      check(vkCreateImageView(device, &view_info, NULL, &views[i]),
            "vkCreateImageView");
   }

   const VkAttachmentDescription attachment = {
      .format = surface_format.format,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   const VkSubpassDependency dependency = {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = g17p_wsi_triangle_frag_spv,
   };
   VkShaderModule vert_module, frag_module;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(vertices[0]),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = 0,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = extent.width,
      .height = extent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = extent};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = pipeline_layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

   for (uint32_t i = 0; i < swapchain_image_count; ++i) {
      const VkFramebufferCreateInfo framebuffer_info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass,
         .attachmentCount = 1,
         .pAttachments = &views[i],
         .width = extent.width,
         .height = extent.height,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &framebuffer_info, NULL,
                                &framebuffers[i]),
            "vkCreateFramebuffer");
   }

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = swapchain_image_count,
   };
   check(vkAllocateCommandBuffers(device, &command_allocate, command_buffers),
         "vkAllocateCommandBuffers");

   for (uint32_t i = 0; i < swapchain_image_count; ++i) {
      const VkCommandBufferBeginInfo begin_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      };
      check(vkBeginCommandBuffer(command_buffers[i], &begin_info),
            "vkBeginCommandBuffer");
      const VkClearValue clear = {.color = {{0.015f, 0.02f, 0.04f, 1.0f}}};
      const VkRenderPassBeginInfo render_begin = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render_pass,
         .framebuffer = framebuffers[i],
         .renderArea = {.extent = extent},
         .clearValueCount = 1,
         .pClearValues = &clear,
      };
      vkCmdBeginRenderPass(command_buffers[i], &render_begin,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline);
      const VkDeviceSize vertex_offset = 0;
      vkCmdBindVertexBuffers(command_buffers[i], 0, 1, &vertex_buffer,
                             &vertex_offset);
      vkCmdDraw(command_buffers[i], 3, 1, 0, 0);
      vkCmdEndRenderPass(command_buffers[i]);
      check(vkEndCommandBuffer(command_buffers[i]), "vkEndCommandBuffer");
   }

   const VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
   };
   VkSemaphore acquired, finished;
   check(vkCreateSemaphore(device, &semaphore_info, NULL, &acquired),
         "vkCreateSemaphore(acquired)");
   check(vkCreateSemaphore(device, &semaphore_info, NULL, &finished),
         "vkCreateSemaphore(finished)");
   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");

   fprintf(stderr,
           "g17p-wsi-triangle: rendering %u frames, %ux%u, format %u, "
           "one layer, 24-byte VBO, no depth, 1x MSAA\n",
           frame_count, extent.width, extent.height, surface_format.format);

   for (unsigned frame = 0; frame < frame_count; ++frame) {
#ifdef G17P_WSI_WAYLAND
      if (window.closed)
         break;
      if (wl_display_dispatch_pending(window.display) < 0)
         die("Wayland dispatch failed");
#endif
      check(vkWaitForFences(device, 1, &fence, true, UINT64_MAX),
            "vkWaitForFences");
      check(vkResetFences(device, 1, &fence), "vkResetFences");

      uint32_t image_index;
      VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                              acquired, VK_NULL_HANDLE,
                                              &image_index);
      if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
         check(result, "vkAcquireNextImageKHR");

      const VkPipelineStageFlags wait_stage =
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &acquired,
         .pWaitDstStageMask = &wait_stage,
         .commandBufferCount = 1,
         .pCommandBuffers = &command_buffers[image_index],
         .signalSemaphoreCount = 1,
         .pSignalSemaphores = &finished,
      };
      check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");

      const VkPresentInfoKHR present = {
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
         .waitSemaphoreCount = 1,
         .pWaitSemaphores = &finished,
         .swapchainCount = 1,
         .pSwapchains = &swapchain,
         .pImageIndices = &image_index,
      };
      result = vkQueuePresentKHR(queue, &present);
      if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
         check(result, "vkQueuePresentKHR");

      if ((frame + 1) % 60 == 0 || frame + 1 == frame_count)
         fprintf(stderr, "g17p-wsi-triangle: frame %u/%u\n", frame + 1,
                 frame_count);
#ifdef G17P_WSI_WAYLAND
      if (wl_display_flush(window.display) < 0)
         die("Wayland flush failed");
#endif
   }

   check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
   fprintf(stderr, "g17p-wsi-triangle: completed\n");
   return 0;
}
