/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public Vulkan oracle for Wayland surface ownership across failed and
 * successful swapchain replacement.  A minimal in-process compositor exposes
 * wl_compositor and wl_shm, keeping the test hermetic under the G17P DRM shim.
 */

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <threads.h>
#include <unistd.h>
#include <wayland-client.h>
#include <wayland-server-core.h>
#include <wayland-server-protocol.h>

#define TEST_NAME "g17p-vulkan-wayland-ownership"
#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkCreateWaylandSurfaceKHR);
VK_FN(vkDestroySurfaceKHR);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceSurfaceSupportKHR);
VK_FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
VK_FN(vkGetPhysicalDeviceSurfaceFormatsKHR);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateSwapchainKHR);
VK_FN(vkDestroySwapchainKHR);
VK_FN(vkGetSwapchainImagesKHR);

struct mock_server {
   struct wl_display *display;
   thrd_t thread;
   atomic_uint surfaces_created;
   atomic_uint surfaces_destroyed;
   atomic_uint pools_created;
   atomic_uint pools_destroyed;
   atomic_uint buffers_created;
   atomic_uint buffers_destroyed;
};

static void
die(const char *what)
{
   fprintf(stderr, TEST_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
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

static void
destroy_resource(struct wl_client *client, struct wl_resource *resource)
{
   (void)client;
   wl_resource_destroy(resource);
}

static void
surface_resource_destroyed(struct wl_resource *resource)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   atomic_fetch_add(&server->surfaces_destroyed, 1);
}

static void
pool_resource_destroyed(struct wl_resource *resource)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   atomic_fetch_add(&server->pools_destroyed, 1);
}

static void
buffer_resource_destroyed(struct wl_resource *resource)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   atomic_fetch_add(&server->buffers_destroyed, 1);
}

static const struct wl_surface_interface surface_implementation = {
   .destroy = destroy_resource,
};

static void
compositor_create_surface(struct wl_client *client,
                          struct wl_resource *resource, uint32_t id)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   uint32_t version = wl_resource_get_version(resource);
   struct wl_resource *surface =
      wl_resource_create(client, &wl_surface_interface, version, id);
   if (!surface) {
      wl_client_post_no_memory(client);
      return;
   }

   atomic_fetch_add(&server->surfaces_created, 1);
   wl_resource_set_implementation(surface, &surface_implementation, server,
                                  surface_resource_destroyed);
}

static const struct wl_compositor_interface compositor_implementation = {
   .create_surface = compositor_create_surface,
};

static const struct wl_buffer_interface buffer_implementation = {
   .destroy = destroy_resource,
};

static void
pool_create_buffer(struct wl_client *client, struct wl_resource *resource,
                   uint32_t id, int32_t offset, int32_t width, int32_t height,
                   int32_t stride, uint32_t format)
{
   (void)offset;
   (void)width;
   (void)height;
   (void)stride;
   (void)format;
   struct mock_server *server = wl_resource_get_user_data(resource);
   struct wl_resource *buffer =
      wl_resource_create(client, &wl_buffer_interface, 1, id);
   if (!buffer) {
      wl_client_post_no_memory(client);
      return;
   }

   atomic_fetch_add(&server->buffers_created, 1);
   wl_resource_set_implementation(buffer, &buffer_implementation, server,
                                  buffer_resource_destroyed);
}

static const struct wl_shm_pool_interface pool_implementation = {
   .create_buffer = pool_create_buffer,
   .destroy = destroy_resource,
};

static void
shm_create_pool(struct wl_client *client, struct wl_resource *resource,
                uint32_t id, int32_t fd, int32_t size)
{
   (void)size;
   struct mock_server *server = wl_resource_get_user_data(resource);
   struct wl_resource *pool =
      wl_resource_create(client, &wl_shm_pool_interface, 1, id);
   close(fd);
   if (!pool) {
      wl_client_post_no_memory(client);
      return;
   }

   atomic_fetch_add(&server->pools_created, 1);
   wl_resource_set_implementation(pool, &pool_implementation, server,
                                  pool_resource_destroyed);
}

static const struct wl_shm_interface shm_implementation = {
   .create_pool = shm_create_pool,
};

static void
bind_compositor(struct wl_client *client, void *data, uint32_t version,
                uint32_t id)
{
   struct wl_resource *resource = wl_resource_create(
      client, &wl_compositor_interface, version < 4 ? version : 4, id);
   if (!resource) {
      wl_client_post_no_memory(client);
      return;
   }

   wl_resource_set_implementation(resource, &compositor_implementation, data,
                                  NULL);
}

static void
bind_shm(struct wl_client *client, void *data, uint32_t version, uint32_t id)
{
   (void)version;
   struct wl_resource *resource =
      wl_resource_create(client, &wl_shm_interface, 1, id);
   if (!resource) {
      wl_client_post_no_memory(client);
      return;
   }

   wl_resource_set_implementation(resource, &shm_implementation, data, NULL);
   wl_shm_send_format(resource, WL_SHM_FORMAT_ARGB8888);
   wl_shm_send_format(resource, WL_SHM_FORMAT_XRGB8888);
}

static int
run_server(void *data)
{
   wl_display_run(((struct mock_server *)data)->display);
   return 0;
}

struct client_globals {
   struct wl_compositor *compositor;
};

static void
registry_global(void *data, struct wl_registry *registry, uint32_t name,
                const char *interface, uint32_t version)
{
   struct client_globals *globals = data;
   if (!strcmp(interface, wl_compositor_interface.name)) {
      globals->compositor = wl_registry_bind(
         registry, name, &wl_compositor_interface, version < 4 ? version : 4);
   }
}

static void
registry_global_remove(void *data, struct wl_registry *registry, uint32_t name)
{
   (void)data;
   (void)registry;
   (void)name;
}

static const struct wl_registry_listener registry_listener = {
   .global = registry_global,
   .global_remove = registry_global_remove,
};

static uint32_t
swapchain_image_count(VkDevice device, VkSwapchainKHR swapchain)
{
   uint32_t count = 0;
   check(vkGetSwapchainImagesKHR(device, swapchain, &count, NULL),
         "vkGetSwapchainImagesKHR(count)");
   if (!count)
      die("swapchain contains no images");
   return count;
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
   for (unsigned i = 0; i < sizeof(choices) / sizeof(choices[0]); i++) {
      if (supported & choices[i])
         return choices[i];
   }

   die("surface has no composite alpha mode");
   return 0;
}

static VkSurfaceFormatKHR
choose_surface_format(const VkSurfaceFormatKHR *formats, uint32_t count)
{
   for (uint32_t i = 0; i < count; i++) {
      if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM ||
          formats[i].format == VK_FORMAT_R8G8B8A8_UNORM)
         return formats[i];
   }

   die("surface has no 8-bit UNORM format");
   return (VkSurfaceFormatKHR){0};
}

static void
expect_window_in_use(VkDevice device, const VkSwapchainCreateInfoKHR *info,
                     const char *what)
{
   VkSwapchainKHR unexpected = VK_NULL_HANDLE;
   VkResult result = vkCreateSwapchainKHR(device, info, NULL, &unexpected);
   if (result == VK_SUCCESS) {
      vkDestroySwapchainKHR(device, unexpected, NULL);
      fprintf(stderr, TEST_NAME ": %s unexpectedly succeeded\n", what);
      exit(EXIT_FAILURE);
   }
   if (result != VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) {
      fprintf(stderr, TEST_NAME ": %s returned %d, expected %d\n", what,
              result, VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
      exit(EXIT_FAILURE);
   }
}

int
main(void)
{
   struct mock_server server = {0};
   server.display = wl_display_create();
   if (!server.display)
      die("wl_display_create failed");
   if (!wl_global_create(server.display, &wl_compositor_interface, 4, &server,
                         bind_compositor) ||
       !wl_global_create(server.display, &wl_shm_interface, 1, &server,
                         bind_shm))
      die("wl_global_create failed");

   int sockets[2];
   if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) < 0)
      die("socketpair failed");
   if (!wl_client_create(server.display, sockets[0]))
      die("wl_client_create failed");
   struct wl_display *client_display = wl_display_connect_to_fd(sockets[1]);
   if (!client_display)
      die("wl_display_connect_to_fd failed");
   if (thrd_create(&server.thread, run_server, &server) != thrd_success)
      die("failed to start Wayland server thread");

   struct client_globals globals = {0};
   struct wl_registry *registry = wl_display_get_registry(client_display);
   if (!registry)
      die("wl_display_get_registry failed");
   if (wl_registry_add_listener(registry, &registry_listener, &globals) < 0 ||
       wl_display_roundtrip(client_display) < 0 || !globals.compositor)
      die("failed to discover mock compositor");
   struct wl_surface *wl_surface =
      wl_compositor_create_surface(globals.compositor);
   if (!wl_surface)
      die("wl_compositor_create_surface failed");

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
      .pApplicationName = TEST_NAME,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const char *instance_extensions[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = instance_extensions,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkCreateWaylandSurfaceKHR);
   LOAD_INSTANCE(instance, vkDestroySurfaceKHR);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceSupportKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceFormatsKHR);
   LOAD_INSTANCE(instance, vkCreateDevice);

   const VkWaylandSurfaceCreateInfoKHR surface_info = {
      .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
      .display = client_display,
      .surface = wl_surface,
   };
   VkSurfaceKHR surface;
   check(vkCreateWaylandSurfaceKHR(instance, &surface_info, NULL, &surface),
         "vkCreateWaylandSurfaceKHR");

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < physical_count; i++) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID) {
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
   for (uint32_t i = 0; i < queue_family_count; i++) {
      VkBool32 supported = VK_FALSE;
      check(vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, surface,
                                                 &supported),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
      if (supported && (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
         queue_family = i;
         break;
      }
   }
   free(queue_families);
   if (queue_family == UINT32_MAX)
      die("surface has no graphics present queue");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
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
   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateSwapchainKHR);
   LOAD_DEVICE(device, vkDestroySwapchainKHR);
   LOAD_DEVICE(device, vkGetSwapchainImagesKHR);

   VkSurfaceCapabilitiesKHR capabilities;
   check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface,
                                                   &capabilities),
         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
   uint32_t format_count = 0;
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count,
                                              NULL),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
   VkSurfaceFormatKHR *formats = calloc(format_count, sizeof(*formats));
   if (!formats)
      die("out of host memory");
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count,
                                              formats),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(list)");
   VkSurfaceFormatKHR format = choose_surface_format(formats, format_count);
   free(formats);

   VkExtent2D extent = capabilities.currentExtent;
   if (extent.width == UINT32_MAX)
      extent = (VkExtent2D){64, 64};
   const VkSwapchainCreateInfoKHR swapchain_info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = capabilities.minImageCount,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = capabilities.currentTransform,
      .compositeAlpha =
         choose_composite_alpha(capabilities.supportedCompositeAlpha),
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE,
   };

   uint32_t expected_buffers = 0;
   VkSwapchainKHR first;
   check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &first),
         "vkCreateSwapchainKHR(first)");
   expected_buffers += swapchain_image_count(device, first);

   expect_window_in_use(device, &swapchain_info, "first rejected create");
   expect_window_in_use(device, &swapchain_info, "second rejected create");

   VkSwapchainCreateInfoKHR replacement_info = swapchain_info;
   replacement_info.oldSwapchain = first;
   VkSwapchainKHR replacement;
   check(vkCreateSwapchainKHR(device, &replacement_info, NULL, &replacement),
         "vkCreateSwapchainKHR(replacement)");
   expected_buffers += swapchain_image_count(device, replacement);

   vkDestroySwapchainKHR(device, first, NULL);
   expect_window_in_use(device, &swapchain_info,
                        "create after retired swapchain destruction");
   vkDestroySwapchainKHR(device, replacement, NULL);

   VkSwapchainKHR final;
   check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &final),
         "vkCreateSwapchainKHR(after release)");
   expected_buffers += swapchain_image_count(device, final);
   vkDestroySwapchainKHR(device, final, NULL);

   vkDestroyDevice(device, NULL);
   vkDestroySurfaceKHR(instance, surface, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   wl_surface_destroy(wl_surface);
   wl_compositor_destroy(globals.compositor);
   wl_registry_destroy(registry);
   if (wl_display_roundtrip(client_display) < 0)
      die("final Wayland roundtrip failed");

   const unsigned surfaces_created = atomic_load(&server.surfaces_created);
   const unsigned surfaces_destroyed = atomic_load(&server.surfaces_destroyed);
   const unsigned pools_created = atomic_load(&server.pools_created);
   const unsigned pools_destroyed = atomic_load(&server.pools_destroyed);
   const unsigned buffers_created = atomic_load(&server.buffers_created);
   const unsigned buffers_destroyed = atomic_load(&server.buffers_destroyed);
   if (expected_buffers != 12 || surfaces_created != 1 ||
       surfaces_destroyed != 1 ||
       pools_created != expected_buffers || pools_destroyed != expected_buffers ||
       buffers_created != expected_buffers ||
       buffers_destroyed != expected_buffers) {
      fprintf(stderr,
              TEST_NAME ": bad Wayland lifetime: surfaces=%u/%u pools=%u/%u "
                        "buffers=%u/%u expected=%u\n",
              surfaces_created, surfaces_destroyed, pools_created,
              pools_destroyed, buffers_created, buffers_destroyed,
              expected_buffers);
      return EXIT_FAILURE;
   }

   wl_display_disconnect(client_display);
   wl_display_terminate(server.display);
   int thread_result = -1;
   thrd_join(server.thread, &thread_result);
   wl_display_destroy_clients(server.display);
   wl_display_destroy(server.display);
   if (thread_result != 0)
      die("Wayland server thread failed");

   printf(TEST_NAME ": PASS — rejected creates preserved the active "
                    "swapchain through replacement and destruction; "
                    "Wayland objects=%u/%u\n",
          buffers_created, buffers_destroyed);
   return EXIT_SUCCESS;
}
