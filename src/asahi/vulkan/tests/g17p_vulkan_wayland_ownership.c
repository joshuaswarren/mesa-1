/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public Vulkan oracle for Wayland surface and swapchain lifetime across
 * failure, retry, and replacement.  A minimal in-process compositor exposes
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
#include <time.h>
#include <unistd.h>
#include <wayland-client.h>
#include <wayland-server-core.h>
#include <wayland-server-protocol.h>

#include "presentation-time-server-protocol.h"

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
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateSwapchainKHR);
VK_FN(vkDestroySwapchainKHR);
VK_FN(vkGetSwapchainImagesKHR);
VK_FN(vkCreateSemaphore);
VK_FN(vkDestroySemaphore);
VK_FN(vkAcquireNextImageKHR);
VK_FN(vkQueueSubmit);
VK_FN(vkQueuePresentKHR);
VK_FN(vkQueueWaitIdle);

struct mock_server {
   struct wl_display *display;
   thrd_t thread;
   atomic_uint surfaces_created;
   atomic_uint surfaces_destroyed;
   atomic_uint pools_created;
   atomic_uint pools_destroyed;
   atomic_uint buffers_created;
   atomic_uint buffers_destroyed;
   atomic_uint presentations_created;
   atomic_uint presentations_destroyed;
   atomic_uint feedback_created;
   atomic_uint feedback_destroyed;
   struct wl_resource *pending_buffer;
};

struct allocation_control {
   atomic_bool fail_next_object;
   atomic_bool poison_frees;
   atomic_uint injected_failures;
};

struct wrapper_control {
   atomic_bool fail_enabled;
   atomic_uint calls_since_arm;
   atomic_uint injected_failures;
   atomic_uint created;
   atomic_uint destroyed;
   unsigned fail_on_call;
   const char *fail_class;
   void *fail_proxy;
};

struct queue_control {
   atomic_bool fail_enabled;
   atomic_uint matching_calls;
   atomic_uint injected_failures;
};

static struct wrapper_control wrapper_control;
static struct queue_control queue_control;
static once_flag wrapper_symbols_once = ONCE_FLAG_INIT;
static once_flag queue_symbol_once = ONCE_FLAG_INIT;
static void *(*real_wl_proxy_create_wrapper)(void *);
static void (*real_wl_proxy_wrapper_destroy)(void *);
static struct wl_event_queue *(*real_wl_display_create_queue_with_name)(
   struct wl_display *, const char *);

static void
load_wrapper_symbols(void)
{
   real_wl_proxy_create_wrapper =
      (void *(*)(void *))dlsym(RTLD_NEXT, "wl_proxy_create_wrapper");
   real_wl_proxy_wrapper_destroy =
      (void (*)(void *))dlsym(RTLD_NEXT, "wl_proxy_wrapper_destroy");
   if (!real_wl_proxy_create_wrapper || !real_wl_proxy_wrapper_destroy) {
      fprintf(stderr, TEST_NAME ": failed to resolve Wayland wrapper symbols\n");
      _Exit(EXIT_FAILURE);
   }
}

static void
load_queue_symbol(void)
{
   real_wl_display_create_queue_with_name =
      (struct wl_event_queue *(*)(struct wl_display *, const char *))dlsym(
         RTLD_NEXT, "wl_display_create_queue_with_name");
}

__attribute__((visibility("default"))) struct wl_event_queue *
wl_display_create_queue_with_name(struct wl_display *display, const char *name)
{
   static const char target_prefix[] = "mesa vk surface ";

   call_once(&queue_symbol_once, load_queue_symbol);
   if (atomic_load(&queue_control.fail_enabled) && name &&
       !strncmp(name, target_prefix, sizeof(target_prefix) - 1)) {
      atomic_fetch_add(&queue_control.matching_calls, 1);
      if (atomic_exchange(&queue_control.fail_enabled, false)) {
         atomic_fetch_add(&queue_control.injected_failures, 1);
         return NULL;
      }
   }

   if (real_wl_display_create_queue_with_name)
      return real_wl_display_create_queue_with_name(display, name);
   return wl_display_create_queue(display);
}

__attribute__((visibility("default"))) void *
wl_proxy_create_wrapper(void *proxy)
{
   call_once(&wrapper_symbols_once, load_wrapper_symbols);

   bool target = wrapper_control.fail_proxy &&
                 proxy == wrapper_control.fail_proxy;
   if (atomic_load(&wrapper_control.fail_enabled) &&
       !target && wrapper_control.fail_class && proxy) {
      const char *proxy_class = wl_proxy_get_class((struct wl_proxy *)proxy);
      target = proxy_class && !strcmp(proxy_class, wrapper_control.fail_class);
   }

   if (atomic_load(&wrapper_control.fail_enabled) && target) {
      const unsigned call =
         atomic_fetch_add(&wrapper_control.calls_since_arm, 1) + 1;
      if (call == wrapper_control.fail_on_call &&
          atomic_exchange(&wrapper_control.fail_enabled, false)) {
         atomic_fetch_add(&wrapper_control.injected_failures, 1);
         return NULL;
      }
   }

   void *wrapper = real_wl_proxy_create_wrapper(proxy);
   if (wrapper)
      atomic_fetch_add(&wrapper_control.created, 1);
   return wrapper;
}

__attribute__((visibility("default"))) void
wl_proxy_wrapper_destroy(void *proxy_wrapper)
{
   call_once(&wrapper_symbols_once, load_wrapper_symbols);
   real_wl_proxy_wrapper_destroy(proxy_wrapper);
   atomic_fetch_add(&wrapper_control.destroyed, 1);
}

struct allocation_header {
   void *base;
   size_t size;
};

static void *
test_alloc(void *user_data, size_t size, size_t alignment,
           VkSystemAllocationScope scope)
{
   struct allocation_control *control = user_data;
   if (scope == VK_SYSTEM_ALLOCATION_SCOPE_OBJECT &&
       atomic_exchange(&control->fail_next_object, false)) {
      atomic_fetch_add(&control->injected_failures, 1);
      return NULL;
   }

   alignment = alignment < sizeof(void *) ? sizeof(void *) : alignment;
   if (alignment - 1 > SIZE_MAX - sizeof(struct allocation_header) ||
       size > SIZE_MAX - sizeof(struct allocation_header) - (alignment - 1))
      return NULL;
   size_t total = size + alignment - 1 + sizeof(struct allocation_header);
   void *base = malloc(total);
   if (!base)
      return NULL;

   uintptr_t address = (uintptr_t)base + sizeof(struct allocation_header);
   address = (address + alignment - 1) & ~(uintptr_t)(alignment - 1);
   struct allocation_header *header =
      (struct allocation_header *)(address - sizeof(*header));
   header->base = base;
   header->size = size;
   return (void *)address;
}

static void
test_free(void *user_data, void *memory)
{
   if (!memory)
      return;

   struct allocation_control *control = user_data;
   struct allocation_header *header =
      (struct allocation_header *)((uintptr_t)memory - sizeof(*header));
   if (atomic_load(&control->poison_frees))
      memset(memory, 0xa5, header->size);
   free(header->base);
}

static void *
test_realloc(void *user_data, void *original, size_t size, size_t alignment,
             VkSystemAllocationScope scope)
{
   if (!original)
      return test_alloc(user_data, size, alignment, scope);
   if (!size) {
      test_free(user_data, original);
      return NULL;
   }

   struct allocation_header *header =
      (struct allocation_header *)((uintptr_t)original - sizeof(*header));
   size_t old_size = header->size;
   void *replacement = test_alloc(user_data, size, alignment, scope);
   if (!replacement)
      return NULL;

   memcpy(replacement, original, old_size < size ? old_size : size);
   test_free(user_data, original);
   return replacement;
}

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

static void
surface_attach(struct wl_client *client, struct wl_resource *resource,
               struct wl_resource *buffer, int32_t x, int32_t y)
{
   (void)client;
   (void)x;
   (void)y;
   struct mock_server *server = wl_resource_get_user_data(resource);
   server->pending_buffer = buffer;
}

static void
surface_damage(struct wl_client *client, struct wl_resource *resource,
               int32_t x, int32_t y, int32_t width, int32_t height)
{
   (void)client;
   (void)resource;
   (void)x;
   (void)y;
   (void)width;
   (void)height;
}

static void
surface_frame(struct wl_client *client, struct wl_resource *resource,
              uint32_t callback_id)
{
   (void)resource;
   struct wl_resource *callback =
      wl_resource_create(client, &wl_callback_interface, 1, callback_id);
   if (!callback) {
      wl_client_post_no_memory(client);
      return;
   }

   wl_callback_send_done(callback, 1);
   wl_resource_destroy(callback);
}

static void
surface_commit(struct wl_client *client, struct wl_resource *resource)
{
   (void)client;
   struct mock_server *server = wl_resource_get_user_data(resource);
   if (server->pending_buffer) {
      wl_buffer_send_release(server->pending_buffer);
      server->pending_buffer = NULL;
   }
}

static const struct wl_surface_interface surface_implementation = {
   .destroy = destroy_resource,
   .attach = surface_attach,
   .damage = surface_damage,
   .frame = surface_frame,
   .commit = surface_commit,
   .damage_buffer = surface_damage,
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

static void
presentation_resource_destroyed(struct wl_resource *resource)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   atomic_fetch_add(&server->presentations_destroyed, 1);
}

static void
feedback_resource_destroyed(struct wl_resource *resource)
{
   struct mock_server *server = wl_resource_get_user_data(resource);
   atomic_fetch_add(&server->feedback_destroyed, 1);
}

static void
presentation_destroy(struct wl_client *client, struct wl_resource *resource)
{
   (void)client;
   wl_resource_destroy(resource);
}

static void
presentation_feedback(struct wl_client *client, struct wl_resource *resource,
                      struct wl_resource *surface, uint32_t id)
{
   (void)surface;
   struct mock_server *server = wl_resource_get_user_data(resource);
   struct wl_resource *feedback = wl_resource_create(
      client, &wp_presentation_feedback_interface,
      wl_resource_get_version(resource), id);
   if (!feedback) {
      wl_client_post_no_memory(client);
      return;
   }

   atomic_fetch_add(&server->feedback_created, 1);
   wl_resource_set_implementation(feedback, NULL, server,
                                  feedback_resource_destroyed);
   wp_presentation_feedback_send_discarded(feedback);
   wl_resource_destroy(feedback);
}

static const struct wp_presentation_interface presentation_implementation = {
   .destroy = presentation_destroy,
   .feedback = presentation_feedback,
};

static void
bind_presentation(struct wl_client *client, void *data, uint32_t version,
                  uint32_t id)
{
   struct mock_server *server = data;
   struct wl_resource *resource = wl_resource_create(
      client, &wp_presentation_interface, version < 1 ? version : 1, id);
   if (!resource) {
      wl_client_post_no_memory(client);
      return;
   }

   atomic_fetch_add(&server->presentations_created, 1);
   wl_resource_set_implementation(resource, &presentation_implementation,
                                  server, presentation_resource_destroyed);
   wp_presentation_send_clock_id(resource, CLOCK_MONOTONIC);
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
main(int argc, char **argv)
{
   const bool present_allocation_failure =
      argc == 2 && !strcmp(argv[1], "--present-allocation-failure");
   const bool surface_init_retry =
      argc == 2 && !strcmp(argv[1], "--surface-init-retry");
   const bool present_wrapper_retry =
      argc == 2 && !strcmp(argv[1], "--present-wrapper-retry");
   const bool present_queue_retry =
      argc == 2 && !strcmp(argv[1], "--present-queue-retry");
   const bool presentation_wrapper_retry =
      argc == 2 && !strcmp(argv[1], "--presentation-wrapper-retry");
   const bool poisoned_retry = surface_init_retry || present_wrapper_retry;
   const bool wrapper_retry = poisoned_retry || presentation_wrapper_retry;
   const bool create_retry = wrapper_retry || present_queue_retry;
   if (argc > 2 ||
       (argc == 2 && !present_allocation_failure && !surface_init_retry &&
        !present_wrapper_retry && !present_queue_retry &&
        !presentation_wrapper_retry))
      die("usage: " TEST_NAME
          " [--present-allocation-failure|--surface-init-retry|"
          "--present-wrapper-retry|--present-queue-retry|"
          "--presentation-wrapper-retry]");

   struct mock_server server = {0};
   server.display = wl_display_create();
   if (!server.display)
      die("wl_display_create failed");
   if (!wl_global_create(server.display, &wl_compositor_interface, 4, &server,
                         bind_compositor) ||
       !wl_global_create(server.display, &wl_shm_interface, 1, &server,
                         bind_shm))
      die("wl_global_create failed");
   if (presentation_wrapper_retry &&
       !wl_global_create(server.display, &wp_presentation_interface, 1,
                         &server, bind_presentation))
      die("wp_presentation global creation failed");

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

   struct allocation_control allocation_control = {0};
   atomic_store(&allocation_control.poison_frees, poisoned_retry);
   const VkAllocationCallbacks allocation_callbacks = {
      .pUserData = &allocation_control,
      .pfnAllocation = test_alloc,
      .pfnReallocation = test_realloc,
      .pfnFree = test_free,
   };
   const VkAllocationCallbacks *instance_allocator =
      (present_allocation_failure || poisoned_retry) ? &allocation_callbacks : NULL;

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
   check(vkCreateInstance(&instance_info, instance_allocator, &instance),
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
   const char *device_extensions[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_PRESENT_ID_EXTENSION_NAME,
   };
   const VkPhysicalDevicePresentIdFeaturesKHR present_id_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
      .presentId = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = present_allocation_failure ? &present_id_features : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = present_allocation_failure ? 2 : 1,
      .ppEnabledExtensionNames = device_extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");
   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateSwapchainKHR);
   LOAD_DEVICE(device, vkDestroySwapchainKHR);
   LOAD_DEVICE(device, vkGetSwapchainImagesKHR);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateSemaphore);
   LOAD_DEVICE(device, vkDestroySemaphore);
   LOAD_DEVICE(device, vkAcquireNextImageKHR);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkQueuePresentKHR);
   LOAD_DEVICE(device, vkQueueWaitIdle);

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
   if (create_retry) {
      if (present_queue_retry) {
         atomic_store(&queue_control.matching_calls, 0);
         atomic_store(&queue_control.fail_enabled, true);
      } else {
         atomic_store(&wrapper_control.calls_since_arm, 0);
         wrapper_control.fail_on_call =
            present_wrapper_retry ? 2 : 1;
         wrapper_control.fail_class =
            presentation_wrapper_retry ? "wp_presentation" : NULL;
         wrapper_control.fail_proxy =
            presentation_wrapper_retry ? NULL : wl_surface;
         atomic_store(&wrapper_control.fail_enabled, true);
      }

      VkSwapchainKHR failed = VK_NULL_HANDLE;
      VkResult result =
         vkCreateSwapchainKHR(device, &swapchain_info, NULL, &failed);
      if (result != VK_ERROR_OUT_OF_HOST_MEMORY) {
         if (result == VK_SUCCESS)
            vkDestroySwapchainKHR(device, failed, NULL);
         fprintf(stderr,
                 TEST_NAME ": targeted Wayland allocation failure returned "
                           "%d, expected %d\n",
                 result, VK_ERROR_OUT_OF_HOST_MEMORY);
         return EXIT_FAILURE;
      }
      if (present_queue_retry) {
         if (atomic_load(&queue_control.injected_failures) != 1 ||
             atomic_load(&queue_control.matching_calls) != 1)
            die("present queue creation did not fail exactly once");
      } else {
         if (atomic_load(&wrapper_control.injected_failures) != 1 ||
             atomic_load(&wrapper_control.calls_since_arm) !=
                wrapper_control.fail_on_call)
            die("target wrapper creation did not fail exactly once");
      }
   }

   VkSwapchainKHR first;
   check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &first),
         "vkCreateSwapchainKHR(first)");
   const uint32_t first_image_count = swapchain_image_count(device, first);
   expected_buffers += first_image_count;

   if (present_allocation_failure) {
      VkQueue queue;
      vkGetDeviceQueue(device, queue_family, 0, &queue);
      const VkSemaphoreCreateInfo semaphore_info = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      };
      VkSemaphore acquired, render_done;
      check(vkCreateSemaphore(device, &semaphore_info, NULL, &acquired),
            "vkCreateSemaphore(acquired)");
      check(vkCreateSemaphore(device, &semaphore_info, NULL, &render_done),
            "vkCreateSemaphore(render done)");

      /* Touch every image once so the common WSI throttle fences are already
       * allocated before the targeted backend allocation failure. */
      for (unsigned present = 0; present <= first_image_count; present++) {
         const bool inject_failure = present == first_image_count;
         uint32_t image_index = UINT32_MAX;
         check(vkAcquireNextImageKHR(device, first, UINT64_MAX, acquired,
                                     VK_NULL_HANDLE, &image_index),
               "vkAcquireNextImageKHR");

         const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
         const VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &acquired,
            .pWaitDstStageMask = &wait_stage,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_done,
         };
         check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE),
               "vkQueueSubmit");

         const uint64_t present_id = 1;
         const VkPresentIdKHR present_id_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .swapchainCount = 1,
            .pPresentIds = &present_id,
         };
         VkResult per_swapchain = VK_SUCCESS;
         const VkPresentInfoKHR present_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = inject_failure ? &present_id_info : NULL,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_done,
            .swapchainCount = 1,
            .pSwapchains = &first,
            .pImageIndices = &image_index,
            .pResults = &per_swapchain,
         };

         if (inject_failure)
            atomic_store(&allocation_control.fail_next_object, true);
         VkResult result = vkQueuePresentKHR(queue, &present_info);
         if (!inject_failure) {
            check(result, "vkQueuePresentKHR(warmup)");
            if (per_swapchain != VK_SUCCESS)
               die("warmup present returned a per-swapchain error");
            check(vkQueueWaitIdle(queue), "vkQueueWaitIdle(warmup)");
         } else if (result != VK_ERROR_DEVICE_LOST ||
                    per_swapchain != VK_ERROR_DEVICE_LOST) {
            fprintf(stderr,
                    TEST_NAME ": allocation failure returned %d/%d, expected "
                              "%d\n",
                    result, per_swapchain, VK_ERROR_DEVICE_LOST);
            return EXIT_FAILURE;
         }
      }

      if (atomic_load(&allocation_control.injected_failures) != 1)
         die("present tracking allocation did not fail exactly once");
      if (vkQueueWaitIdle(queue) != VK_ERROR_DEVICE_LOST)
         die("present allocation failure did not latch device loss");

      vkDestroySemaphore(device, render_done, NULL);
      vkDestroySemaphore(device, acquired, NULL);
      vkDestroySwapchainKHR(device, first, NULL);
   } else if (create_retry) {
      vkDestroySwapchainKHR(device, first, NULL);

      VkSwapchainKHR second;
      check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &second),
            "vkCreateSwapchainKHR(after recovered destruction)");
      expected_buffers += swapchain_image_count(device, second);
      vkDestroySwapchainKHR(device, second, NULL);
   } else {
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
   }

   vkDestroyDevice(device, NULL);
   vkDestroySurfaceKHR(instance, surface, NULL);
   vkDestroyInstance(instance, instance_allocator);
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
   const unsigned presentations_created =
      atomic_load(&server.presentations_created);
   const unsigned presentations_destroyed =
      atomic_load(&server.presentations_destroyed);
   const unsigned feedback_created = atomic_load(&server.feedback_created);
   const unsigned feedback_destroyed = atomic_load(&server.feedback_destroyed);
   const unsigned wrappers_created = atomic_load(&wrapper_control.created);
   const unsigned wrappers_destroyed = atomic_load(&wrapper_control.destroyed);
   const unsigned exact_buffers =
      present_allocation_failure ? 4 : create_retry ? 8 : 12;
   const bool exact_presentation_lifetime =
      !presentation_wrapper_retry ||
      (presentations_created == 4 && presentations_destroyed == 4 &&
       feedback_created == 0 && feedback_destroyed == 0);
   if (expected_buffers != exact_buffers || surfaces_created != 1 ||
       surfaces_destroyed != 1 ||
       pools_created != expected_buffers || pools_destroyed != expected_buffers ||
       buffers_created != expected_buffers ||
       buffers_destroyed != expected_buffers ||
       wrappers_created != wrappers_destroyed || !exact_presentation_lifetime) {
      fprintf(stderr,
              TEST_NAME ": bad Wayland lifetime: surfaces=%u/%u pools=%u/%u "
                        "buffers=%u/%u wrappers=%u/%u presentation=%u/%u "
                        "feedback=%u/%u expected=%u\n",
              surfaces_created, surfaces_destroyed, pools_created,
              pools_destroyed, buffers_created, buffers_destroyed,
              wrappers_created, wrappers_destroyed, presentations_created,
              presentations_destroyed, feedback_created, feedback_destroyed,
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

   if (present_allocation_failure) {
      printf(TEST_NAME ": PASS — post-enqueue present allocation failure "
                       "latched device loss; Wayland objects=%u/%u\n",
             buffers_created, buffers_destroyed);
   } else if (surface_init_retry) {
      printf(TEST_NAME ": PASS — surface wrapper allocation failed once, "
                       "then retry and reuse succeeded; Wayland objects=%u/%u "
                       "wrappers=%u/%u\n",
             buffers_created, buffers_destroyed, wrappers_created,
             wrappers_destroyed);
   } else if (present_wrapper_retry) {
      printf(TEST_NAME ": PASS — present wrapper allocation failed once, "
                       "then retry and reuse succeeded; Wayland objects=%u/%u "
                       "wrappers=%u/%u\n",
             buffers_created, buffers_destroyed, wrappers_created,
             wrappers_destroyed);
   } else if (present_queue_retry) {
      printf(TEST_NAME ": PASS — present queue allocation failed once, then "
                       "retry and reuse succeeded; Wayland objects=%u/%u "
                       "wrappers=%u/%u\n",
             buffers_created, buffers_destroyed, wrappers_created,
             wrappers_destroyed);
   } else if (presentation_wrapper_retry) {
      printf(TEST_NAME ": PASS — presentation protocol wrapper allocation "
                       "failed once, then retry and reuse succeeded; Wayland "
                       "objects=%u/%u wrappers=%u/%u presentation=%u/%u\n",
             buffers_created, buffers_destroyed, wrappers_created,
             wrappers_destroyed, presentations_created,
             presentations_destroyed);
   } else {
      printf(TEST_NAME ": PASS — rejected creates preserved the active "
                       "swapchain through replacement and destruction; "
                       "Wayland objects=%u/%u\n",
             buffers_created, buffers_destroyed);
   }
   return EXIT_SUCCESS;
}
