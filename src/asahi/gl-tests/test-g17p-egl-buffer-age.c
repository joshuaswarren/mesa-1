/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact runtime probe for the Honeykrisp/Zink EGL buffer-age containment.
 * It deliberately queries before every draw: both the GBM platform and the
 * Wayland/Kopper platform must still acquire a back buffer, but report age 0.
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wayland-client.h>
#include <wayland-egl-core.h>

#include "xdg-shell-client-protocol.h"

#define WIDTH  256
#define HEIGHT 192
#define FRAMES 8

struct wayland_window {
   struct wl_display *display;
   struct wl_registry *registry;
   struct wl_compositor *compositor;
   struct xdg_wm_base *wm_base;
   struct wl_surface *surface;
   struct xdg_surface *xdg_surface;
   struct xdg_toplevel *toplevel;
   struct wl_egl_window *egl_window;
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

static bool
has_extension(const char *extensions, const char *name)
{
   const size_t length = strlen(name);
   const char *match = extensions;

   while (match && (match = strstr(match, name))) {
      if ((match == extensions || match[-1] == ' ') &&
          (match[length] == '\0' || match[length] == ' '))
         return true;
      match += length;
   }

   return false;
}

static EGLConfig
choose_config(EGLDisplay display, bool gbm)
{
   const EGLint attrs[] = {
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig configs[64];
   EGLint count = 0;

   if (!eglChooseConfig(display, attrs, configs, 64, &count))
      return NULL;

   for (EGLint i = 0; i < count; ++i) {
      EGLint visual = 0;
      if (!gbm ||
          (eglGetConfigAttrib(display, configs[i], EGL_NATIVE_VISUAL_ID,
                              &visual) &&
           visual == GBM_FORMAT_ARGB8888))
         return configs[i];
   }

   return NULL;
}

static bool
setup_wayland(struct wayland_window *window)
{
   window->display = wl_display_connect(NULL);
   if (!window->display)
      return false;

   window->registry = wl_display_get_registry(window->display);
   if (!window->registry)
      return false;
   wl_registry_add_listener(window->registry, &registry_listener, window);
   if (wl_display_roundtrip(window->display) < 0 || !window->compositor ||
       !window->wm_base)
      return false;

   xdg_wm_base_add_listener(window->wm_base, &wm_base_listener, window);
   window->surface = wl_compositor_create_surface(window->compositor);
   window->xdg_surface =
      xdg_wm_base_get_xdg_surface(window->wm_base, window->surface);
   window->toplevel = xdg_surface_get_toplevel(window->xdg_surface);
   if (!window->surface || !window->xdg_surface || !window->toplevel)
      return false;

   xdg_surface_add_listener(window->xdg_surface, &xdg_surface_listener,
                            window);
   xdg_toplevel_add_listener(window->toplevel, &toplevel_listener, window);
   xdg_toplevel_set_title(window->toplevel, "G17P EGL buffer-age probe");
   wl_surface_commit(window->surface);
   while (!window->configured && !window->closed) {
      if (wl_display_dispatch(window->display) < 0)
         return false;
   }

   window->egl_window = wl_egl_window_create(window->surface, WIDTH, HEIGHT);
   return window->egl_window != NULL && !window->closed;
}

static void
destroy_wayland(struct wayland_window *window)
{
   if (window->egl_window)
      wl_egl_window_destroy(window->egl_window);
   if (window->toplevel)
      xdg_toplevel_destroy(window->toplevel);
   if (window->xdg_surface)
      xdg_surface_destroy(window->xdg_surface);
   if (window->surface)
      wl_surface_destroy(window->surface);
   if (window->wm_base)
      xdg_wm_base_destroy(window->wm_base);
   if (window->compositor)
      wl_compositor_destroy(window->compositor);
   if (window->registry)
      wl_registry_destroy(window->registry);
   if (window->display)
      wl_display_disconnect(window->display);
}

int
main(int argc, char **argv)
{
   const bool use_gbm = argc == 2 && !strcmp(argv[1], "--gbm");
   const bool use_wayland = argc == 2 && !strcmp(argv[1], "--wayland");
   struct wayland_window window = {0};
   struct gbm_device *gbm_device = NULL;
   struct gbm_surface *gbm_surface = NULL;
   int drm_fd = -1;

   if (!use_gbm && !use_wayland) {
      fprintf(stderr, "usage: %s {--gbm|--wayland}\n", argv[0]);
      return 2;
   }

   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");
   if (!get_platform_display) {
      fputs("EGL_EXT_platform_base entrypoint missing\n", stderr);
      return 1;
   }

   void *native_display = NULL;
   EGLenum platform;
   if (use_gbm) {
      drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
      gbm_device = drm_fd >= 0 ? gbm_create_device(drm_fd) : NULL;
      native_display = gbm_device;
      platform = EGL_PLATFORM_GBM_KHR;
   } else {
      if (!setup_wayland(&window)) {
         fputs("Wayland xdg-toplevel setup failed\n", stderr);
         destroy_wayland(&window);
         return 1;
      }
      native_display = window.display;
      platform = EGL_PLATFORM_WAYLAND_KHR;
   }

   EGLDisplay display = native_display
                           ? get_platform_display(platform, native_display, NULL)
                           : EGL_NO_DISPLAY;
   EGLint major = 0, minor = 0;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "EGL %s initialization failed: %#x\n",
              use_gbm ? "GBM" : "Wayland", eglGetError());
      destroy_wayland(&window);
      return 1;
   }

   const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
   PFNEGLGETDISPLAYDRIVERNAMEPROC get_display_driver_name =
      (PFNEGLGETDISPLAYDRIVERNAMEPROC)eglGetProcAddress(
         "eglGetDisplayDriverName");
   if (!has_extension(extensions, "EGL_EXT_buffer_age") ||
       !has_extension(extensions, "EGL_MESA_query_driver") ||
       !get_display_driver_name) {
      fputs("EGL_EXT_buffer_age/EGL_MESA_query_driver missing\n", stderr);
      return 1;
   }
   const char *egl_driver = get_display_driver_name(display);

   EGLConfig config = choose_config(display, use_gbm);
   if (use_gbm && config) {
      gbm_surface = gbm_surface_create(gbm_device, WIDTH, HEIGHT,
                                       GBM_FORMAT_ARGB8888,
                                       GBM_BO_USE_RENDERING);
   }
   EGLNativeWindowType native_window = use_gbm
                                          ? (EGLNativeWindowType)gbm_surface
                                          : (EGLNativeWindowType)window.egl_window;
   EGLSurface surface = config
                           ? eglCreateWindowSurface(display, config,
                                                    native_window, NULL)
                           : EGL_NO_SURFACE;
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE,
   };
   EGLContext context = config
                           ? eglCreateContext(display, config, EGL_NO_CONTEXT,
                                              context_attrs)
                           : EGL_NO_CONTEXT;
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context)) {
      fprintf(stderr, "EGL window/context setup failed: %#x\n", eglGetError());
      return 1;
   }

   eglSwapInterval(display, 0);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("platform=%s egl-driver=%s renderer=%s egl=%d.%d\n",
          use_gbm ? "gbm" : "wayland",
          egl_driver ? egl_driver : "(null)",
          renderer ? renderer : "(null)", major, minor);
   if (!egl_driver || strcmp(egl_driver, "zink") || !renderer ||
       !strstr(renderer, "zink") ||
       !strstr(renderer, "MESA_HONEYKRISP")) {
      fputs("SKIP: probe did not select Honeykrisp through Zink\n", stderr);
      return 77;
   }

   for (unsigned frame = 0; frame < FRAMES; ++frame) {
      if (frame == FRAMES / 2) {
         puts("retained-wait=3s");
         sleep(3);
      }

      EGLint age = -1;
      if (!eglQuerySurface(display, surface, EGL_BUFFER_AGE_EXT, &age)) {
         fprintf(stderr, "frame=%u query failed: %#x\n", frame,
                 eglGetError());
         return 1;
      }
      printf("frame=%u age=%d phase=%s\n", frame, age,
             frame < FRAMES / 2 ? "immediate" : "retained");
      if (age != 0) {
         fprintf(stderr, "FAIL: %s buffer age escaped containment: %d\n",
                 use_gbm ? "GBM" : "Wayland/Kopper", age);
         return 1;
      }

      glViewport(0, 0, WIDTH, HEIGHT);
      glClearColor((float)(frame + 1) / FRAMES, 0.25f, 0.75f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      if (glGetError() != GL_NO_ERROR || !eglSwapBuffers(display, surface)) {
         fprintf(stderr, "frame=%u swap failed: %#x\n", frame, eglGetError());
         return 1;
      }

      if (use_gbm) {
         struct gbm_bo *front = gbm_surface_lock_front_buffer(gbm_surface);
         if (!front) {
            fprintf(stderr, "frame=%u GBM front-buffer lock failed\n", frame);
            return 1;
         }
         gbm_surface_release_buffer(gbm_surface, front);
      } else if (wl_display_roundtrip(window.display) < 0 || window.closed) {
         fprintf(stderr, "frame=%u Wayland roundtrip failed\n", frame);
         return 1;
      }
   }

   printf("PASS: %s Honeykrisp/Zink buffer age stayed zero across %u "
          "acquire/swap cycles and retained wait\n",
          use_gbm ? "GBM" : "Wayland/Kopper", FRAMES);

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   if (gbm_surface)
      gbm_surface_destroy(gbm_surface);
   if (gbm_device)
      gbm_device_destroy(gbm_device);
   if (drm_fd >= 0)
      close(drm_fd);
   destroy_wayland(&window);
   return 0;
}
