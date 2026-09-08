/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include "drm-uapi/drm_fourcc.h"
#include <fcntl.h>
#include <gbm.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FIRST_WIDTH  240
#define FIRST_HEIGHT 160
#define NEXT_WIDTH   320
#define NEXT_HEIGHT  192
#define FRAMES       8
#define FENCE_WIDTH  1024
#define FENCE_HEIGHT 1024
#define FENCE_DRAWS  256

struct native_fence_api {
   PFNEGLCREATESYNCKHRPROC create;
   PFNEGLDESTROYSYNCKHRPROC destroy;
   PFNEGLWAITSYNCKHRPROC wait;
   PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fd;
   PFNEGLSETDAMAGEREGIONKHRPROC set_damage;
};

static bool gl_ok(const char *where);

static bool
has_extension(const char *extensions, const char *name)
{
   const size_t length = strlen(name);
   const char *match = extensions;
   while ((match = strstr(match, name))) {
      if ((match == extensions || match[-1] == ' ') &&
          (match[length] == '\0' || match[length] == ' '))
         return true;
      match += length;
   }
   return false;
}

static bool
native_fence_handoff(EGLDisplay display, EGLContext consumer_context,
                     EGLSurface consumer_surface, EGLContext producer_context,
                     const struct native_fence_api *api, GLuint program)
{
   int acquire_fd = -1, release_fd = -1;
   EGLSyncKHR source = EGL_NO_SYNC_KHR;
   EGLSyncKHR acquire = EGL_NO_SYNC_KHR;
   EGLSyncKHR release = EGL_NO_SYNC_KHR;
   GLuint vao = 0, texture = 0, fbo = 0;
   bool producer_current = false;
   bool ok = false;

   /* Retire the consumer first. Its next exported fence must therefore stay
    * unsignalled only because of the foreign server wait below. */
   glFinish();
   if (!gl_ok("pre-handoff consumer finish") ||
       !eglMakeCurrent(display, consumer_surface, consumer_surface,
                       producer_context))
      goto out;
   producer_current = true;

   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, FENCE_WIDTH, FENCE_HEIGHT);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      goto out;
   glUseProgram(program);
   glViewport(0, 0, FENCE_WIDTH, FENCE_HEIGHT);
   const GLint colour = glGetUniformLocation(program, "colour");
   glUniform4f(colour, 0.125f, 0.5f, 0.875f, 1.0f);
   for (unsigned draw = 0; draw < FENCE_DRAWS; ++draw)
      glDrawArrays(GL_TRIANGLES, 0, 3);

   const EGLint export_attrs[] = {
      EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
      EGL_NO_NATIVE_FENCE_FD_ANDROID,
      EGL_NONE,
   };
   source = api->create(display, EGL_SYNC_NATIVE_FENCE_ANDROID, export_attrs);
   if (!gl_ok("native-fence producer draw") || source == EGL_NO_SYNC_KHR)
      goto out;
   acquire_fd = api->dup_fd(display, source);
   if (acquire_fd < 0 || !(fcntl(acquire_fd, F_GETFD, 0) & FD_CLOEXEC))
      goto out;

   struct pollfd source_poll = {.fd = acquire_fd, .events = POLLIN};
   const int source_ready = poll(&source_poll, 1, 0);
   if (source_ready < 0 ||
       (source_ready > 0 && !(source_poll.revents & POLLIN)))
      goto out;
   if (!api->destroy(display, source))
      goto out;
   source = EGL_NO_SYNC_KHR;
   if (!eglMakeCurrent(display, consumer_surface, consumer_surface,
                       consumer_context))
      goto out;
   producer_current = false;

   const EGLint acquire_attrs[] = {
      EGL_SYNC_NATIVE_FENCE_FD_ANDROID, acquire_fd, EGL_NONE,
   };
   acquire = api->create(display, EGL_SYNC_NATIVE_FENCE_ANDROID,
                         acquire_attrs);
   if (acquire == EGL_NO_SYNC_KHR)
      goto out;
   /* EGL owns acquire_fd after successful import. */
   acquire_fd = -1;
   if (!api->wait(display, acquire, 0) || !api->destroy(display, acquire))
      goto out;
   acquire = EGL_NO_SYNC_KHR;

   release = api->create(display, EGL_SYNC_NATIVE_FENCE_ANDROID, export_attrs);
   if (release == EGL_NO_SYNC_KHR)
      goto out;
   release_fd = api->dup_fd(display, release);
   if (release_fd < 0 || !(fcntl(release_fd, F_GETFD, 0) & FD_CLOEXEC))
      goto out;

   struct pollfd pollfd = {.fd = release_fd, .events = POLLIN};
   if (source_ready == 0 && poll(&pollfd, 1, 0) != 0) {
      fprintf(stderr, "native release fence bypassed foreign acquire\n");
      goto out;
   }

   pollfd.revents = 0;
   if (poll(&pollfd, 1, 5000) != 1 || !(pollfd.revents & POLLIN)) {
      fprintf(stderr, "native release fence timed out\n");
      goto out;
   }

   ok = true;

out:
   if (release_fd >= 0)
      close(release_fd);
   if (release != EGL_NO_SYNC_KHR)
      api->destroy(display, release);
   if (acquire != EGL_NO_SYNC_KHR)
      api->destroy(display, acquire);
   else if (acquire_fd >= 0)
      close(acquire_fd);
   if (source != EGL_NO_SYNC_KHR)
      api->destroy(display, source);
   if (vao || fbo || texture) {
      if (!producer_current &&
          eglMakeCurrent(display, consumer_surface, consumer_surface,
                         producer_context))
         producer_current = true;
      if (producer_current) {
         if (fbo)
            glDeleteFramebuffers(1, &fbo);
         if (texture)
            glDeleteTextures(1, &texture);
         glDeleteVertexArrays(1, &vao);
      }
   }
   if (producer_current)
      eglMakeCurrent(display, consumer_surface, consumer_surface,
                     consumer_context);
   return ok;
}

static bool
gl_ok(const char *where)
{
   bool ok = true;
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "%s: GL error %#x\n", where, error);
      ok = false;
   }
   return ok;
}

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint compiled = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
   if (!compiled) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "swap shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(void)
{
   static const char vs_source[] =
      "#version 300 es\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
   static const char fs_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform vec4 colour;\n"
      "layout(location=0) out vec4 out_colour;\n"
      "void main() { out_colour = colour; }\n";
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 0;
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "swap program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static EGLConfig
choose_config(EGLDisplay display)
{
   const EGLint attrs[] = {
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
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
      if (eglGetConfigAttrib(display, configs[i], EGL_NATIVE_VISUAL_ID,
                             &visual) &&
          visual == GBM_FORMAT_ARGB8888)
         return configs[i];
   }
   return NULL;
}

static struct gbm_bo *
render_frames(EGLDisplay display, EGLSurface surface, struct gbm_surface *gbm,
              struct gbm_device *device, EGLContext consumer_context,
              EGLContext producer_context,
              const struct native_fence_api *fence_api, bool noop,
              GLuint program, unsigned width, unsigned height,
              unsigned generation)
{
   struct gbm_bo *retained = NULL;
   const GLint colour = glGetUniformLocation(program, "colour");
   glUseProgram(program);

   for (unsigned frame = 0; frame < FRAMES; ++frame) {
      EGLint age = 0;
      EGLint damage[] = {
         (EGLint)(width / 4),
         (EGLint)(height / 4),
         (EGLint)(width / 2),
         (EGLint)(height / 2),
      };
      if (!eglQuerySurface(display, surface, EGL_BUFFER_AGE_EXT, &age) ||
          !fence_api->set_damage(display, surface, damage, 1)) {
         fprintf(stderr, "EGL partial update %u/%u failed: %#x\n",
                 generation, frame, eglGetError());
         return NULL;
      }

      glViewport(0, 0, width, height);
      glEnable(GL_SCISSOR_TEST);
      glScissor(damage[0], damage[1], damage[2], damage[3]);
      glUniform4f(colour, (float)(frame + 1) / (float)FRAMES,
                  (float)(generation + 1) / 3.0f, 0.25f, 1.0f);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glDisable(GL_SCISSOR_TEST);
      if (!gl_ok("animated window draw") || !eglSwapBuffers(display, surface)) {
         fprintf(stderr, "EGL window swap %u failed: %#x\n", frame,
                 eglGetError());
         return NULL;
      }

      struct gbm_bo *front = gbm_surface_lock_front_buffer(gbm);
      if (!front || gbm_bo_get_width(front) != width ||
          gbm_bo_get_height(front) != height ||
          gbm_bo_get_format(front) != GBM_FORMAT_ARGB8888 ||
          gbm_bo_get_plane_count(front) != 1 ||
          gbm_bo_get_modifier(front) !=
             DRM_FORMAT_MOD_APPLE_GPU_TILED) {
         fprintf(stderr,
                 "GBM front buffer %u has invalid geometry: %ux%u format=%#x "
                 "planes=%d stride=%u modifier=%#llx\n",
                 frame, front ? gbm_bo_get_width(front) : 0,
                 front ? gbm_bo_get_height(front) : 0,
                 front ? gbm_bo_get_format(front) : 0,
                 front ? gbm_bo_get_plane_count(front) : 0,
                 front ? gbm_bo_get_stride(front) : 0,
                 (unsigned long long)(front ? gbm_bo_get_modifier(front) : 0));
         return NULL;
      }

      int fd = gbm_bo_get_fd(front);
      const uint64_t modifier = gbm_bo_get_modifier(front);
      struct gbm_import_fd_modifier_data import = {
         .width = width,
         .height = height,
         .format = GBM_FORMAT_ARGB8888,
         .num_fds = 1,
         .fds = {fd},
         .strides = {(int)gbm_bo_get_stride(front)},
         .offsets = {(int)gbm_bo_get_offset(front, 0)},
         .modifier = modifier,
      };

      /* A compositor can race an old geometry description against a newly
       * resized allocation. The tiled stride is part of the modifier ABI and
       * must reject exactly, without retaining the already imported BO. */
      struct gbm_import_fd_modifier_data malformed = import;
      malformed.strides[0] += 16;
      for (unsigned attempt = 0; attempt < 8; ++attempt) {
         struct gbm_bo *bad =
            gbm_bo_import(device, GBM_BO_IMPORT_FD_MODIFIER, &malformed,
                          GBM_BO_USE_RENDERING);
         if (bad) {
            fprintf(stderr, "GBM accepted malformed tiled stride on frame %u\n",
                    frame);
            gbm_bo_destroy(bad);
            close(fd);
            return NULL;
         }
      }

      struct gbm_bo *alias =
         fd >= 0 && (fcntl(fd, F_GETFD, 0) & FD_CLOEXEC)
            ? gbm_bo_import(device, GBM_BO_IMPORT_FD_MODIFIER, &import,
                            GBM_BO_USE_RENDERING)
            : NULL;
      if (fd >= 0)
         close(fd);
      if (!alias || gbm_bo_get_modifier(alias) != modifier ||
          gbm_bo_get_stride(alias) != gbm_bo_get_stride(front)) {
         fprintf(stderr, "GBM front-buffer dma-buf import %u failed\n", frame);
         return NULL;
      }

      gbm_surface_release_buffer(gbm, front);
      if (!noop && generation == 0 && frame == 0 &&
          !native_fence_handoff(display, consumer_context, surface,
                                producer_context, fence_api, program)) {
         fprintf(stderr, "native fence handoff %u/%u failed: %#x\n",
                 generation, frame, eglGetError());
         gbm_bo_destroy(alias);
         return NULL;
      }
      if (retained)
         gbm_bo_destroy(retained);
      retained = alias;
   }

   return retained;
}

static bool
retained_bo_ok(struct gbm_bo *bo, unsigned width, unsigned height)
{
   if (!bo || gbm_bo_get_width(bo) != width ||
       gbm_bo_get_height(bo) != height)
      return false;
   int fd = gbm_bo_get_fd(bo);
   const bool ok = fd >= 0 && (fcntl(fd, F_GETFD, 0) & FD_CLOEXEC);
   if (fd >= 0)
      close(fd);
   return ok;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   const bool expect_submit_failure =
      getenv("ASAHI_GL_EXPECT_SUBMIT_FAILURE") != NULL;
   const int drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   struct gbm_device *device =
      drm_fd >= 0 ? gbm_create_device(drm_fd) : NULL;
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");
   EGLDisplay display = device && get_platform_display
                           ? get_platform_display(EGL_PLATFORM_GBM_KHR,
                                                  device, NULL)
                           : EGL_NO_DISPLAY;
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "EGL GBM initialization failed: %#x\n", eglGetError());
      return 1;
   }

   const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
   const struct native_fence_api fence_api = {
      .create = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR"),
      .destroy =
         (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR"),
      .wait = (PFNEGLWAITSYNCKHRPROC)eglGetProcAddress("eglWaitSyncKHR"),
      .dup_fd = (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress(
         "eglDupNativeFenceFDANDROID"),
      .set_damage = (PFNEGLSETDAMAGEREGIONKHRPROC)eglGetProcAddress(
         "eglSetDamageRegionKHR"),
   };
   if (!extensions ||
       !has_extension(extensions, "EGL_ANDROID_native_fence_sync") ||
       !has_extension(extensions, "EGL_KHR_wait_sync") ||
       !has_extension(extensions, "EGL_EXT_buffer_age") ||
       !has_extension(extensions, "EGL_KHR_partial_update") ||
       !fence_api.create || !fence_api.destroy || !fence_api.wait ||
       !fence_api.dup_fd || !fence_api.set_damage) {
      fprintf(stderr, "required EGL fence/damage extensions missing\n");
      return 1;
   }

   EGLConfig config = choose_config(display);
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   const EGLint robust_context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT, EGL_TRUE,
      EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT,
      EGL_LOSE_CONTEXT_ON_RESET_EXT,
      EGL_NONE,
   };
   const EGLint *selected_context_attrs =
      expect_submit_failure ? robust_context_attrs : context_attrs;
   EGLContext context = config ? eglCreateContext(display, config,
                                                   EGL_NO_CONTEXT,
                                                   selected_context_attrs)
                               : EGL_NO_CONTEXT;
   if (!config || context == EGL_NO_CONTEXT)
      return 1;

   EGLContext producer_context =
      noop ? EGL_NO_CONTEXT
           : eglCreateContext(display, config, context,
                              selected_context_attrs);
   if (!noop && producer_context == EGL_NO_CONTEXT) {
      fprintf(stderr, "native-fence producer setup failed: %#x\n",
              eglGetError());
      return 1;
   }

   const unsigned widths[2] = {FIRST_WIDTH, NEXT_WIDTH};
   const unsigned heights[2] = {FIRST_HEIGHT, NEXT_HEIGHT};
   struct gbm_bo *old_generation = NULL;
   GLuint program = 0, vao = 0;
   for (unsigned generation = 0; generation < 2; ++generation) {
      const uint64_t modifiers[] = {
         DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
         DRM_FORMAT_MOD_APPLE_GPU_TILED,
         DRM_FORMAT_MOD_LINEAR,
      };
      struct gbm_surface *gbm_surface = gbm_surface_create_with_modifiers2(
         device, widths[generation], heights[generation],
         GBM_FORMAT_ARGB8888, modifiers,
         sizeof(modifiers) / sizeof(modifiers[0]), GBM_BO_USE_RENDERING);
      EGLSurface surface = gbm_surface
                              ? eglCreateWindowSurface(
                                   display, config,
                                   (EGLNativeWindowType)gbm_surface, NULL)
                              : EGL_NO_SURFACE;
      if (surface == EGL_NO_SURFACE ||
          !eglMakeCurrent(display, surface, surface, context)) {
         fprintf(stderr, "EGL GBM window generation %u failed: %#x\n",
                 generation, eglGetError());
         return 1;
      }

      if (!program) {
         program = make_program();
         glGenVertexArrays(1, &vao);
         glBindVertexArray(vao);
      }
      if (!program)
         return 1;

      if (expect_submit_failure) {
         glUseProgram(program);
         glViewport(0, 0, widths[generation], heights[generation]);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         eglSwapBuffers(display, surface);

         const GLenum status = glGetGraphicsResetStatus();
         if (status != GL_GUILTY_CONTEXT_RESET) {
            fprintf(stderr, "failed submit returned reset status %#x\n",
                    status);
            return 1;
         }

         puts("GL05 failed submit retired without a fence hang and reported "
              "a guilty reset");
         return 0;
      }

      struct gbm_bo *retained =
         render_frames(display, surface, gbm_surface, device, context,
                       producer_context, &fence_api, noop, program,
                       widths[generation], heights[generation], generation);
      if (!retained)
         return 1;

      if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                          EGL_NO_CONTEXT) ||
          !eglDestroySurface(display, surface))
         return 1;
      gbm_surface_destroy(gbm_surface);

      /* The compositor-imported dma-buf must outlive the producer's old
       * swapchain while the same GL context is rebound to the resized one. */
      if (!retained_bo_ok(retained, widths[generation], heights[generation]) ||
          (old_generation &&
           !retained_bo_ok(old_generation, FIRST_WIDTH, FIRST_HEIGHT)))
         return 1;
      if (old_generation)
         gbm_bo_destroy(old_generation);
      old_generation = retained;
   }

   gbm_bo_destroy(old_generation);
   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT))
      return 1;
   if (!noop) {
      eglDestroyContext(display, producer_context);
   }
   eglDestroyContext(display, context);
   eglTerminate(display);
   gbm_device_destroy(device);
   close(drm_fd);
   puts("GL05 EGL/GBM swap, partial update, native-fence handoff, dma-buf "
        "retention, resize and 16 frames exact");
   return 0;
}
