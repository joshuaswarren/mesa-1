/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 32
#define HEIGHT 32
#define ROW_BYTES (WIDTH * 4)
#define DATA_BYTES (ROW_BYTES * HEIGHT)
#define GUARD_SIZE 32
#define GUARD_BYTE 0xa5
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static const uint8_t first_bgra[4] = {0x11, 0x22, 0x33, 0xff};
static const uint8_t second_bgra[4] = {0x77, 0x66, 0x55, 0xff};
static const uint8_t first_rgba[4] = {0x33, 0x22, 0x11, 0xff};
static const uint8_t second_rgba[4] = {0x55, 0x66, 0x77, 0xff};
static const uint8_t clear_bgra[4] = {0x66, 0x55, 0x44, 0xff};
static const uint8_t clear_rgba[4] = {0x44, 0x55, 0x66, 0xff};

struct guarded_pixels {
   uint8_t before[GUARD_SIZE];
   uint8_t data[DATA_BYTES];
   uint8_t after[GUARD_SIZE];
};

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
guard_ok(const uint8_t guard[GUARD_SIZE])
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (guard[i] != GUARD_BYTE)
         return false;
   }
   return true;
}

static bool
gl_ok(const char *where)
{
   bool ok = true;
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "%s: GL error 0x%04x\n", where, error);
      ok = false;
   }
   return ok;
}

static void
fill_raw(struct guarded_pixels *pixels, const uint8_t bgra[4])
{
   memset(pixels, GUARD_BYTE, sizeof(*pixels));
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i)
      memcpy(&pixels->data[4 * i], bgra, 4);
}

static bool
map_write(struct gbm_bo *bo, const struct guarded_pixels *pixels)
{
   uint32_t stride = 0;
   void *token = NULL;
   uint8_t *map = gbm_bo_map(bo, 0, 0, WIDTH, HEIGHT,
                             GBM_BO_TRANSFER_WRITE, &stride, &token);
   if (!map || !token || stride < ROW_BYTES)
      return false;
   for (unsigned y = 0; y < HEIGHT; ++y)
      memcpy(map + y * stride, pixels->data + y * ROW_BYTES, ROW_BYTES);
   gbm_bo_unmap(bo, token);
   return guard_ok(pixels->before) && guard_ok(pixels->after);
}

static bool
map_read(struct gbm_bo *bo, const struct guarded_pixels *expected)
{
   struct guarded_pixels got;
   memset(&got, GUARD_BYTE, sizeof(got));
   uint32_t stride = 0;
   void *token = NULL;
   const uint8_t *map = gbm_bo_map(bo, 0, 0, WIDTH, HEIGHT,
                                   GBM_BO_TRANSFER_READ, &stride, &token);
   if (!map || !token || stride < ROW_BYTES)
      return false;
   for (unsigned y = 0; y < HEIGHT; ++y)
      memcpy(got.data + y * ROW_BYTES, map + y * stride, ROW_BYTES);
   gbm_bo_unmap(bo, token);
   return guard_ok(got.before) && guard_ok(got.after) &&
          memcmp(got.data, expected->data, DATA_BYTES) == 0;
}

static bool
gl_read(const char *phase, const uint8_t rgba[4], const uint8_t storage[4],
        const uint8_t stale[4], bool noop)
{
   struct guarded_pixels got;
   memset(&got, GUARD_BYTE, sizeof(got));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, got.data);
   if (!gl_ok("dma-buf readback") || !guard_ok(got.before) ||
       !guard_ok(got.after))
      return false;
   if (noop)
      return true;
   unsigned exact = 0, zero = 0, raw = 0, old = 0, other = 0;
   unsigned first_bad = UINT32_MAX;
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
      const uint8_t *pixel = &got.data[4 * i];
      if (!memcmp(pixel, rgba, 4))
         exact++;
      else if (!memcmp(pixel, (const uint8_t[4]){0, 0, 0, 0}, 4))
         zero++;
      else if (storage && !memcmp(pixel, storage, 4))
         raw++;
      else if (stale && !memcmp(pixel, stale, 4))
         old++;
      else
         other++;
      if (first_bad == UINT32_MAX && memcmp(pixel, rgba, 4))
         first_bad = i;
   }

   if (exact == WIDTH * HEIGHT)
      return true;

   const uint8_t *pixel = &got.data[4 * first_bad];
   fprintf(stderr,
           "%s: first bad pixel %u got=%02x%02x%02x%02x "
           "expected=%02x%02x%02x%02x classes exact=%u zero=%u "
           "raw-bgra=%u stale=%u other=%u\n",
           phase, first_bad, pixel[0], pixel[1], pixel[2], pixel[3], rgba[0],
           rgba[1], rgba[2], rgba[3], exact, zero, raw, old, other);
   return false;
}

static bool
native_texture_control(bool noop)
{
   struct guarded_pixels upload;
   fill_raw(&upload, first_rgba);

   GLuint texture, fbo;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, upload.data);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   const bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
                      GL_FRAMEBUFFER_COMPLETE &&
                   gl_ok("native texture control") &&
                   gl_read("native texture control", first_rgba, first_rgba,
                           NULL, noop);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   return ok;
}

static bool
format_and_modifier_supported(EGLDisplay display,
                              PFNEGLQUERYDMABUFFORMATSEXTPROC query_formats,
                              PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers)
{
   EGLint count = 0;
   if (!query_formats(display, 0, NULL, &count) || count <= 0)
      return false;
   EGLint *formats = calloc(count, sizeof(*formats));
   if (!formats)
      return false;
   if (!query_formats(display, count, formats, &count)) {
      free(formats);
      return false;
   }
   bool found_format = false;
   for (EGLint i = 0; i < count; ++i)
      found_format |= formats[i] == DRM_FORMAT_ARGB8888;
   free(formats);
   if (!found_format)
      return false;

   count = 0;
   if (!query_modifiers(display, DRM_FORMAT_ARGB8888, 0, NULL, NULL,
                        &count) || count <= 0)
      return false;
   EGLuint64KHR *modifiers = calloc(count, sizeof(*modifiers));
   EGLBoolean *external = calloc(count, sizeof(*external));
   if (!modifiers || !external ||
       !query_modifiers(display, DRM_FORMAT_ARGB8888, count, modifiers,
                        external, &count)) {
      free(external);
      free(modifiers);
      return false;
   }
   bool found_linear = false;
   for (EGLint i = 0; i < count; ++i)
      found_linear |= modifiers[i] == DRM_FORMAT_MOD_LINEAR;
   free(external);
   free(modifiers);
   return found_linear;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor))
      return 1;

   const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
   if (!extensions ||
       !has_extension(extensions, "EGL_EXT_image_dma_buf_import") ||
       !has_extension(extensions, "EGL_EXT_image_dma_buf_import_modifiers") ||
       !has_extension(extensions, "EGL_KHR_image_base")) {
      fprintf(stderr, "required EGL dma-buf image extensions missing\n");
      return 1;
   }

   PFNEGLCREATEIMAGEKHRPROC create_image =
      (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   PFNEGLDESTROYIMAGEKHRPROC destroy_image =
      (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
   PFNEGLQUERYDMABUFFORMATSEXTPROC query_formats =
      (PFNEGLQUERYDMABUFFORMATSEXTPROC)eglGetProcAddress(
         "eglQueryDmaBufFormatsEXT");
   PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers =
      (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress(
         "eglQueryDmaBufModifiersEXT");
   PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture =
      (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress(
         "glEGLImageTargetTexture2DOES");
   if (!create_image || !destroy_image || !query_formats || !query_modifiers ||
       !image_target_texture ||
       !format_and_modifier_supported(display, query_formats,
                                      query_modifiers)) {
      fprintf(stderr, "EGL dma-buf entrypoint/format fallback detected\n");
      return 1;
   }

   if (!eglBindAPI(EGL_OPENGL_ES_API))
      return 1;
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   const char *gl_extensions = (const char *)glGetString(GL_EXTENSIONS);
   if (!gl_extensions ||
       !has_extension(gl_extensions, "GL_OES_EGL_image")) {
      fprintf(stderr, "GL_OES_EGL_image missing\n");
      return 1;
   }

   const bool native_control_ok = native_texture_control(noop);

   const int drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   struct gbm_device *gbm = drm_fd >= 0 ? gbm_create_device(drm_fd) : NULL;
   const uint32_t use = GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR;
   struct gbm_bo *original =
      gbm ? gbm_bo_create(gbm, WIDTH, HEIGHT, GBM_FORMAT_ARGB8888, use) : NULL;
   if (!original || gbm_bo_get_modifier(original) != DRM_FORMAT_MOD_LINEAR ||
       gbm_bo_get_plane_count(original) != 1)
      return 1;

   struct guarded_pixels first, second, cleared;
   fill_raw(&first, first_bgra);
   fill_raw(&second, second_bgra);
   fill_raw(&cleared, clear_bgra);
   if (!map_write(original, &first) || !map_read(original, &first))
      return 1;

   int fd = gbm_bo_get_fd(original);
   if (fd < 0 || !(fcntl(fd, F_GETFD, 0) & FD_CLOEXEC))
      return 1;
   const uint32_t stride = gbm_bo_get_stride(original);
   const uint32_t offset = gbm_bo_get_offset(original, 0);
   const uint64_t modifier = gbm_bo_get_modifier(original);
   struct gbm_import_fd_modifier_data import = {
      .width = WIDTH,
      .height = HEIGHT,
      .format = GBM_FORMAT_ARGB8888,
      .num_fds = 1,
      .fds = {fd},
      .strides = {(int)stride},
      .offsets = {(int)offset},
      .modifier = modifier,
   };
   struct gbm_bo *retained =
      gbm_bo_import(gbm, GBM_BO_IMPORT_FD_MODIFIER, &import, use);
   if (!retained)
      return 1;

   const EGLint attrs[] = {
      EGL_WIDTH, WIDTH,
      EGL_HEIGHT, HEIGHT,
      EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
      EGL_DMA_BUF_PLANE0_FD_EXT, fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)offset,
      EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)modifier,
      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(modifier >> 32),
      EGL_NONE,
   };
   EGLImageKHR image = create_image(display, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
   if (image == EGL_NO_IMAGE_KHR)
      return 1;

   EGLint invalid_attrs[ARRAY_SIZE(attrs)];
   memcpy(invalid_attrs, attrs, sizeof(attrs));
   invalid_attrs[7] = -1;
   EGLImageKHR invalid = create_image(display, EGL_NO_CONTEXT,
                                      EGL_LINUX_DMA_BUF_EXT, NULL,
                                      invalid_attrs);
   if (invalid != EGL_NO_IMAGE_KHR || eglGetError() == EGL_SUCCESS)
      return 1;

   GLuint texture, fbo;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   image_target_texture(GL_TEXTURE_2D, image);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !gl_ok("EGLImage bind"))
      return 1;

   /* The texture retains imported storage after both source handles vanish. */
   if (!destroy_image(display, image))
      return 1;
   close(fd);
   gbm_bo_destroy(original);
   original = NULL;
   if (!map_read(retained, &first))
      return 1;
   bool gl_pixels_ok = gl_read("first CPU upload", first_rgba, first_bgra,
                               NULL, noop);

   glClearColor(68.0f / 255.0f, 85.0f / 255.0f, 102.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();
   if (!gl_ok("first imported clear") ||
       !map_read(retained, noop ? &first : &cleared))
      return 1;
   gl_pixels_ok &= gl_read("first GPU clear", clear_rgba, clear_bgra,
                            first_rgba, noop);

   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) ||
       !eglMakeCurrent(display, surface, surface, context) ||
       !glIsTexture(texture) || !glIsFramebuffer(fbo))
      return 1;
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   if (!map_write(retained, &second))
      return 1;
   gl_pixels_ok &= gl_read("second CPU upload", second_rgba, second_bgra,
                            clear_rgba, noop);

   glClearColor(68.0f / 255.0f, 85.0f / 255.0f, 102.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();
   if (!map_read(retained, noop ? &second : &cleared))
      return 1;
   gl_pixels_ok &= gl_read("second GPU clear", clear_rgba, clear_bgra,
                            second_rgba, noop);

   if (!native_control_ok || !gl_pixels_ok ||
       !guard_ok(first.before) || !guard_ok(first.after) ||
       !guard_ok(second.before) || !guard_ok(second.after) ||
       !guard_ok(cleared.before) || !guard_ok(cleared.after))
      return 1;

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   gbm_bo_destroy(retained);
   gbm_device_destroy(gbm);
   close(drm_fd);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL06 EGLImage dma-buf lifecycle exact; no-op pixels skipped"
             : "GL06 EGLImage dma-buf bidirectional pixels and lifetime exact");
   return 0;
}
