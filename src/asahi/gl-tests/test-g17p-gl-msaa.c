/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <fcntl.h>
#include <gbm.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH  8
#define HEIGHT 8

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
      fprintf(stderr, "MSAA shader compile failed: %s\n", log);
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
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = vec4(1.0, 0.5, 0.25, 1.0); }\n";
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
      fprintf(stderr, "MSAA program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static void
draw_masked(GLuint program, unsigned samples, bool clear)
{
   glViewport(0, 0, WIDTH, HEIGHT);
   if (clear) {
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
   }
   glEnable(GL_SAMPLE_COVERAGE);
   glSampleCoverage(1.0f / samples, GL_FALSE);
   glUseProgram(program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_SAMPLE_COVERAGE);
}

/* Ordinary GLES multisample FBOs exercise the API resolve independently of
 * WSI. The software control additionally proves the exact averaged colour.
 */
static bool
run_offscreen_sample_count(GLuint program, unsigned samples, bool partial,
                           bool noop)
{
   GLuint source, target, source_fb, target_fb;
   glGenRenderbuffers(1, &source);
   glBindRenderbuffer(GL_RENDERBUFFER, source);
   glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8,
                                    WIDTH, HEIGHT);

   glGenTextures(1, &target);
   glBindTexture(GL_TEXTURE_2D, target);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);

   glGenFramebuffers(1, &target_fb);
   glBindFramebuffer(GL_FRAMEBUFFER, target_fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D, target, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;
   if (partial) {
      glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
   }

   glGenFramebuffers(1, &source_fb);
   glBindFramebuffer(GL_FRAMEBUFFER, source_fb);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, source);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;
   draw_masked(program, samples, true);

   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fb);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, source_fb);
   const unsigned extent = partial ? WIDTH / 2 : WIDTH;
   glBlitFramebuffer(0, 0, extent, extent, 0, 0, extent, extent,
                     GL_COLOR_BUFFER_BIT, GL_NEAREST);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, target_fb);
   glFinish();

   uint8_t pixel[4] = {0};
   const unsigned observed = partial ? 1 : WIDTH / 2;
   glReadPixels(observed, observed, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   const uint8_t expected2[4] = {128, 64, 32, 128};
   const uint8_t expected4[4] = {64, 32, 16, 64};
   const uint8_t *expected = samples == 2 ? expected2 : expected4;
   const bool exact = noop || !memcmp(pixel, expected, sizeof(pixel));
   if (!exact)
      fprintf(stderr,
              "%ux MSAA pixel {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
              samples, pixel[0], pixel[1], pixel[2], pixel[3], expected[0],
              expected[1], expected[2], expected[3]);

   uint8_t preserved[4] = {0};
   if (partial) {
      glReadPixels(WIDTH - 1, HEIGHT - 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                   preserved);
      static const uint8_t blue[4] = {0, 0, 255, 255};
      if (!noop && memcmp(preserved, blue, sizeof(blue))) {
         fprintf(stderr, "%ux partial MSAA overwrote {%u,%u,%u,%u}\n",
                 samples, preserved[0], preserved[1], preserved[2],
                 preserved[3]);
         return false;
      }
   }

   glDeleteFramebuffers(1, &target_fb);
   glDeleteFramebuffers(1, &source_fb);
   glDeleteTextures(1, &target);
   glDeleteRenderbuffers(1, &source);
   return gl_ok("software MSAA render/resolve") && exact;
}

static EGLConfig
choose_config(EGLDisplay display, EGLint surface_type, unsigned samples,
              bool require_argb8888)
{
   const EGLint attributes[] = {
      EGL_SURFACE_TYPE, surface_type,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_SAMPLE_BUFFERS, samples ? 1 : 0,
      EGL_SAMPLES, samples,
      EGL_NONE,
   };
   EGLConfig configs[64];
   EGLint count = 0;
   if (!eglChooseConfig(display, attributes, configs, 64, &count))
      return NULL;

   for (EGLint i = 0; i < count; ++i) {
      EGLint actual_samples = 0, visual = 0;
      if (!eglGetConfigAttrib(display, configs[i], EGL_SAMPLES,
                              &actual_samples) ||
          actual_samples != (EGLint)samples)
         continue;
      if (require_argb8888 &&
          (!eglGetConfigAttrib(display, configs[i], EGL_NATIVE_VISUAL_ID,
                               &visual) ||
           visual != GBM_FORMAT_ARGB8888))
         continue;
      return configs[i];
   }
   return NULL;
}

/* A GBM window surface is the real Gallium WSI contract: the loader supplies
 * a transient multisample cbuf and pipe_framebuffer_state::resolve names the
 * single-sample BO that eglSwapBuffers presents.
 */
static bool
run_shim_sample_count(EGLDisplay display, struct gbm_device *device,
                      unsigned samples)
{
   EGLConfig config = choose_config(display, EGL_WINDOW_BIT, samples, true);
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLContext context = config ? eglCreateContext(display, config,
                                                   EGL_NO_CONTEXT,
                                                   context_attributes)
                               : EGL_NO_CONTEXT;
   struct gbm_surface *gbm = gbm_surface_create(
      device, WIDTH, HEIGHT, GBM_FORMAT_ARGB8888, GBM_BO_USE_RENDERING);
   EGLSurface surface = gbm && context != EGL_NO_CONTEXT
                           ? eglCreateWindowSurface(
                                display, config, (EGLNativeWindowType)gbm, NULL)
                           : EGL_NO_SURFACE;
   if (!config || !gbm || surface == EGL_NO_SURFACE ||
       !eglMakeCurrent(display, surface, surface, context)) {
      fprintf(stderr, "no exact %ux RGBA8 EGL/GBM window config: %#x\n",
              samples, eglGetError());
      return false;
   }

   GLint default_samples = 0;
   glGetIntegerv(GL_SAMPLES, &default_samples);
   GLuint vao, program = make_program();
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   bool ok = program && default_samples == (GLint)samples;
   for (unsigned frame = 0; ok && frame < 2; ++frame) {
      draw_masked(program, samples, frame == 0);
      ok = gl_ok("G17P default MSAA draw") &&
           eglSwapBuffers(display, surface);
      struct gbm_bo *front = ok ? gbm_surface_lock_front_buffer(gbm) : NULL;
      ok &= front != NULL;
      if (front)
         gbm_surface_release_buffer(gbm, front);
   }
   if (ok)
      ok = run_offscreen_sample_count(program, samples, false, true) &&
           run_offscreen_sample_count(program, samples, true, true);

   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(display, surface);
   eglDestroyContext(display, context);
   gbm_surface_destroy(gbm);
   return ok;
}

static bool
run_offscreen(EGLDisplay display, bool noop)
{
   EGLConfig config = choose_config(display, EGL_PBUFFER_BIT, 0, false);
   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface = config ? eglCreatePbufferSurface(
                                  display, config, surface_attributes)
                               : EGL_NO_SURFACE;
   EGLContext context = config ? eglCreateContext(display, config,
                                                   EGL_NO_CONTEXT,
                                                   context_attributes)
                               : EGL_NO_CONTEXT;
   if (!config || surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context)) {
      fprintf(stderr, "offscreen EGL context creation failed: %#x\n",
              eglGetError());
      return false;
   }

   GLuint vao, program = make_program();
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   const bool exact =
      program && run_offscreen_sample_count(program, 2, false, noop) &&
      run_offscreen_sample_count(program, 4, false, noop) &&
      run_offscreen_sample_count(program, 2, true, noop) &&
      run_offscreen_sample_count(program, 4, true, noop);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   return exact;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   int drm_fd = -1;
   struct gbm_device *device = NULL;
   EGLDisplay display = EGL_NO_DISPLAY;

   if (noop) {
      drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
      device = drm_fd >= 0 ? gbm_create_device(drm_fd) : NULL;
      PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
         (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
            "eglGetPlatformDisplayEXT");
      display = device && get_platform_display
                   ? get_platform_display(EGL_PLATFORM_GBM_KHR, device, NULL)
                   : EGL_NO_DISPLAY;
   } else {
      display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   }

   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return EXIT_FAILURE;

   const bool exact = noop ? run_shim_sample_count(display, device, 2) &&
                                run_shim_sample_count(display, device, 4)
                           : run_offscreen(display, false);
   eglTerminate(display);
   if (device)
      gbm_device_destroy(device);
   if (drm_fd >= 0)
      close(drm_fd);
   if (!exact)
      return EXIT_FAILURE;
   printf("g17p-gl-msaa: PASS — 2x/4x %s\n",
          noop ? "GBM, full, and partial offscreen resolves submitted"
               : "resolve pixels exact");
   return EXIT_SUCCESS;
}
