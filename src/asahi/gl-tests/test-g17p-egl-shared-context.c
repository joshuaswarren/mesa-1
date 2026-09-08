/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "drm-uapi/drm_fourcc.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static bool
modifier_contract(EGLDisplay display, bool require_apple)
{
   PFNEGLQUERYDMABUFMODIFIERSEXTPROC query =
      (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress(
         "eglQueryDmaBufModifiersEXT");
   EGLint count = 0;
   if (!query || !query(display, DRM_FORMAT_ARGB8888, 0, NULL, NULL, &count) ||
       count <= 0)
      return !require_apple;

   EGLuint64KHR *modifiers = calloc(count, sizeof(*modifiers));
   EGLBoolean *external = calloc(count, sizeof(*external));
   if (!modifiers || !external ||
       !query(display, DRM_FORMAT_ARGB8888, count, modifiers, external,
              &count)) {
      free(external);
      free(modifiers);
      return false;
   }

   bool compressed = false, tiled = false, linear = false, ok = true;
   for (EGLint i = 0; i < count; ++i) {
      if (modifiers[i] == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED) {
         compressed = true;
         ok &= external[i] == EGL_TRUE;
      } else if (modifiers[i] == DRM_FORMAT_MOD_APPLE_GPU_TILED) {
         tiled = true;
         ok &= external[i] == EGL_FALSE;
      } else if (modifiers[i] == DRM_FORMAT_MOD_LINEAR) {
         linear = true;
         ok &= external[i] == EGL_FALSE;
      }
   }
   free(external);
   free(modifiers);

   if (require_apple && (!compressed || !tiled || !linear || !ok)) {
      fprintf(stderr,
              "G17P dma-buf modifier contract: compressed=%u tiled=%u "
              "linear=%u class_ok=%u\n",
              compressed, tiled, linear, ok);
      return false;
   }
   return true;
}

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint status = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
   if (!status) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "shared-context shader failed: %s\n", log);
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
      "uniform sampler2D source;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = texture(source, vec2(0.5)); }\n";
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
   GLint status = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &status);
   if (!status) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "shared-context link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static EGLSurface
make_surface(EGLDisplay display, EGLConfig config, unsigned width,
             unsigned height)
{
   const EGLint attrs[] = {
      EGL_WIDTH, (EGLint)width, EGL_HEIGHT, (EGLint)height, EGL_NONE,
   };
   return eglCreatePbufferSurface(display, config, attrs);
}

static bool
sample_default(GLuint program, GLuint texture, unsigned width, unsigned height,
               const uint8_t expected[4], bool noop)
{
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glViewport(0, 0, width, height);
   glUseProgram(program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);
   glUniform1i(glGetUniformLocation(program, "source"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   uint8_t pixel[4] = {0};
   glReadPixels(width / 2, height / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   if (!gl_ok("shared-context sample"))
      return false;
   if (noop || !memcmp(pixel, expected, 4))
      return true;
   fprintf(stderr,
           "shared-context pixel {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
           pixel[0], pixel[1], pixel[2], pixel[3], expected[0], expected[1],
           expected[2], expected[3]);
   return false;
}

static bool
clear_texture(GLuint texture, const float colour[4])
{
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;
   glClearColor(colour[0], colour[1], colour[2], colour[3]);
   glClear(GL_COLOR_BUFFER_BIT);
   glDeleteFramebuffers(1, &fbo);
   return gl_ok("shared texture clear");
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return 1;
   if (!modifier_contract(display, noop))
      return 1;

   PFNEGLCREATESYNCKHRPROC create_sync =
      (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
   PFNEGLDESTROYSYNCKHRPROC destroy_sync =
      (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
   PFNEGLWAITSYNCKHRPROC wait_sync =
      (PFNEGLWAITSYNCKHRPROC)eglGetProcAddress("eglWaitSyncKHR");
   if (!create_sync || !destroy_sync || !wait_sync)
      return 1;

   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE,
      EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE,
      EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE,
      8,
      EGL_GREEN_SIZE,
      8,
      EGL_BLUE_SIZE,
      8,
      EGL_ALPHA_SIZE,
      8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION,
      3,
      EGL_NONE,
   };
   EGLContext a =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   EGLContext b = eglCreateContext(display, config, a, context_attrs);
   EGLSurface surface_a = make_surface(display, config, 32, 32);
   EGLSurface surface_b = make_surface(display, config, 48, 24);
   if (a == EGL_NO_CONTEXT || b == EGL_NO_CONTEXT ||
       surface_a == EGL_NO_SURFACE || surface_b == EGL_NO_SURFACE ||
       !eglMakeCurrent(display, surface_a, surface_a, a))
      return 1;

   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 32, 32);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   static const float red_f[4] = {1.0f, 0.0f, 0.0f, 1.0f};
   static const uint8_t red[4] = {255, 0, 0, 255};
   if (!clear_texture(texture, red_f))
      return 1;
   EGLSyncKHR first = create_sync(display, EGL_SYNC_FENCE_KHR, NULL);
   glFlush();
   if (first == EGL_NO_SYNC_KHR || !gl_ok("producer fence"))
      return 1;

   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) ||
       !eglDestroySurface(display, surface_a) ||
       !eglMakeCurrent(display, surface_b, surface_b, b) ||
       !wait_sync(display, first, 0))
      return 1;
   if (!destroy_sync(display, first) || !eglDestroyContext(display, a))
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   GLuint program = make_program();
   if (!program || !glIsTexture(texture) ||
       !sample_default(program, texture, 48, 24, red, noop))
      return 1;

   static const float blue_f[4] = {0.0f, 0.0f, 1.0f, 1.0f};
   static const uint8_t blue[4] = {0, 0, 255, 255};
   if (!clear_texture(texture, blue_f))
      return 1;
   EGLSyncKHR second = create_sync(display, EGL_SYNC_FENCE_KHR, NULL);
   glFlush();
   EGLContext c = eglCreateContext(display, config, b, context_attrs);
   EGLSurface surface_c = make_surface(display, config, 17, 31);
   if (second == EGL_NO_SYNC_KHR || c == EGL_NO_CONTEXT ||
       surface_c == EGL_NO_SURFACE ||
       !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) ||
       !eglDestroySurface(display, surface_b) ||
       !eglDestroyContext(display, b) ||
       !eglMakeCurrent(display, surface_c, surface_c, c) ||
       !wait_sync(display, second, 0) || !destroy_sync(display, second) ||
       !glIsTexture(texture) || !glIsProgram(program) ||
       !sample_default(program, texture, 17, 31, blue, noop))
      return 1;

   glDeleteProgram(program);
   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(1, &texture);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(display, surface_c);
   eglDestroyContext(display, c);
   eglTerminate(display);
   puts(
      noop
         ? "GL05 shared contexts, fence handoff and resized surfaces submitted"
         : "GL05 shared contexts, fence handoff and resized pixels exact");
   return 0;
}
