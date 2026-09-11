/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * QuickShell/Qt glyph discriminator. Render a linearly filtered R8 coverage
 * atlas into a premultiplied BGRA8 client attachment, then sample and
 * source-over composite that client attachment onto an opaque target.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH       32
#define HEIGHT      8
#define ATLAS_WIDTH 16
#define DATA_BYTES  (WIDTH * HEIGHT * 4)
#define GUARD_SIZE  64
#define GUARD_BYTE  0xa5

#ifndef GL_BGRA8_EXT
#define GL_BGRA8_EXT 0x93A1
#endif

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
      fprintf(stderr, "GL42 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *fragment)
{
   static const char vertex[] =
      "#version 300 es\n"
      "const vec2 p[3]=vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),"
      "vec2(-1.0,3.0));\n"
      "out vec2 uv;\n"
      "void main(){gl_Position=vec4(p[gl_VertexID],0.0,1.0);"
      "uv=p[gl_VertexID]*0.5+0.5;}\n";
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment);
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
      fprintf(stderr, "GL42 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
guards_ok(const struct guarded_pixels *pixels)
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (pixels->before[i] != GUARD_BYTE ||
          pixels->after[i] != GUARD_BYTE)
         return false;
   }
   return true;
}

static uint8_t
coverage_for_x(unsigned x)
{
   if (x == 0)
      return 0;
   if (x == WIDTH - 1)
      return 255;
   return (x & 2) ? 191 : 64;
}

static uint8_t
round_unorm(float value)
{
   unsigned byte = (unsigned)(value * 255.0f + 0.5f);
   return byte > 255 ? 255 : (uint8_t)byte;
}

static void
expected_pixel(unsigned x, bool final, uint8_t out[4])
{
   const float alpha = coverage_for_x(x) / 255.0f;
   const float tint[3] = {1.0f, 0.25f, 0.75f};
   const float background[3] = {32.0f / 255.0f, 128.0f / 255.0f,
                                224.0f / 255.0f};
   for (unsigned c = 0; c < 3; ++c) {
      float value = tint[c] * alpha;
      if (final)
         value += background[c] * (1.0f - alpha);
      out[c] = round_unorm(value);
   }
   out[3] = final ? 255 : coverage_for_x(x);
}

static bool
near_byte(uint8_t actual, uint8_t expected)
{
   return actual + 2 >= expected && actual <= expected + 2;
}

static bool
read_expected(const char *phase, bool final, bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   GLenum error = glGetError();
   if (!guards_ok(&pixels)) {
      fprintf(stderr, "GL42 %s readback guard changed\n", phase);
      return false;
   }
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         uint8_t expected[4];
         expected_pixel(x, final, expected);
         const uint8_t *got = &pixels.data[(y * WIDTH + x) * 4];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_byte(got[c], expected[c])) {
               fprintf(stderr,
                       "GL42 %s (%u,%u) got %02x%02x%02x%02x, "
                       "expected %02x%02x%02x%02x (+/-2), error %#x\n",
                       phase, x, y, got[0], got[1], got[2], got[3],
                       expected[0], expected[1], expected[2], expected[3],
                       error);
               return false;
            }
         }
      }
   }
   fprintf(stderr, "GL42 %s: exact R8 glyph coverage (+/-2)\n", phase);
   return error == GL_NO_ERROR;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return EXIT_FAILURE;

   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return EXIT_FAILURE;
   const EGLint surface_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
   if (!extensions ||
       !has_extension(extensions, "GL_EXT_texture_format_BGRA8888")) {
      fprintf(stderr, "GL42 BGRA8 render attachment support missing\n");
      return EXIT_FAILURE;
   }

   static const char glyph_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D atlas;"
      "layout(location=0) out vec4 colour;"
      "void main(){float a=texture(atlas,uv).r;"
      "colour=vec4(vec3(1.0,0.25,0.75)*a,a);}";
   static const char sample_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D client_tex;"
      "layout(location=0) out vec4 colour;"
      "void main(){colour=texture(client_tex,uv);}";
   GLuint glyph_program = make_program(glyph_fragment);
   GLuint sample_program = make_program(sample_fragment);
   if (!glyph_program || !sample_program)
      return EXIT_FAILURE;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   static const uint8_t atlas_pixels[ATLAS_WIDTH] = {
      0, 255, 0, 255, 0, 255, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
   };
   GLuint atlas;
   glGenTextures(1, &atlas);
   glBindTexture(GL_TEXTURE_2D, atlas);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, ATLAS_WIDTH, 1, 0, GL_RED,
                GL_UNSIGNED_BYTE, atlas_pixels);

   GLuint client_texture, client_fbo;
   glGenTextures(1, &client_texture);
   glBindTexture(GL_TEXTURE_2D, client_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_BGRA8_EXT, WIDTH, HEIGHT);
   glGenFramebuffers(1, &client_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, client_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          client_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "GL42 BGRA8 client FBO incomplete\n");
      return EXIT_FAILURE;
   }

   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_DITHER);
   glDisable(GL_BLEND);
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(glyph_program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, atlas);
   glUniform1i(glGetUniformLocation(glyph_program, "atlas"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   if (!read_expected("R8 atlas into BGRA8 client attachment", false, noop))
      return EXIT_FAILURE;

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glClearColor(32.0f / 255.0f, 128.0f / 255.0f, 224.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glUseProgram(sample_program);
   glBindTexture(GL_TEXTURE_2D, client_texture);
   glUniform1i(glGetUniformLocation(sample_program, "client_tex"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   if (!read_expected("R8 glyph client composite immediate", true, noop))
      return EXIT_FAILURE;
   sleep(3);
   if (!read_expected("R8 glyph client composite retained", true, noop))
      return EXIT_FAILURE;

   glDeleteFramebuffers(1, &client_fbo);
   glDeleteTextures(1, &client_texture);
   glDeleteTextures(1, &atlas);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(glyph_program);
   glDeleteProgram(sample_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL42 R8 glyph pipeline submitted; pixels skipped"
             : "GL42 R8 glyph atlas/BGRA8 client/composite exact "
               "immediately/3s");
   return EXIT_SUCCESS;
}
