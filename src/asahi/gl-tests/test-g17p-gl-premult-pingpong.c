/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * QuickShell/Qt-style transparent client rendering discriminator:
 * premultiplied draws into RGBA8, then sample and premultiplied-composite that
 * client image onto an opaque target. This is intentionally independent of
 * Hyprland blur and external-memory import.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH      8
#define HEIGHT     4
#define DATA_BYTES (WIDTH * HEIGHT * 4)
#define GUARD_SIZE 64
#define GUARD_BYTE 0xa5

struct guarded_pixels {
   uint8_t before[GUARD_SIZE];
   uint8_t data[DATA_BYTES];
   uint8_t after[GUARD_SIZE];
};

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
      fprintf(stderr, "GL40 shader compile failed: %s\n", log);
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
      "const vec2 p[3] = vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),"
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
      fprintf(stderr, "GL40 program link failed: %s\n", log);
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

static bool
near_byte(uint8_t actual, uint8_t expected)
{
   return actual + 1 >= expected && actual <= expected + 1;
}

static bool
read_expected(const char *phase, const uint8_t expected[WIDTH][4], bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   GLenum error = glGetError();
   if (!guards_ok(&pixels)) {
      fprintf(stderr, "GL40 %s readback guard changed\n", phase);
      return false;
   }
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const uint8_t *got = &pixels.data[(y * WIDTH + x) * 4];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_byte(got[c], expected[x][c])) {
               fprintf(stderr,
                       "GL40 %s (%u,%u) got %02x%02x%02x%02x, "
                       "expected %02x%02x%02x%02x (+/-1), error %#x\n",
                       phase, x, y, got[0], got[1], got[2], got[3],
                       expected[x][0], expected[x][1], expected[x][2],
                       expected[x][3], error);
               return false;
            }
         }
      }
   }
   fprintf(stderr, "GL40 %s: exact premultiplied regions (+/-1)\n", phase);
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
      EGL_WIDTH, WIDTH,
      EGL_HEIGHT, HEIGHT,
      EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   static const char solid_fragment[] =
      "#version 300 es\n"
      "precision highp float; uniform vec4 colour;"
      "layout(location=0) out vec4 out_colour;"
      "void main(){out_colour=colour;}";
   static const char sample_fragment[] =
      "#version 300 es\n"
      "precision highp float; in vec2 uv; uniform sampler2D client_tex;"
      "layout(location=0) out vec4 out_colour;"
      "void main(){out_colour=texture(client_tex,uv);}";
   GLuint solid_program = make_program(solid_fragment);
   GLuint sample_program = make_program(sample_fragment);
   if (!solid_program || !sample_program)
      return EXIT_FAILURE;
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   GLuint client_texture, client_fbo;
   glGenTextures(1, &client_texture);
   glBindTexture(GL_TEXTURE_2D, client_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &client_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, client_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          client_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "GL40 transparent client FBO incomplete\n");
      return EXIT_FAILURE;
   }

   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_DITHER);
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(solid_program);
   GLint colour = glGetUniformLocation(solid_program, "colour");
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glEnable(GL_SCISSOR_TEST);

   /* First glyph edge: premultiplied cyan, alpha 1/2. */
   glScissor(0, 0, 6, HEIGHT);
   glUniform4f(colour, 0.0f, 0.25f, 0.5f, 0.5f);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   /* Second edge overlaps four columns: premultiplied orange, alpha 1/2. */
   glScissor(2, 0, 6, HEIGHT);
   glUniform4f(colour, 0.5f, 0.125f, 0.0f, 0.5f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_SCISSOR_TEST);
   glFinish();

   static const uint8_t client_expected[WIDTH][4] = {
      {0, 64, 128, 128}, {0, 64, 128, 128},
      {128, 64, 64, 191}, {128, 64, 64, 191},
      {128, 64, 64, 191}, {128, 64, 64, 191},
      {128, 32, 0, 128}, {128, 32, 0, 128},
   };
   if (!read_expected("transparent client attachment", client_expected, noop))
      return EXIT_FAILURE;

   /* Hyprland-equivalent second stage: sample the client image and composite
    * it over an opaque workbuffer with the same premultiplied blend factors. */
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(sample_program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, client_texture);
   glUniform1i(glGetUniformLocation(sample_program, "client_tex"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   static const uint8_t final_expected[WIDTH][4] = {
      {32, 128, 223, 255}, {32, 128, 223, 255},
      {143, 96, 112, 255}, {143, 96, 112, 255},
      {143, 96, 112, 255}, {143, 96, 112, 255},
      {159, 96, 96, 255}, {159, 96, 96, 255},
   };
   if (!read_expected("sampled client composite immediate", final_expected,
                      noop))
      return EXIT_FAILURE;
   sleep(3);
   if (!read_expected("sampled client composite retained", final_expected,
                      noop))
      return EXIT_FAILURE;

   glDeleteFramebuffers(1, &client_fbo);
   glDeleteTextures(1, &client_texture);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(solid_program);
   glDeleteProgram(sample_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL40 premultiplied client ping-pong submitted; pixels skipped"
             : "GL40 premultiplied client attachment/sample/composite exact "
               "immediately/3s");
   return EXIT_SUCCESS;
}
