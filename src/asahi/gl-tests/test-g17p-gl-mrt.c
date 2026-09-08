/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
      fprintf(stderr, "MRT shader compile failed: %s\n", log);
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
      "layout(location=0) out vec4 colour0;\n"
      "layout(location=1) out vec4 colour1;\n"
      "void main() {\n"
      "  colour0 = vec4(0.25, 0.125, 0.5, 0.25);\n"
      "  colour1 = vec4(0.125, 0.25, 0.0625, 0.5);\n"
      "}\n";

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
      fprintf(stderr, "MRT program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
check_pixel(const char *phase, GLenum attachment, unsigned x, unsigned y,
            const uint8_t expected[4], bool noop)
{
   uint8_t actual[4] = {0};
   glReadBuffer(attachment);
   glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, actual);
   if (!gl_ok(phase))
      return false;
   if (noop || !memcmp(actual, expected, sizeof(actual)))
      return true;

   fprintf(stderr,
           "%s attachment %#x pixel {%u,%u,%u,%u}, expected "
           "{%u,%u,%u,%u}\n",
           phase, attachment, actual[0], actual[1], actual[2], actual[3],
           expected[0], expected[1], expected[2], expected[3]);
   return false;
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

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return EXIT_FAILURE;

   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   GLuint program = make_program();
   if (!program)
      return EXIT_FAILURE;

   GLuint textures[2];
   glGenTextures(2, textures);
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   for (unsigned i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, textures[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                             GL_TEXTURE_2D, textures[i], 0);
   }
   const GLenum draw_buffers[2] = {
      GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
   };
   glDrawBuffers(2, draw_buffers);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !gl_ok("MRT framebuffer"))
      return EXIT_FAILURE;

   /* Exercise the exact dual-clear program with independently addressed
    * values, then force it to retire before the preservation draw. */
   const GLfloat clear0[4] = {0.0625f, 0.125f, 0.1875f, 0.25f};
   const GLfloat clear1[4] = {0.3125f, 0.375f, 0.4375f, 0.5f};
   glClearBufferfv(GL_COLOR, 0, clear0);
   glClearBufferfv(GL_COLOR, 1, clear1);
   glFinish();
   static const uint8_t clear0_pixel[4] = {16, 32, 48, 64};
   static const uint8_t clear1_pixel[4] = {80, 96, 112, 128};
   if (!check_pixel("dual clear RT0", GL_COLOR_ATTACHMENT0, WIDTH - 1,
                    HEIGHT / 2, clear0_pixel, noop) ||
       !check_pixel("dual clear RT1", GL_COLOR_ATTACHMENT1, WIDTH - 1,
                    HEIGHT / 2, clear1_pixel, noop))
      return EXIT_FAILURE;

   /* A scissored two-output draw requires both attachments to reload the
    * clear outside the scissor and stores both distinct packed results inside
    * it.  This covers clear, reload, and EOT/PBE wiring through GLES 3. */
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glViewport(0, 0, WIDTH, HEIGHT);
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, WIDTH / 2, HEIGHT);
   glUseProgram(program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   static const uint8_t drawn0[4] = {64, 32, 128, 64};
   static const uint8_t drawn1[4] = {32, 64, 16, 128};
   const bool exact =
      check_pixel("MRT draw RT0", GL_COLOR_ATTACHMENT0, 1, HEIGHT / 2,
                  drawn0, noop) &&
      check_pixel("MRT draw RT1", GL_COLOR_ATTACHMENT1, 1, HEIGHT / 2,
                  drawn1, noop) &&
      check_pixel("MRT preserve RT0", GL_COLOR_ATTACHMENT0, WIDTH - 1,
                  HEIGHT / 2, clear0_pixel, noop) &&
      check_pixel("MRT preserve RT1", GL_COLOR_ATTACHMENT1, WIDTH - 1,
                  HEIGHT / 2, clear1_pixel, noop);

   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(2, textures);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (!exact)
      return EXIT_FAILURE;
   printf("g17p-gl-mrt: PASS — dual RGBA8 clear/reload/store %s\n",
          noop ? "submitted" : "pixels exact");
   return EXIT_SUCCESS;
}
