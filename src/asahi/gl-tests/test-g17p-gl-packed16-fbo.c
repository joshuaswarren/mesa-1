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

#define WIDTH  8
#define HEIGHT 8

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
      fprintf(stderr, "packed16 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

static GLuint
make_program(void)
{
   static const char vertex[] =
      "#version 300 es\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
   static const char fragment[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform vec4 source_colour;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = source_colour; }\n";

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
      fprintf(stderr, "packed16 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }

   return program;
}

static bool
near_pixel(const uint8_t actual[4], const uint8_t expected[4])
{
   for (unsigned c = 0; c < 4; ++c) {
      const int delta = (int)actual[c] - (int)expected[c];
      if (delta < -1 || delta > 1)
         return false;
   }

   return true;
}

static bool
exercise_format(GLenum internal_format, const char *name, GLuint program,
                GLint colour, const uint8_t expected_drawn[4],
                const uint8_t expected_preserved[4], bool noop)
{
   GLuint texture = 0, fbo = 0;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, WIDTH, HEIGHT);

   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "%s framebuffer is incomplete\n", name);
      return false;
   }

   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_SCISSOR_TEST);
   glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();

   /* A second batch must reload the packed target before the scissored draw
    * and store all four F16 tile lanes back through the packed PBE. */
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, WIDTH / 2, HEIGHT);
   glUseProgram(program);
   glUniform4f(colour, 0.8f, 0.6f, 0.4f, 0.0f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   glDisable(GL_SCISSOR_TEST);

   uint8_t drawn[4] = {0}, preserved[4] = {0};
   glReadPixels(1, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, drawn);
   glReadPixels(WIDTH - 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                preserved);
   const GLenum error = glGetError();

   const bool ok = error == GL_NO_ERROR &&
      (noop || (near_pixel(drawn, expected_drawn) &&
                near_pixel(preserved, expected_preserved)));
   if (!ok) {
      fprintf(stderr,
              "%s pixels {%u,%u,%u,%u} {%u,%u,%u,%u}, error %#x\n",
              name, drawn[0], drawn[1], drawn[2], drawn[3], preserved[0],
              preserved[1], preserved[2], preserved[3], error);
   }

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   return ok;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0;
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
   EGLint count = 0;
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

   GLuint program = make_program();
   GLuint vao = 0;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   const GLint colour = glGetUniformLocation(program, "source_colour");

   static const uint8_t rgba4_drawn[4] = {204, 153, 102, 0};
   static const uint8_t rgba4_preserved[4] = {51, 102, 153, 255};
   static const uint8_t rgb5a1_drawn[4] = {206, 156, 99, 0};
   static const uint8_t rgb5a1_preserved[4] = {49, 99, 156, 255};
   const bool rgba4 = exercise_format(GL_RGBA4, "RGBA4", program, colour,
                                      rgba4_drawn, rgba4_preserved, noop);
   const bool rgb5a1 = exercise_format(GL_RGB5_A1, "RGB5_A1", program,
                                       colour, rgb5a1_drawn,
                                       rgb5a1_preserved, noop);

   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (!rgba4 || !rgb5a1)
      return EXIT_FAILURE;

   printf("GL packed16 FBO clear, reload, draw, and store exact%s\n",
          noop ? "; no-op pixels skipped" : "");
   return EXIT_SUCCESS;
}
