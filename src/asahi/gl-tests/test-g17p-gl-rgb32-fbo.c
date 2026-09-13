/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH  8
#define HEIGHT 8

static bool
gl_ok(const char *where)
{
   GLenum error = glGetError();
   if (error == GL_NO_ERROR)
      return true;

   fprintf(stderr, "%s: GL error %#x\n", where, error);
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
      fprintf(stderr, "RGB32 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

static GLuint
make_program(void)
{
   static const char vertex[] =
      "#version 330 core\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
   static const char fragment[] =
      "#version 330 core\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = vec4(0.75, 0.5, 0.25, 0.125); }\n";

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
      fprintf(stderr, "RGB32 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }

   return program;
}

static bool
pixel_exact(const char *phase, unsigned x, const float expected[4], bool noop)
{
   float actual[4] = {0};
   glReadPixels(x, HEIGHT / 2, 1, 1, GL_RGBA, GL_FLOAT, actual);
   if (!gl_ok(phase))
      return false;

   if (noop)
      return true;

   for (unsigned c = 0; c < 4; ++c) {
      if (fabsf(actual[c] - expected[c]) > 0.0001f) {
         fprintf(stderr,
                 "%s pixel {%g,%g,%g,%g}, expected {%g,%g,%g,%g}\n",
                 phase, actual[0], actual[1], actual[2], actual[3], expected[0],
                 expected[1], expected[2], expected[3]);
         return false;
      }
   }

   return true;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return EXIT_FAILURE;

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count = 0;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return EXIT_FAILURE;

   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   GLuint program = make_program();
   GLuint texture = 0, fbo = 0, vao = 0;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB32F, WIDTH, HEIGHT);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D, texture, 0);
   if (!program ||
       glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !gl_ok("RGB32F framebuffer"))
      return EXIT_FAILURE;

   glViewport(0, 0, WIDTH, HEIGHT);
   glClearColor(0.125f, 0.25f, 0.5f, 0.75f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();

   /* Force a new batch to reload all four physical lanes, then modify only
    * half the RGB target and store the untouched half back through the PBE. */
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, WIDTH / 2, HEIGHT);
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glUseProgram(program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   glDisable(GL_SCISSOR_TEST);

   static const float drawn[4] = {0.75f, 0.5f, 0.25f, 1.0f};
   static const float preserved[4] = {0.125f, 0.25f, 0.5f, 1.0f};
   const bool exact =
      pixel_exact("RGB32F draw", 1, drawn, noop) &&
      pixel_exact("RGB32F preserve", WIDTH - 2, preserved, noop);

   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (!exact)
      return EXIT_FAILURE;

   printf("GL RGB32F clear, reload, draw, and store exact%s\n",
          noop ? "; no-op pixels skipped" : "");
   return EXIT_SUCCESS;
}
