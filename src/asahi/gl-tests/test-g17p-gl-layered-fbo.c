/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <GLES3/gl32.h>

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
      fprintf(stderr, "layered-FBO shader compile failed: %s\n", log);
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
      "void main() { colour = vec4(0.75, 0.25, 0.5, 1.0); }\n";
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
      fprintf(stderr, "layered-FBO program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
read_layer(GLuint texture, unsigned layer, const uint8_t expected[4],
           bool noop)
{
   glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0,
                             layer);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;

   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   if (!gl_ok("layered-FBO readback"))
      return false;
   if (noop || !memcmp(pixel, expected, sizeof(pixel)))
      return true;

   fprintf(stderr,
           "layer %u pixel {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
           layer, pixel[0], pixel[1], pixel[2], pixel[3], expected[0],
           expected[1], expected[2], expected[3]);
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
      EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 2,
      EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   GLuint texture, framebuffer;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
   glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, WIDTH, HEIGHT, 2);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

   /* Whole-resource attachment: clear must execute the layered BG/EOT path
    * and store both slices through the native Gen9 2D-array PBE descriptor. */
   glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return EXIT_FAILURE;
   glClearColor(17.0f / 255.0f, 93.0f / 255.0f, 201.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();

   static const uint8_t cleared[4] = {17, 93, 201, 255};
   if (!read_layer(texture, 0, cleared, noop) ||
       !read_layer(texture, 1, cleared, noop))
      return EXIT_FAILURE;

   /* A draw to the whole-resource attachment reloads both slices through the
    * layered texture table and stores them through the array PBE. With no
    * shader-written layer, rasterization targets slice 0 and slice 1 must be
    * preserved by the layered background/EOT pair. */
   GLuint vao, program = make_program();
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   static const uint8_t drawn[4] = {191, 64, 128, 255};
   bool exact = program && read_layer(texture, 0, drawn, noop) &&
                read_layer(texture, 1, cleared, noop);

   /* A single nonzero slice deliberately collapses back to the ordinary 2D
    * PBE with its base address advanced to layer 1. */
   glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0,
                             1);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   exact &= read_layer(texture, 1, drawn, noop) &&
            read_layer(texture, 0, drawn, noop);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   glDeleteFramebuffers(1, &framebuffer);
   glDeleteTextures(1, &texture);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (!exact)
      return EXIT_FAILURE;
   puts(noop ? "G17P layered array clear/reload/layer draw submitted"
             : "G17P layered array clear/reload/layer pixels exact");
   return EXIT_SUCCESS;
}
