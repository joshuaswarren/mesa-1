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

#define WIDTH 64
#define HEIGHT 64

static bool
gl_ok(const char *step)
{
   GLenum error = glGetError();
   if (error == GL_NO_ERROR)
      return true;

   fprintf(stderr, "%s: GL error %#x\n", step, error);
   while (glGetError() != GL_NO_ERROR)
      ;
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
      fprintf(stderr, "shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(bool varying)
{
   static const char varying_vs[] =
      "#version 300 es\n"
      "layout(location=0) in vec4 a_pos;\n"
      "out vec4 v_colour;\n"
      "void main() { gl_Position = a_pos; v_colour = a_pos * 0.25 + 0.5; }\n";
   static const char varying_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec4 v_colour;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = v_colour; }\n";
   static const char solid_vs[] =
      "#version 300 es\n"
      "layout(location=0) in vec4 a_pos;\n"
      "void main() { gl_Position = a_pos; }\n";
   static const char solid_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = vec4(0.25, 0.5, 0.75, 1.0); }\n";

   GLuint vs = compile_shader(GL_VERTEX_SHADER, varying ? varying_vs : solid_vs);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, varying ? varying_fs : solid_fs);
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
      fprintf(stderr, "program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static char *
read_shader(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   if (fseek(f, 0, SEEK_END) || ftell(f) < 0) {
      fclose(f);
      return NULL;
   }
   const long length = ftell(f);
   rewind(f);
   char *source = malloc((size_t)length + 1);
   if (!source || fread(source, 1, (size_t)length, f) != (size_t)length) {
      free(source);
      fclose(f);
      return NULL;
   }
   source[length] = '\0';
   fclose(f);
   return source;
}

/* Compile and link already-preprocessed GLSL ES sources. Passing the real
 * Hyprland vertex shader followed by all thirteen fragment shaders makes the
 * GL link result an API-level compiler oracle: a fail-closed AGX refusal must
 * now make glLinkProgram() fail instead of being discovered at first draw. */
static bool
compile_shader_corpus(int argc, char **argv)
{
   char *vs_source = read_shader(argv[1]);
   if (!vs_source) {
      fprintf(stderr, "could not read corpus vertex shader %s\n", argv[1]);
      return false;
   }
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   free(vs_source);
   if (!vs)
      return false;

   bool ok = true;
   for (int i = 2; i < argc; ++i) {
      char *fs_source = read_shader(argv[i]);
      GLuint fs = fs_source ? compile_shader(GL_FRAGMENT_SHADER, fs_source) : 0;
      free(fs_source);
      if (!fs) {
         fprintf(stderr, "corpus fragment compile failed: %s\n", argv[i]);
         ok = false;
         continue;
      }

      GLuint program = glCreateProgram();
      glAttachShader(program, vs);
      glAttachShader(program, fs);
      glLinkProgram(program);
      glDeleteShader(fs);
      GLint linked = GL_FALSE;
      glGetProgramiv(program, GL_LINK_STATUS, &linked);
      if (!linked) {
         char log[4096] = {0};
         glGetProgramInfoLog(program, sizeof(log), NULL, log);
         fprintf(stderr, "corpus fragment link failed: %s: %s\n", argv[i],
                 log);
         ok = false;
      } else {
         printf("shader-corpus: %s\n", argv[i]);
      }
      glDeleteProgram(program);
   }
   glDeleteShader(vs);
   return ok && gl_ok("shader corpus");
}

static bool
finish_case(const char *name)
{
   setenv("ASAHI_SHIM_G17P_CASE", name, 1);
   glFinish();
   unsetenv("ASAHI_SHIM_G17P_CASE");
   if (!gl_ok(name))
      return false;
   printf("state-matrix: %s\n", name);
   return true;
}

int
main(int argc, char **argv)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return 1;

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
      return 1;

   const EGLint pbuffer_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, pbuffer_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   if (argc > 1) {
      if (argc < 3) {
         fprintf(stderr, "usage: %s <vertex.glsl> <fragment.glsl>...\n",
                 argv[0]);
         return 2;
      }
      const bool ok = compile_shader_corpus(argc, argv);
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroyContext(display, context);
      eglDestroySurface(display, surface);
      eglTerminate(display);
      return ok ? 0 : 1;
   }

   GLuint varying_program = make_program(true);
   GLuint solid_program = make_program(false);
   if (!varying_program || !solid_program)
      return 1;

   static const float vertices[][4] = {
      {-0.8f, -0.8f, 0.25f, 1.0f},
      { 0.8f, -0.8f, 0.50f, 1.0f},
      { 0.8f,  0.8f, 0.75f, 1.0f},
      {-0.8f,  0.8f, 1.00f, 1.0f},
   };
   static const uint8_t indices_u8[] = {0, 1, 2};
   static const uint16_t indices_u16[] = {
      0, 1, 2, UINT16_MAX, 0, 2, 3,
   };
   static const uint32_t indices_u32[] = {0, 1, 2};
   static const uint32_t arrays_indirect[] = {3, 1, 0, 0};
   static const uint32_t elements_indirect[] = {3, 1, 0, 0, 0};

   GLuint vao, vbo, ibo[3], indirect_bo[2], colour, depth_stencil, fbo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), NULL);
   glEnableVertexAttribArray(0);
   glGenBuffers(3, ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[0]);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_u8), indices_u8,
                GL_STATIC_DRAW);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[1]);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_u16), indices_u16,
                GL_STATIC_DRAW);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[2]);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_u32), indices_u32,
                GL_STATIC_DRAW);
   glGenBuffers(2, indirect_bo);
   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_bo[0]);
   glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(arrays_indirect),
                arrays_indirect, GL_STATIC_DRAW);
   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_bo[1]);
   glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(elements_indirect),
                elements_indirect, GL_STATIC_DRAW);
   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

   glGenTextures(1, &colour);
   glBindTexture(GL_TEXTURE_2D, colour);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenRenderbuffers(1, &depth_stencil);
   glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          colour, 0);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, depth_stencil);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !gl_ok("setup"))
      return 1;

   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   if (!finish_case("clear-only"))
      return 1;

   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(varying_program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDrawArrays(GL_TRIANGLES, 0, 3); /* Same-state body-only record. */
   if (!finish_case("nonindexed unchanged-state body"))
      return 1;
   glDrawArrays(GL_LINES, 0, 2);
   if (!finish_case("nonindexed line topology"))
      return 1;

   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[0]);
   glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, NULL);
   if (!finish_case("indexed u8 without restart"))
      return 1;

   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[1]);
   glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
   glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_SHORT, NULL);
   if (!finish_case("indexed u16 topology and restart"))
      return 1;
   glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);

   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[2]);
   glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);
   if (!finish_case("indexed u32 without restart"))
      return 1;

   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_bo[0]);
   glDrawArraysIndirect(GL_TRIANGLES, NULL);
   if (!finish_case("indirect nonindexed"))
      return 1;

   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo[1]);
   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_bo[1]);
   glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, NULL);
   if (!finish_case("indirect indexed"))
      return 1;
   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

   glViewport(3, 5, 48, 40);
   glEnable(GL_SCISSOR_TEST);
   glScissor(7, 9, 23, 19);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
   if (!finish_case("viewport and scissor"))
      return 1;
   glDisable(GL_SCISSOR_TEST);
   glViewport(0, 0, WIDTH, HEIGHT);

   glEnable(GL_CULL_FACE);
   glCullFace(GL_FRONT);
   glFrontFace(GL_CW);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   if (!finish_case("cull and front face"))
      return 1;
   glDisable(GL_CULL_FACE);
   glFrontFace(GL_CCW);

   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_GEQUAL);
   glDepthMask(GL_TRUE);
   glEnable(GL_STENCIL_TEST);
   glStencilFunc(GL_NOTEQUAL, 3, 0x7f);
   glStencilOp(GL_REPLACE, GL_INCR_WRAP, GL_DECR_WRAP);
   glStencilMask(0x3f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   if (!finish_case("depth and stencil"))
      return 1;
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);

   glEnable(GL_BLEND);
   glBlendEquation(GL_FUNC_ADD);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   if (!finish_case("blend"))
      return 1;
   glDisable(GL_BLEND);

   glUseProgram(solid_program);
   glDrawArrays(GL_POINTS, 0, 1);
   if (!finish_case("fragment program without varyings"))
      return 1;

   glEnable(GL_RASTERIZER_DISCARD);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   if (!finish_case("rasterizer discard"))
      return 1;
   glDisable(GL_RASTERIZER_DISCARD);

   glUseProgram(varying_program);
   for (unsigned i = 0; i < 5000; ++i) {
      glViewport(i & 1, 0, WIDTH - (i & 1), HEIGHT);
      glDrawArrays(GL_TRIANGLES, 0, 3);
   }
   if (!finish_case("linked stream with state re-emission"))
      return 1;

   glDeleteFramebuffers(1, &fbo);
   glDeleteRenderbuffers(1, &depth_stencil);
   glDeleteTextures(1, &colour);
   glDeleteBuffers(2, indirect_bo);
   glDeleteBuffers(3, ibo);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(solid_program);
   glDeleteProgram(varying_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts("G17P real Gallium state matrix completed");
   return 0;
}
