/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VALUES 64
#define GUARD_WORDS 8

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
make_program(void)
{
   static const char source[] =
      "#version 310 es\n"
      "layout(local_size_x=8) in;\n"
      "layout(std430, binding=0) buffer Output { uint value[]; };\n"
      "void main() { uint i=gl_GlobalInvocationID.x; value[i]=i*3u+7u; }\n";
   GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
   const GLchar *source_ptr = source;
   glShaderSource(shader, 1, &source_ptr, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "compute compile failed: %s\n", log);
      return 0;
   }

   GLuint program = glCreateProgram();
   glAttachShader(program, shader);
   glLinkProgram(program);
   glDeleteShader(shader);
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "compute link failed: %s\n", log);
      return 0;
   }
   return program;
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

   const EGLint config_attributes[] = {
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
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   const EGLint surface_attributes[] = {
      EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("EGL %d.%d; GL %s; renderer %s\n", major, minor,
          (const char *)glGetString(GL_VERSION), renderer);
   if (noop && (!renderer || !strstr(renderer, "Apple A18 Pro") ||
                strstr(renderer, "zink")))
      return 1;

   GLuint program = make_program();
   if (!program)
      return 1;

   uint32_t initial[VALUES];
   for (unsigned i = 0; i < VALUES; ++i)
      initial[i] = UINT32_C(0xdead0000) | i;
   GLuint buffer;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initial), initial,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUES / 8, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   glFinish();
   if (!gl_ok("dispatch"))
      return 1;

   uint32_t guarded[VALUES + 2 * GUARD_WORDS];
   for (unsigned i = 0; i < sizeof(guarded) / sizeof(guarded[0]); ++i)
      guarded[i] = UINT32_C(0xa5a5a5a5);
   const void *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                      sizeof(initial), GL_MAP_READ_BIT);
   if (!map)
      return 1;
   memcpy(&guarded[GUARD_WORDS], map, sizeof(initial));
   if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER) || !gl_ok("readback"))
      return 1;

   if (!noop) {
      for (unsigned i = 0; i < VALUES; ++i) {
         if (guarded[GUARD_WORDS + i] != i * 3u + 7u) {
            fprintf(stderr, "value %u is %#x\n", i,
                    guarded[GUARD_WORDS + i]);
            return 1;
         }
      }
   }
   for (unsigned i = 0; i < GUARD_WORDS; ++i) {
      if (guarded[i] != UINT32_C(0xa5a5a5a5) ||
          guarded[GUARD_WORDS + VALUES + i] != UINT32_C(0xa5a5a5a5))
         return 1;
   }

   puts("G17P GL compute PASS");
   return 0;
}
