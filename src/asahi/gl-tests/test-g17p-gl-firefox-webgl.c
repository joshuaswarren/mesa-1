/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Offline GLES analogue of Firefox's public WebGL/WebGL2 paths.  This is not
 * a substitute for running Firefox; it keeps the exact shader/draw failures
 * local and reproducible when a browser is unavailable.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 16
#define HEIGHT 16

static bool noop;

static void
fail_gl(const char *where)
{
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "F54 %s: GL error %#x\n", where, error);
      exit(1);
   }
}

static GLuint
compile(GLenum stage, const char *name, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint status = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
   if (!status) {
      char log[8192] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "F54 %s compile failed: %s\n", name, log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
link_program(const char *name, const char *vs_source, const char *fs_source)
{
   GLuint vs = compile(GL_VERTEX_SHADER, name, vs_source);
   GLuint fs = compile(GL_FRAGMENT_SHADER, name, fs_source);
   if (!vs || !fs)
      return 0;
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "position");
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   GLint status = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &status);
   if (!status) {
      char log[8192] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "F54 %s link failed: %s\n", name, log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static void
check_pixel(const char *name, const uint8_t expected[4])
{
   glFinish();
   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA,
                GL_UNSIGNED_BYTE, pixel);
   fail_gl(name);
   if (!noop && memcmp(pixel, expected, 4) != 0) {
      fprintf(stderr,
              "F54 %s pixel {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
              name, pixel[0], pixel[1], pixel[2], pixel[3], expected[0],
              expected[1], expected[2], expected[3]);
      exit(1);
   }
}

static void
make_geometry(GLuint *vbo, GLuint *ibo)
{
   static const GLfloat positions[] = {-1.0f, -1.0f, 3.0f, -1.0f,
                                        -1.0f, 3.0f};
   static const GLushort indices[] = {2, 0, 1};
   glGenBuffers(1, vbo);
   glBindBuffer(GL_ARRAY_BUFFER, *vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   glGenBuffers(1, ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                GL_STATIC_DRAW);
}

static void
run_webgl1(void)
{
   static const char vs[] =
      "attribute vec2 position;\n"
      "uniform vec4 tint;\n"
      "varying vec4 colour;\n"
      "void main() {\n"
      "  gl_Position=vec4(position,0.0,1.0);\n"
      "  int signedValue=int(position.x*8.0);\n"
      "  float keep=signedValue < 32768 ? 1.0 : 0.0;\n"
      "  colour=tint*keep;\n"
      "}\n";
   static const char fs[] =
      "precision highp float;\n"
      "varying vec4 colour;\n"
      "void main() {\n"
      "  if (colour.a < 0.5) discard;\n"
      "  gl_FragColor=colour;\n"
      "}\n";
   static const uint8_t expected[4] = {64, 128, 192, 255};
   GLuint program = link_program("WebGL1 baseline", vs, fs);
   if (!program)
      exit(1);
   GLuint vbo, ibo;
   make_geometry(&vbo, &ibo);
   glUseProgram(program);
   glUniform4f(glGetUniformLocation(program, "tint"), 64.0f / 255.0f,
               128.0f / 255.0f, 192.0f / 255.0f, 1.0f);
   glViewport(0, 0, WIDTH, HEIGHT);
   glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);
   check_pixel("WebGL1 indexed shader", expected);
   glDeleteBuffers(1, &ibo);
   glDeleteBuffers(1, &vbo);
   glDeleteProgram(program);
}

static void
run_webgl2(void)
{
   static const char vs[] =
      "#version 300 es\n"
      "layout(location=0) in vec2 position;\n"
      "uniform highp sampler2D source;\n"
      "flat out highp vec4 sampled;\n"
      "void main() {\n"
      "  gl_Position=vec4(position,0.0,1.0);\n"
      "  int id=gl_VertexID;\n"
      "  ivec2 tc=ivec2((id*3+2)&3,(id*2+1)&3);\n"
      "  vec4 a=texelFetch(source,tc,0);\n"
      "  vec4 b=textureLod(source,(vec2(tc)+0.5)/4.0,0.0);\n"
      "  sampled=(a+b)*0.5;\n"
      "}\n";
   static const char fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "precision highp int;\n"
      "flat in highp vec4 sampled;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() {\n"
      "  uint lane=(uint(gl_FragCoord.x)^uint(gl_FragCoord.y))&1u;\n"
      "  if (sampled.a < 0.5) discard;\n"
      "  colour=sampled+vec4(float(lane)*0.0);\n"
      "}\n";
   static const uint8_t expected[4] = {17, 123, 201, 255};
   GLuint program = link_program("WebGL2 indexed vertex texture", vs, fs);
   if (!program)
      exit(1);

   uint8_t texels[4 * 4 * 4];
   for (unsigned i = 0; i < 16; ++i) {
      texels[4 * i + 0] = 3 + i * 7;
      texels[4 * i + 1] = 9 + i * 5;
      texels[4 * i + 2] = 211 - i * 3;
      texels[4 * i + 3] = 255;
   }
   const unsigned selected[] = {1 * 4 + 2, 3 * 4 + 1, 1 * 4 + 0};
   for (unsigned i = 0; i < 3; ++i)
      memcpy(&texels[4 * selected[i]], expected, 4);

   GLuint texture;
   glGenTextures(1, &texture);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   GLuint vao, vbo, ibo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   make_geometry(&vbo, &ibo);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "source"), 0);
   glViewport(0, 0, WIDTH, HEIGHT);

   /* Repeated link/use/delete pressure mirrors Firefox's shader churn while
    * staying bounded. The retained program is drawn after every temporary. */
   for (unsigned iteration = 0; iteration < 12; ++iteration) {
      GLuint temporary = link_program("WebGL2 compiler churn", vs, fs);
      if (!temporary)
         exit(1);
      glDeleteProgram(temporary);
      glUseProgram(program);
      glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);
      check_pixel("WebGL2 indexed vertex texture", expected);
   }

   glDeleteBuffers(1, &ibo);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(1, &texture);
   glDeleteProgram(program);
}

int
main(void)
{
   noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "F54 EGL initialization failed %#x\n", eglGetError());
      return 1;
   }
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count = 0;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config,
                                                 surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                         context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   printf("F54 GL_VERSION: %s\n", glGetString(GL_VERSION));
   printf("F54 GL_RENDERER: %s\n", glGetString(GL_RENDERER));
   run_webgl1();
   run_webgl2();
   fail_gl("completion");

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts(noop ? "F54 PASS: G17P Firefox WebGL submission validated; pixels skipped"
             : "F54 PASS: Firefox WebGL/WebGL2 pixels exact");
   return 0;
}
