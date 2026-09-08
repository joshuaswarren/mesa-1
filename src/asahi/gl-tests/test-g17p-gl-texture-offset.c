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
      fprintf(stderr, "texture-offset shader compile failed: %s\n", log);
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
      "void main() {\n"
      "  vec4 filtered = textureOffset(source, vec2(0.5), ivec2(1, -1));\n"
      "  vec4 fetched = texelFetchOffset(source, ivec2(1), 0, "
      "ivec2(1, -1));\n"
      "  colour = vec4(filtered.rgb, fetched.r);\n"
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
      fprintf(stderr, "texture-offset program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
read_pixel(const char *phase, unsigned x, unsigned y,
           const uint8_t expected[4], bool noop)
{
   uint8_t pixel[4] = {0};
   glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   if (!gl_ok(phase))
      return false;
   if (noop || !memcmp(pixel, expected, sizeof(pixel))) {
      printf("texture-path phase %s %s\n", phase,
             noop ? "submitted" : "exact");
      return true;
   }

   fprintf(stderr,
           "texture-path phase %s pixel {%u,%u,%u,%u}, expected "
           "{%u,%u,%u,%u}\n",
           phase, pixel[0], pixel[1], pixel[2], pixel[3], expected[0],
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
      return 1;

   const EGLint config_attributes[] = {
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
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;

   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION,
      3,
      EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   GLuint program = make_program();
   if (!program)
      return 1;

   static const uint8_t pixels[3 * 3 * 4] = {
      10,  20,  30,  255, 40,  50,  60,  255, 17,  93,  201, 255,
      70,  80,  90,  255, 100, 110, 120, 255, 130, 140, 150, 255,
      160, 170, 180, 255, 190, 200, 210, 255, 220, 230, 240, 255,
   };
   GLuint textures[3];
   glGenTextures(3, textures);
   const GLuint texture = textures[0];
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 3, 3, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                pixels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   /* Keep upload/readback separate from the texture consumed by the shader,
    * so completing the diagnostic read cannot make the sample pass by
    * accident. */
   glBindTexture(GL_TEXTURE_2D, textures[1]);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 3, 3, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, pixels);

   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          textures[1], 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 1;
   static const uint8_t uploaded[4] = {17, 93, 201, 255};
   if (!read_pixel("uploaded-source", 2, 0, uploaded, noop))
      return 1;

   glBindTexture(GL_TEXTURE_2D, textures[2]);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          textures[2], 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glBindTexture(GL_TEXTURE_2D, texture);
   glUniform1i(glGetUniformLocation(program, "source"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   static const uint8_t sampled[4] = {17, 93, 201, 17};
   if (!read_pixel("sample-to-user-fbo", WIDTH / 2, HEIGHT / 2, sampled,
                   noop))
      return 1;

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   if (!read_pixel("sample-to-default", WIDTH / 2, HEIGHT / 2, sampled, noop))
      return 1;

   glDeleteFramebuffers(1, &fbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(3, textures);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts(noop ? "G17P WebGL2 texture upload/FBO/default sampling submitted"
             : "G17P WebGL2 texture upload/FBO/default sampling exact");
   return 0;
}
