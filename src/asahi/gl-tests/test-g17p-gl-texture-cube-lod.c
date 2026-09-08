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
      fprintf(stderr, "cube-LOD shader compile failed: %s\n", log);
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
      "precision highp samplerCube;\n"
      "uniform samplerCube source;\n"
      "uniform float lod;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() {\n"
      "  vec4 fixed_level = textureLod(source, vec3(1.0, 0.0, 0.0), 1.0);\n"
      "  vec4 dynamic_level = textureLod(source, vec3(-1.0, 0.0, 0.0), "
      "lod);\n"
      "  colour = vec4(fixed_level.r, dynamic_level.g, fixed_level.b, 1.0);\n"
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
      fprintf(stderr, "cube-LOD program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static void
fill(uint8_t *pixels, unsigned width, unsigned height,
     const uint8_t colour[4])
{
   for (unsigned i = 0; i < width * height; ++i) {
      pixels[4 * i + 0] = colour[0];
      pixels[4 * i + 1] = colour[1];
      pixels[4 * i + 2] = colour[2];
      pixels[4 * i + 3] = colour[3];
   }
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
      return 1;

   GLuint program = make_program();
   if (!program)
      return 1;

   static const uint8_t zero[4] = {0, 0, 0, 255};
   static const uint8_t positive_x[4] = {17, 20, 201, 255};
   static const uint8_t negative_x[4] = {30, 123, 40, 255};
   uint8_t level0[4 * 4 * 4];
   uint8_t level1[2 * 2 * 4];
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
   glTexStorage2D(GL_TEXTURE_CUBE_MAP, 2, GL_RGBA8, 4, 4);
   for (unsigned face = 0; face < 6; ++face) {
      fill(level0, 4, 4, zero);
      const uint8_t *colour = face == 0 ? positive_x
                              : face == 1 ? negative_x
                                          : zero;
      fill(level1, 2, 2, colour);
      glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, 0, 0, 4, 4,
                      GL_RGBA, GL_UNSIGNED_BYTE, level0);
      glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 1, 0, 0, 2, 2,
                      GL_RGBA, GL_UNSIGNED_BYTE, level1);
   }
   glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                   GL_NEAREST_MIPMAP_NEAREST);
   glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "source"), 0);
   glUniform1f(glGetUniformLocation(program, "lod"), 1.0f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   if (!gl_ok("cube-LOD draw/readback"))
      return 1;
   if (!noop && (pixel[0] != 17 || pixel[1] != 123 || pixel[2] != 201 ||
                 pixel[3] != 255)) {
      fprintf(stderr, "cube-LOD pixel {%u,%u,%u,%u}\n", pixel[0], pixel[1],
              pixel[2], pixel[3]);
      return 1;
   }

   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(1, &texture);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts(noop ? "G17P WebGL2 cube immediate/register LOD samples submitted"
             : "G17P WebGL2 cube immediate/register LOD pixel exact");
   return 0;
}
