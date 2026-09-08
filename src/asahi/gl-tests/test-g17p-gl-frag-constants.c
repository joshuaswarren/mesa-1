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
      fprintf(stderr, "fragment-constant shader compile failed: %s\n", log);
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
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
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
      fprintf(stderr, "fragment-constant program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
draw_and_read(GLuint program, GLint uniform, uint8_t pixel[4])
{
   glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(program);
   if (uniform >= 0)
      glUniform4f(uniform, 0.125f, 0.375f, 0.625f, 1.0f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "fragment-constant GL error %#x\n", error);
      return false;
   }
   return true;
}

static bool
draw_texture_blend(GLuint program, GLint sampler, GLuint texture,
                   GLenum src_rgb, GLenum dst_rgb, GLenum src_alpha,
                   GLenum dst_alpha, const float blend[4], uint8_t pixel[4])
{
   glDisable(GL_BLEND);
   glClearColor(16.0f / 255.0f, 32.0f / 255.0f, 48.0f / 255.0f,
                64.0f / 255.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);
   glUniform1i(sampler, 0);
   glBlendColor(blend[0], blend[1], blend[2], blend[3]);
   glBlendFuncSeparate(src_rgb, dst_rgb, src_alpha, dst_alpha);
   glEnable(GL_BLEND);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   glDisable(GL_BLEND);

   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "textured fragment-constant GL error %#x\n", error);
      return false;
   }
   return true;
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
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) ||
       count != 1)
      return EXIT_FAILURE;
   const EGLint surface_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   static const char literal_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = vec4(0.125, 0.375, 0.625, 1.0); }\n";
   static const char uniform_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform vec4 source_colour;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = source_colour; }\n";
   static const char texture_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform sampler2D source_texture;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = texture(source_texture, vec2(0.5)); }\n";
   GLuint literal = make_program(literal_fs);
   GLuint uniform = make_program(uniform_fs);
   GLuint textured = make_program(texture_fs);
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glViewport(0, 0, WIDTH, HEIGHT);
   const GLint location = glGetUniformLocation(uniform, "source_colour");
   const GLint sampler = glGetUniformLocation(textured, "source_texture");
   GLuint texture;
   static const uint8_t source_pixel[4] = {64, 128, 192, 128};
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA,
                   GL_UNSIGNED_BYTE, source_pixel);
   uint8_t literal_pixel[4] = {0};
   uint8_t uniform_pixel[4] = {0};
   uint8_t constant_pixel[4] = {0};
   uint8_t inverse_pixel[4] = {0};
   uint8_t alpha_pixel[4] = {0};
   /* Avoid exact x.5 UNORM boundaries: native and software conversions may
    * legally choose opposite tie directions, while literal and uniform
    * shader sources must still produce byte-identical pixels. */
   static const uint8_t expected[4] = {32, 96, 159, 255};
   static const float half[4] = {0.5f, 0.5f, 0.5f, 0.5f};
   static const float quarter_half[4] = {0.25f, 0.25f, 0.25f, 0.5f};
   static const float three_quarters[4] = {0.75f, 0.75f, 0.75f, 0.75f};
   static const uint8_t expected_constant[4] = {32, 64, 96, 64};
   static const uint8_t expected_inverse[4] = {28, 56, 84, 96};
   static const uint8_t expected_alpha[4] = {52, 104, 156, 112};
   bool ok = literal && uniform && textured && location >= 0 && sampler >= 0 &&
             draw_and_read(literal, -1, literal_pixel) &&
             draw_and_read(uniform, location, uniform_pixel) &&
             draw_texture_blend(textured, sampler, texture,
                                GL_CONSTANT_COLOR, GL_ZERO,
                                GL_CONSTANT_ALPHA, GL_ZERO, half,
                                constant_pixel) &&
             draw_texture_blend(textured, sampler, texture,
                                GL_CONSTANT_COLOR,
                                GL_ONE_MINUS_CONSTANT_COLOR,
                                GL_CONSTANT_ALPHA,
                                GL_ONE_MINUS_CONSTANT_ALPHA, quarter_half,
                                inverse_pixel) &&
             draw_texture_blend(textured, sampler, texture,
                                GL_CONSTANT_ALPHA,
                                GL_ONE_MINUS_CONSTANT_ALPHA,
                                GL_CONSTANT_ALPHA,
                                GL_ONE_MINUS_CONSTANT_ALPHA, three_quarters,
                                alpha_pixel);
   if (!noop &&
       (memcmp(literal_pixel, uniform_pixel, sizeof(literal_pixel)) ||
        memcmp(literal_pixel, expected, sizeof(literal_pixel)))) {
      fprintf(stderr,
              "literal pixel {%u,%u,%u,%u}, uniform pixel {%u,%u,%u,%u}, "
              "expected {%u,%u,%u,%u}\n",
              literal_pixel[0], literal_pixel[1], literal_pixel[2],
              literal_pixel[3], uniform_pixel[0], uniform_pixel[1],
              uniform_pixel[2], uniform_pixel[3], expected[0], expected[1],
              expected[2], expected[3]);
      ok = false;
   }
   if (!noop &&
       (memcmp(constant_pixel, expected_constant, sizeof(constant_pixel)) ||
        memcmp(inverse_pixel, expected_inverse, sizeof(inverse_pixel)) ||
        memcmp(alpha_pixel, expected_alpha, sizeof(alpha_pixel)))) {
      fprintf(stderr,
              "textured blend pixels {%u,%u,%u,%u} {%u,%u,%u,%u} "
              "{%u,%u,%u,%u}\n",
              constant_pixel[0], constant_pixel[1], constant_pixel[2],
              constant_pixel[3], inverse_pixel[0], inverse_pixel[1],
              inverse_pixel[2], inverse_pixel[3], alpha_pixel[0],
              alpha_pixel[1], alpha_pixel[2], alpha_pixel[3]);
      ok = false;
   }

   glDeleteTextures(1, &texture);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(literal);
   glDeleteProgram(uniform);
   glDeleteProgram(textured);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   if (!ok)
      return EXIT_FAILURE;
   puts(noop ? "G17P literal/uniform and textured blend draws submitted"
             : "G17P literal/uniform and textured blend pixels exact");
   return EXIT_SUCCESS;
}
