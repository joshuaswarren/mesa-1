/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Qt Quick's gray distance-field text material through a GLES QRhi context.
 * QRhi maps RED_OR_ALPHA8 to legacy GL_ALPHA outside desktop core profile and
 * selects distancefieldtext_a.frag, which samples .a. An asymmetric raw-sample
 * pass first proves that the stored byte is exposed as (0, 0, 0, alpha), then
 * the exact Qt distance-field shader, UBO, indexed geometry, and premultiplied
 * blend are checked immediately and after three seconds.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GLYPH_WIDTH 16
#define GLYPHS      4
#define WIDTH       (GLYPH_WIDTH * GLYPHS)
#define HEIGHT      8
#define DATA_BYTES  (WIDTH * HEIGHT * 4)
#define GUARD_SIZE  64
#define GUARD_BYTE  0xa5

#ifndef GL_BGRA8_EXT
#define GL_BGRA8_EXT 0x93A1
#endif

struct guarded_pixels {
   uint8_t before[GUARD_SIZE];
   uint8_t data[DATA_BYTES];
   uint8_t after[GUARD_SIZE];
};

struct qt_uniforms {
   float matrix[16];
   float texture_scale[2];
   float vec2_padding[2];
   float color[4];
   float alpha_min;
   float alpha_max;
   float block_padding[2];
};

struct vertex {
   float position[2];
   float texcoord[2];
};

static const uint8_t atlas[GLYPH_WIDTH] = {
   0, 64, 96, 112, 120, 124, 128, 132,
   136, 144, 160, 192, 224, 240, 248, 255,
};

static bool
has_extension(const char *extensions, const char *name)
{
   const size_t length = strlen(name);
   const char *match = extensions;
   while ((match = strstr(match, name))) {
      if ((match == extensions || match[-1] == ' ') &&
          (match[length] == '\0' || match[length] == ' '))
         return true;
      match += length;
   }
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
      fprintf(stderr, "GL45 shader compile failed: %s\n", log);
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
      "layout(location=0) in vec4 vCoord;"
      "layout(location=1) in vec2 tCoord;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "out vec2 sampleCoord;"
      "void main(){sampleCoord=tCoord*textureScale;"
      "gl_Position=matrix*vCoord;}";
   static const char fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=color*smoothstep(alphaMin,alphaMax,"
      "texture(_qt_texture,sampleCoord).a);}";
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
      fprintf(stderr, "GL45 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static GLuint
make_sentinel_program(void)
{
   static const char vertex[] =
      "#version 300 es\n"
      "layout(location=0) in vec4 vCoord;"
      "layout(location=1) in vec2 tCoord;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "out vec2 sampleCoord;"
      "void main(){sampleCoord=tCoord*textureScale;"
      "gl_Position=matrix*vCoord;}";
   static const char fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=texture(_qt_texture,sampleCoord);}";
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
      fprintf(stderr, "GL45 sentinel link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
guards_ok(const struct guarded_pixels *pixels)
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (pixels->before[i] != GUARD_BYTE ||
          pixels->after[i] != GUARD_BYTE)
         return false;
   }
   return true;
}

static uint8_t
to_unorm(float value)
{
   const unsigned byte = (unsigned)(fminf(fmaxf(value, 0.0f), 1.0f) *
                                    255.0f + 0.5f);
   return (uint8_t)byte;
}

static float
smoothstep_reference(float edge0, float edge1, float x)
{
   const float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static void
expected_pixel(unsigned x, uint8_t out[4])
{
   static const float background[4] = {
      24.0f / 255.0f, 28.0f / 255.0f, 38.0f / 255.0f, 1.0f,
   };
   static const float color[4] = {0.50f, 0.25f, 0.75f, 0.75f};
   const float coverage = smoothstep_reference(
      0.44f, 0.56f, atlas[x % GLYPH_WIDTH] / 255.0f);
   const float source_alpha = color[3] * coverage;
   for (unsigned c = 0; c < 3; ++c)
      out[c] = to_unorm(color[c] * coverage +
                        background[c] * (1.0f - source_alpha));
   out[3] = 255;
}

static bool
near_byte(uint8_t actual, uint8_t want)
{
   return actual + 2 >= want && actual <= want + 2;
}

static bool
read_alpha_sentinel(const char *phase, bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   GLenum error = glGetError();
   if (!guards_ok(&pixels)) {
      fprintf(stderr, "GL45 %s readback guard changed\n", phase);
      return false;
   }
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const uint8_t want[4] = {0, 0, 0, atlas[x % GLYPH_WIDTH]};
         const uint8_t *got = &pixels.data[(y * WIDTH + x) * 4];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_byte(got[c], want[c])) {
               fprintf(stderr,
                       "GL45 %s (%u,%u) got %02x%02x%02x%02x, expected "
                       "%02x%02x%02x%02x (+/-2), error %#x\n",
                       phase, x, y, got[0], got[1], got[2], got[3],
                       want[0], want[1], want[2], want[3], error);
               return false;
            }
         }
      }
   }
   fprintf(stderr, "GL45 %s: exact GL_ALPHA (0,0,0,A) pixels (+/-2)\n",
           phase);
   return error == GL_NO_ERROR;
}

static bool
read_expected(const char *phase, bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   GLenum error = glGetError();
   if (!guards_ok(&pixels)) {
      fprintf(stderr, "GL45 %s readback guard changed\n", phase);
      return false;
   }
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         uint8_t want[4];
         expected_pixel(x, want);
         const uint8_t *got = &pixels.data[(y * WIDTH + x) * 4];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_byte(got[c], want[c])) {
               fprintf(stderr,
                       "GL45 %s (%u,%u) got %02x%02x%02x%02x, expected "
                       "%02x%02x%02x%02x (+/-2), error %#x\n",
                       phase, x, y, got[0], got[1], got[2], got[3],
                       want[0], want[1], want[2], want[3], error);
               return false;
            }
         }
      }
   }
   fprintf(stderr, "GL45 %s: exact Qt alpha distance-field pixels (+/-2)\n",
           phase);
   return error == GL_NO_ERROR;
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
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
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

   const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
   if (!extensions ||
       !has_extension(extensions, "GL_EXT_texture_format_BGRA8888"))
      return EXIT_FAILURE;

   GLuint program = make_program();
   GLuint sentinel_program = make_sentinel_program();
   if (!program || !sentinel_program)
      return EXIT_FAILURE;

   GLuint atlas_texture;
   glGenTextures(1, &atlas_texture);
   glBindTexture(GL_TEXTURE_2D, atlas_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, GLYPH_WIDTH, 1, 0, GL_ALPHA,
                GL_UNSIGNED_BYTE, atlas);

   GLuint target_texture, framebuffer;
   glGenTextures(1, &target_texture);
   glBindTexture(GL_TEXTURE_2D, target_texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_BGRA8_EXT, WIDTH, HEIGHT);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          target_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return EXIT_FAILURE;

   struct qt_uniforms uniforms = {
      .matrix = {
         1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1,
      },
      .texture_scale = {1.0f / GLYPH_WIDTH, 1.0f},
      .color = {0.50f, 0.25f, 0.75f, 0.75f},
      .alpha_min = 0.44f,
      .alpha_max = 0.56f,
   };
   GLuint uniform_buffer;
   glGenBuffers(1, &uniform_buffer);
   glBindBuffer(GL_UNIFORM_BUFFER, uniform_buffer);
   glBufferData(GL_UNIFORM_BUFFER, sizeof(uniforms), &uniforms,
                GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniform_buffer);

   struct vertex vertices[GLYPHS * 4];
   uint16_t indices[GLYPHS * 6];
   for (unsigned glyph = 0; glyph < GLYPHS; ++glyph) {
      const float left = -1.0f + 2.0f * glyph / GLYPHS;
      const float right = -1.0f + 2.0f * (glyph + 1) / GLYPHS;
      const struct vertex quad[4] = {
         {{left, -1}, {0.0f, 0.5f}},
         {{right, -1}, {GLYPH_WIDTH, 0.5f}},
         {{left, 1}, {0.0f, 0.5f}},
         {{right, 1}, {GLYPH_WIDTH, 0.5f}},
      };
      memcpy(&vertices[glyph * 4], quad, sizeof(quad));
      const uint16_t base = glyph * 4;
      const uint16_t tris[6] = {
         base, (uint16_t)(base + 1), (uint16_t)(base + 2),
         (uint16_t)(base + 2), (uint16_t)(base + 1),
         (uint16_t)(base + 3),
      };
      memcpy(&indices[glyph * 6], tris, sizeof(tris));
   }

   GLuint vao, vertex_buffer, index_buffer;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vertex_buffer);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)(2 * sizeof(float)));
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glGenBuffers(1, &index_buffer);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                GL_STATIC_DRAW);

   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_DITHER);
   glDisable(GL_BLEND);
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(sentinel_program);
   glUniformBlockBinding(sentinel_program,
                         glGetUniformBlockIndex(sentinel_program, "buf"), 0);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, atlas_texture);
   glUniform1i(glGetUniformLocation(sentinel_program, "_qt_texture"), 1);
   glDrawElements(GL_TRIANGLES, GLYPHS * 6, GL_UNSIGNED_SHORT, NULL);
   glFinish();
   if (!read_alpha_sentinel("raw alpha swizzle immediate", noop))
      return EXIT_FAILURE;
   sleep(3);
   if (!read_alpha_sentinel("raw alpha swizzle retained", noop))
      return EXIT_FAILURE;

   glClearColor(24.0f / 255.0f, 28.0f / 255.0f, 38.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glUseProgram(program);
   glUniformBlockBinding(program, glGetUniformBlockIndex(program, "buf"), 0);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, atlas_texture);
   glUniform1i(glGetUniformLocation(program, "_qt_texture"), 1);
   glDrawElements(GL_TRIANGLES, GLYPHS * 6, GL_UNSIGNED_SHORT, NULL);
   glFinish();
   if (!read_expected("indexed distance-field draw immediate", noop))
      return EXIT_FAILURE;
   sleep(3);
   if (!read_expected("indexed distance-field draw retained", noop))
      return EXIT_FAILURE;

   glDeleteBuffers(1, &index_buffer);
   glDeleteBuffers(1, &vertex_buffer);
   glDeleteBuffers(1, &uniform_buffer);
   glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(1, &framebuffer);
   glDeleteTextures(1, &target_texture);
   glDeleteTextures(1, &atlas_texture);
   glDeleteProgram(sentinel_program);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL45 Qt GL_ALPHA/.a distance-field state submitted; pixels skipped"
             : "GL45 Qt GL_ALPHA/.a swizzle and indexed BGRA8 distance-field "
               "pixels exact immediately/3s");
   return EXIT_SUCCESS;
}
