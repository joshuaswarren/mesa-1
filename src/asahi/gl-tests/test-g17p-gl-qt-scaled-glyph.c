/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Qt Quick distance-field glyph -> BGRA8 client -> compositor scaling oracle.
 * Covers both RED_OR_ALPHA8 texture realizations and the transforms Hyprland
 * applies after the client has already rasterized text.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ATLAS_W 16
#define GLYPHS 4
#define CLIENT_W (ATLAS_W * GLYPHS)
#define CLIENT_H 8
#define MAX_OUT_W (CLIENT_W * 2)
#define MAX_OUT_H (CLIENT_H * 2)

#ifndef GL_BGRA8_EXT
#define GL_BGRA8_EXT 0x93a1
#endif

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

struct mode {
   const char *name;
   unsigned scale;
   float translation;
};

static const uint8_t atlas[ATLAS_W] = {
   0, 64, 96, 112, 120, 124, 128, 132,
   136, 144, 160, 192, 224, 240, 248, 255,
};

static const struct mode modes[] = {
   {"1x", 1, 0.0f},
   {"2x", 2, 0.0f},
   {"2x+0.25px", 2, 0.25f},
};

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "GL47 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *vertex_source, const char *fragment_source)
{
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
   if (!vs || !fs)
      return 0;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "GL47 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static uint8_t
to_unorm(float value)
{
   value = fminf(fmaxf(value, 0.0f), 1.0f);
   return (uint8_t)(value * 255.0f + 0.5f);
}

static float
smoothstep_reference(float edge0, float edge1, float x)
{
   float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static void
client_expected(unsigned x, uint8_t out[4])
{
   static const float background[4] = {
      24.0f / 255.0f, 28.0f / 255.0f, 38.0f / 255.0f, 1.0f,
   };
   static const float color[4] = {0.50f, 0.25f, 0.75f, 0.75f};
   const float coverage = smoothstep_reference(
      0.44f, 0.56f, atlas[x % ATLAS_W] / 255.0f);
   const float source_alpha = color[3] * coverage;
   for (unsigned c = 0; c < 3; ++c) {
      out[c] = to_unorm(color[c] * coverage +
                        background[c] * (1.0f - source_alpha));
   }
   out[3] = 255;
}

static unsigned
clamp_x(int x)
{
   return x < 0 ? 0 : x >= CLIENT_W ? CLIENT_W - 1 : (unsigned)x;
}

static void
scaled_expected(unsigned x, const struct mode *mode, uint8_t out[4])
{
   const float source_center =
      ((float)x + 0.5f - mode->translation) / (float)mode->scale;
   const float texel = source_center - 0.5f;
   const int x0 = (int)floorf(texel);
   const float fraction = texel - floorf(texel);
   uint8_t a[4], b[4];
   client_expected(clamp_x(x0), a);
   client_expected(clamp_x(x0 + 1), b);
   for (unsigned c = 0; c < 4; ++c)
      out[c] = to_unorm(((1.0f - fraction) * a[c] + fraction * b[c]) / 255.0f);
}

static bool
near_byte(uint8_t actual, uint8_t expected)
{
   return actual + 2 >= expected && actual <= expected + 2;
}

static bool
check_scaled(const char *atlas_name, const struct mode *mode, bool noop)
{
   const unsigned width = CLIENT_W * mode->scale;
   const unsigned height = CLIENT_H * mode->scale;
   uint8_t pixels[MAX_OUT_W * MAX_OUT_H * 4];
   memset(pixels, 0xa5, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   GLenum error = glGetError();
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
         uint8_t want[4];
         scaled_expected(x, mode, want);
         const uint8_t *got = &pixels[(y * width + x) * 4];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_byte(got[c], want[c])) {
               fprintf(stderr,
                       "GL47 %s %s (%u,%u) got %02x%02x%02x%02x expected "
                       "%02x%02x%02x%02x (+/-2), error %#x\n",
                       atlas_name, mode->name, x, y, got[0], got[1], got[2],
                       got[3], want[0], want[1], want[2], want[3], error);
               return false;
            }
         }
      }
   }

   fprintf(stderr, "GL47 %s %s: exact scaled glyph pixels (+/-2)\n",
           atlas_name, mode->name);
   return error == GL_NO_ERROR;
}

static GLuint
make_texture_target(GLenum internal_format, unsigned width, unsigned height,
                    GLuint *framebuffer)
{
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glGenFramebuffers(1, framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, *framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 0;
   return texture;
}

static bool
run_atlas_variant(GLenum atlas_internal, GLenum atlas_format,
                  const char *sample_channel, GLuint glyph_vao,
                  GLuint glyph_program, GLuint composite_vao,
                  GLuint composite_program, bool noop)
{
   const char *atlas_name = atlas_format == GL_ALPHA ? "GL_ALPHA/.a" : "R8/.r";
   (void)sample_channel;

   GLuint atlas_texture;
   glGenTextures(1, &atlas_texture);
   glBindTexture(GL_TEXTURE_2D, atlas_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, atlas_internal, ATLAS_W, 1, 0, atlas_format,
                GL_UNSIGNED_BYTE, atlas);

   GLuint client_fbo;
   GLuint client =
      make_texture_target(GL_BGRA8_EXT, CLIENT_W, CLIENT_H, &client_fbo);
   if (!client)
      return false;

   glViewport(0, 0, CLIENT_W, CLIENT_H);
   glDisable(GL_DITHER);
   glClearColor(24.0f / 255.0f, 28.0f / 255.0f, 38.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glUseProgram(glyph_program);
   glBindVertexArray(glyph_vao);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, atlas_texture);
   glUniform1i(glGetUniformLocation(glyph_program, "_qt_texture"), 1);
   glDrawElements(GL_TRIANGLES, GLYPHS * 6, GL_UNSIGNED_SHORT, NULL);

   bool ok = true;
   for (unsigned i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
      const struct mode *mode = &modes[i];
      const unsigned width = CLIENT_W * mode->scale;
      const unsigned height = CLIENT_H * mode->scale;
      GLuint output_fbo;
      GLuint output = make_texture_target(GL_RGBA8, width, height, &output_fbo);
      if (!output) {
         ok = false;
         break;
      }

      glDisable(GL_BLEND);
      glViewport(0, 0, width, height);
      glUseProgram(composite_program);
      glBindVertexArray(composite_vao);
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, client);
      glUniform1i(glGetUniformLocation(composite_program, "clientTexture"), 2);
      glUniform1f(glGetUniformLocation(composite_program, "uOffset"),
                  -mode->translation / (float)width);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFinish();
      ok &= check_scaled(atlas_name, mode, noop);

      glDeleteFramebuffers(1, &output_fbo);
      glDeleteTextures(1, &output);
   }

   glDeleteFramebuffers(1, &client_fbo);
   glDeleteTextures(1, &client);
   glDeleteTextures(1, &atlas_texture);
   return ok;
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
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
      EGL_OPENGL_ES3_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return EXIT_FAILURE;
   const EGLint surface_attrs[] = {
      EGL_WIDTH, MAX_OUT_W, EGL_HEIGHT, MAX_OUT_H, EGL_NONE,
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

   static const char glyph_vs[] =
      "#version 300 es\n"
      "layout(location=0) in vec4 vCoord;layout(location=1) in vec2 tCoord;"
      "layout(std140) uniform buf{mat4 matrix;vec2 textureScale;vec4 color;"
      "float alphaMin;float alphaMax;};out vec2 sampleCoord;"
      "void main(){sampleCoord=tCoord*textureScale;gl_Position=matrix*vCoord;}";
   static const char glyph_r_fs[] =
      "#version 300 es\nprecision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;layout(std140) uniform buf{mat4 matrix;"
      "vec2 textureScale;vec4 color;float alphaMin;float alphaMax;};"
      "layout(location=0) out vec4 fragColor;void main(){fragColor=color*"
      "smoothstep(alphaMin,alphaMax,texture(_qt_texture,sampleCoord).r);}";
   static const char glyph_a_fs[] =
      "#version 300 es\nprecision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;layout(std140) uniform buf{mat4 matrix;"
      "vec2 textureScale;vec4 color;float alphaMin;float alphaMax;};"
      "layout(location=0) out vec4 fragColor;void main(){fragColor=color*"
      "smoothstep(alphaMin,alphaMax,texture(_qt_texture,sampleCoord).a);}";
   static const char composite_vs[] =
      "#version 300 es\nlayout(location=0) in vec2 pos;out vec2 uv;"
      "void main(){gl_Position=vec4(pos,0.0,1.0);uv=pos*0.5+0.5;}";
   static const char composite_fs[] =
      "#version 300 es\nprecision highp float;in vec2 uv;"
      "uniform sampler2D clientTexture;uniform float uOffset;"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=texture(clientTexture,uv+vec2(uOffset,0.0));}";

   GLuint glyph_r = make_program(glyph_vs, glyph_r_fs);
   GLuint glyph_a = make_program(glyph_vs, glyph_a_fs);
   GLuint composite = make_program(composite_vs, composite_fs);
   if (!glyph_r || !glyph_a || !composite)
      return EXIT_FAILURE;

   struct qt_uniforms uniforms = {
      .matrix = {1, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 0, 0, 1},
      .texture_scale = {1.0f / ATLAS_W, 1.0f},
      .color = {0.50f, 0.25f, 0.75f, 0.75f},
      .alpha_min = 0.44f,
      .alpha_max = 0.56f,
   };
   GLuint ubo;
   glGenBuffers(1, &ubo);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo);
   glBufferData(GL_UNIFORM_BUFFER, sizeof(uniforms), &uniforms,
                GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
   glUniformBlockBinding(glyph_r, glGetUniformBlockIndex(glyph_r, "buf"), 0);
   glUniformBlockBinding(glyph_a, glGetUniformBlockIndex(glyph_a, "buf"), 0);

   struct vertex vertices[GLYPHS * 4];
   uint16_t indices[GLYPHS * 6];
   for (unsigned glyph = 0; glyph < GLYPHS; ++glyph) {
      const float left = -1.0f + 2.0f * glyph / GLYPHS;
      const float right = -1.0f + 2.0f * (glyph + 1) / GLYPHS;
      const struct vertex quad[4] = {
         {{left, -1}, {0, 0.5f}}, {{right, -1}, {ATLAS_W, 0.5f}},
         {{left, 1}, {0, 0.5f}},  {{right, 1}, {ATLAS_W, 0.5f}},
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

   GLuint glyph_vao, vbo, ibo;
   glGenVertexArrays(1, &glyph_vao);
   glBindVertexArray(glyph_vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex), 0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)(2 * sizeof(float)));
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glGenBuffers(1, &ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                GL_STATIC_DRAW);

   static const float fullscreen[] = {-1, -1, 1, -1, -1, 1, 1, 1};
   GLuint composite_vao, composite_vbo;
   glGenVertexArrays(1, &composite_vao);
   glBindVertexArray(composite_vao);
   glGenBuffers(1, &composite_vbo);
   glBindBuffer(GL_ARRAY_BUFFER, composite_vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreen), fullscreen, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);

   bool ok = run_atlas_variant(GL_R8, GL_RED, ".r", glyph_vao, glyph_r,
                               composite_vao, composite, noop);
   ok &= run_atlas_variant(GL_ALPHA, GL_ALPHA, ".a", glyph_vao, glyph_a,
                           composite_vao, composite, noop);

   glDeleteBuffers(1, &composite_vbo);
   glDeleteVertexArrays(1, &composite_vao);
   glDeleteBuffers(1, &ibo);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &glyph_vao);
   glDeleteBuffers(1, &ubo);
   glDeleteProgram(composite);
   glDeleteProgram(glyph_a);
   glDeleteProgram(glyph_r);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (!ok)
      return EXIT_FAILURE;
   puts(noop ? "GL47 scaled Qt glyph state submitted; pixels skipped"
             : "GL47 Qt R8/.r and GL_ALPHA/.a glyph client 1x/2x/2x+0.25 "
               "pixels exact");
   return EXIT_SUCCESS;
}
