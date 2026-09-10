/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * F50: Qt Quick GL 2.1 native-mask <-> distance-field material transition.
 * The native shader inputs are exact GLSL captured from the failing Omarchy
 * process. The distance-field pair reconstructs Qt's source with the same
 * compatibility-profile conventions and alpha-swizzled glyph texture.
 */

#define GL_GLEXT_PROTOTYPES 1

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 128
#define BAND_H 16
#define HEIGHT (BAND_H * 4)
#define CYCLES 64

struct vertex {
   float x, y, s, t, order;
};

/* Byte layouts written by Qt before its GL 2.1 backend expands the blocks
 * into the classic uniforms visible in the captured compatibility shaders. */
struct native_uniform_data {
   float model_view[16];       /* offset 0 */
   float projection[16];       /* offset 64 */
   float texture_scale[2];     /* offset 128 */
   float dpr;                  /* offset 136 */
   float vec4_padding;         /* offset 140 */
   float color[4];             /* offset 144 */
};

struct sdf_uniform_data {
   float matrix[16];           /* offset 0 */
   float texture_scale[2];     /* offset 64 */
   float vec4_padding[2];      /* offset 72 */
   float color[4];             /* offset 80 */
   float alpha_min;            /* offset 96 */
   float alpha_max;            /* offset 100 */
};

_Static_assert(offsetof(struct native_uniform_data, texture_scale) == 128,
               "Qt native textureScale offset");
_Static_assert(offsetof(struct native_uniform_data, dpr) == 136,
               "Qt native DPR offset");
_Static_assert(offsetof(struct native_uniform_data, color) == 144,
               "Qt native color offset");
_Static_assert(sizeof(struct native_uniform_data) == 160,
               "Qt native uniform byte size");
_Static_assert(offsetof(struct sdf_uniform_data, texture_scale) == 64,
               "Qt SDF textureScale offset");
_Static_assert(offsetof(struct sdf_uniform_data, color) == 80,
               "Qt SDF color offset");
_Static_assert(offsetof(struct sdf_uniform_data, alpha_min) == 96,
               "Qt SDF alphaMin offset");
_Static_assert(sizeof(struct sdf_uniform_data) == 104,
               "Qt SDF uniform byte size");

enum workload {
   SDF_ONLY,
   NATIVE_SDF,
   FORCED_WRONG_MASK,
};

static const uint8_t distance_values[16] = {
   0, 48, 80, 96, 108, 116, 122, 126,
   130, 134, 140, 152, 176, 208, 240, 255,
};

static const uint8_t mask_values[16] = {
   0, 24, 64, 112, 160, 208, 255, 208,
   160, 112, 64, 24, 0, 96, 192, 255,
};

static const float background[4] = {
   19.0f / 255.0f, 25.0f / 255.0f, 37.0f / 255.0f, 1.0f,
};

static char *
read_file(const char *path)
{
   FILE *file = fopen(path, "rb");
   if (!file)
      return NULL;
   if (fseek(file, 0, SEEK_END)) {
      fclose(file);
      return NULL;
   }
   long length = ftell(file);
   if (length < 1 || fseek(file, 0, SEEK_SET)) {
      fclose(file);
      return NULL;
   }
   char *source = malloc((size_t)length + 1);
   if (!source || fread(source, 1, (size_t)length, file) != (size_t)length) {
      free(source);
      fclose(file);
      return NULL;
   }
   source[length] = '\0';
   fclose(file);
   return source;
}

static GLuint
compile_shader(GLenum stage, const char *source, const char *label)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "F50 %s shader compile failed: %s\n", label, log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *vs_source, const char *fs_source, const char *label)
{
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source, label);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source, label);
   if (!vs || !fs) {
      glDeleteShader(vs);
      glDeleteShader(fs);
      return 0;
   }
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "vCoord");
   glBindAttribLocation(program, 1, "tCoord");
   glBindAttribLocation(program, 2, "_qt_order");
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "F50 %s program link failed: %s\n", label, log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
gl_ok(const char *where)
{
   bool ok = true;
   for (GLenum error; (error = glGetError()) != GL_NO_ERROR;) {
      fprintf(stderr, "F50 %s GL error %#x\n", where, error);
      ok = false;
   }
   return ok;
}

static void
ortho(float matrix[16])
{
   memset(matrix, 0, 16 * sizeof(float));
   matrix[0] = 2.0f / WIDTH;
   matrix[5] = 2.0f / HEIGHT;
   matrix[10] = 1.0f;
   matrix[12] = -1.0f;
   matrix[13] = -1.0f;
   matrix[15] = 1.0f;
}

static void
set_vec2(GLuint program, const char *name, float x, float y)
{
   GLint location = glGetUniformLocation(program, name);
   if (location >= 0)
      glUniform2f(location, x, y);
}

static void
set_vec4(GLuint program, const char *name, const float value[4])
{
   GLint location = glGetUniformLocation(program, name);
   if (location >= 0)
      glUniform4fv(location, 1, value);
}

static void
set_matrix(GLuint program, const char *name, const float value[16])
{
   GLint location = glGetUniformLocation(program, name);
   if (location >= 0)
      glUniformMatrix4fv(location, 1, GL_FALSE, value);
}

static void
set_sdf_uniforms(GLuint program, const float color[4])
{
   struct sdf_uniform_data data = {0};
   ortho(data.matrix);
   data.texture_scale[0] = 1.0f / WIDTH;
   data.texture_scale[1] = 1.0f / BAND_H;
   memcpy(data.color, color, sizeof(data.color));
   data.alpha_min = 0.44f;
   data.alpha_max = 0.56f;
   set_matrix(program, "_17.matrix", data.matrix);
   set_matrix(program, "_14.matrix", data.matrix);
   set_vec2(program, "_17.textureScale", data.texture_scale[0],
            data.texture_scale[1]);
   set_vec2(program, "_14.textureScale", data.texture_scale[0],
            data.texture_scale[1]);
   set_vec4(program, "_17.color", data.color);
   set_vec4(program, "_14.color", data.color);
   GLint min_location = glGetUniformLocation(program, "_14.alphaMin");
   GLint max_location = glGetUniformLocation(program, "_14.alphaMax");
   if (min_location >= 0)
      glUniform1f(min_location, data.alpha_min);
   if (max_location >= 0)
      glUniform1f(max_location, data.alpha_max);
}

static void
set_native_uniforms(GLuint program, const float color[4])
{
   struct native_uniform_data data = {0};
   data.model_view[0] = data.model_view[5] = 1.0f;
   data.model_view[10] = data.model_view[15] = 1.0f;
   ortho(data.projection);
   data.texture_scale[0] = 1.0f / WIDTH;
   data.texture_scale[1] = 1.0f / BAND_H;
   data.dpr = 1.0f;
   memcpy(data.color, color, sizeof(data.color));
   set_matrix(program, "_17.modelViewMatrix", data.model_view);
   set_matrix(program, "_14.modelViewMatrix", data.model_view);
   set_matrix(program, "_17.projectionMatrix", data.projection);
   set_matrix(program, "_14.projectionMatrix", data.projection);
   set_vec2(program, "_17.textureScale", data.texture_scale[0],
            data.texture_scale[1]);
   set_vec2(program, "_14.textureScale", data.texture_scale[0],
            data.texture_scale[1]);
   GLint dpr = glGetUniformLocation(program, "_17.dpr");
   if (dpr >= 0)
      glUniform1f(dpr, data.dpr);
   set_vec4(program, "_17.color", data.color);
   set_vec4(program, "_14.color", data.color);
}

static GLuint
make_alpha_texture(const uint8_t values[16], unsigned generation)
{
   uint8_t pixels[WIDTH * BAND_H];
   for (unsigned y = 0; y < BAND_H; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x)
         pixels[y * WIDTH + x] = values[(x + generation) & 15u];
   }
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, WIDTH, BAND_H, 0, GL_ALPHA,
                GL_UNSIGNED_BYTE, pixels);
   return texture;
}

static void
fill_geometry(struct vertex vertices[16])
{
   for (unsigned band = 0; band < 4; ++band) {
      const float y0 = band * BAND_H;
      const float y1 = y0 + BAND_H;
      struct vertex *v = &vertices[band * 4];
      v[0] = (struct vertex){0, y0, 0, 0, 0.1f};
      v[1] = (struct vertex){WIDTH, y0, WIDTH, 0, 0.1f};
      v[2] = (struct vertex){0, y1, 0, BAND_H, 0.1f};
      v[3] = (struct vertex){WIDTH, y1, WIDTH, BAND_H, 0.1f};
   }
}

static void
draw_band(GLuint program, GLuint texture, bool native, const float color[4],
          GLuint vbo, GLuint ebo, unsigned band, bool finish_between)
{
   glUseProgram(program);
   if (native)
      set_native_uniforms(program, color);
   else
      set_sdf_uniforms(program, color);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                   native ? GL_NEAREST : GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                   native ? GL_NEAREST : GL_LINEAR);
   GLint sampler = glGetUniformLocation(program, "_qt_texture");
   if (sampler >= 0)
      glUniform1i(sampler, 1);

   const uintptr_t vertex_offset = band * 4u * sizeof(struct vertex);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)vertex_offset);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)(vertex_offset + 2u * sizeof(float)));
   glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)(vertex_offset + 4u * sizeof(float)));
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
   glScissor(0, band * BAND_H, WIDTH, BAND_H);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT,
                  (const void *)(uintptr_t)(band * 6u * sizeof(uint16_t)));
   if (finish_between)
      glFinish();
}

static float
smoothstep_reference(float x)
{
   float t = fminf(fmaxf((x - 0.44f) / 0.12f, 0.0f), 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static uint8_t
to_unorm(float x)
{
   x = fminf(fmaxf(x, 0.0f), 1.0f);
   return (uint8_t)(x * 255.0f + 0.5f);
}

static bool
validate(const bool native_band[4], const bool wrong_band[4],
         const float colors[4][4], unsigned generation, bool noop)
{
   if (noop)
      return true;
   uint8_t pixels[WIDTH * HEIGHT * 4];
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   GLenum error = glGetError();
   for (unsigned band = 0; band < 4; ++band) {
      for (unsigned y = 0; y < BAND_H; ++y) {
         for (unsigned x = 0; x < WIDTH; ++x) {
            const uint8_t raw = native_band[band]
               ? mask_values[(x + generation) & 15u]
               : distance_values[(x + generation) & 15u];
            float coverage = raw / 255.0f;
            if (!native_band[band] && !wrong_band[band])
               coverage = smoothstep_reference(coverage);
            const float source_alpha = colors[band][3] * coverage;
            uint8_t expected[4];
            for (unsigned c = 0; c < 3; ++c) {
               expected[c] = to_unorm(colors[band][c] * coverage +
                                      background[c] * (1.0f - source_alpha));
            }
            expected[3] = 255;
            const uint8_t *got = &pixels[((band * BAND_H + y) * WIDTH + x) * 4];
            for (unsigned c = 0; c < 4; ++c) {
               int delta = (int)got[c] - (int)expected[c];
               if (delta < -2 || delta > 2) {
                  fprintf(stderr,
                          "F50 band%u (%u,%u) got %u,%u,%u,%u expected "
                          "%u,%u,%u,%u (+/-2), error %#x\n",
                          band, x, y, got[0], got[1], got[2], got[3],
                          expected[0], expected[1], expected[2], expected[3],
                          error);
                  return false;
               }
            }
         }
      }
   }
   return error == GL_NO_ERROR;
}

int
main(int argc, char **argv)
{
   if (argc < 5 || argc > 7) {
      fprintf(stderr, "usage: %s native.vert native.frag sdf.vert sdf.frag "
                      "[--sdf-only|--native-sdf|--forced-wrong-mask] "
                      "[--tight|--finish-between]\n", argv[0]);
      return 2;
   }
   enum workload workload = NATIVE_SDF;
   if (argc > 5 && strcmp(argv[5], "--sdf-only") == 0)
      workload = SDF_ONLY;
   else if (argc > 5 && strcmp(argv[5], "--native-sdf") == 0)
      workload = NATIVE_SDF;
   else if (argc > 5 && strcmp(argv[5], "--forced-wrong-mask") == 0)
      workload = FORCED_WRONG_MASK;
   else if (argc > 5)
      return 2;
   const bool finish_between = argc > 6 && strcmp(argv[6], "--finish-between") == 0;
   if (argc > 6 && !finish_between && strcmp(argv[6], "--tight"))
      return 2;

   char *native_vs = read_file(argv[1]);
   char *native_fs = read_file(argv[2]);
   char *sdf_vs = read_file(argv[3]);
   char *sdf_fs = read_file(argv[4]);
   if (!native_vs || !native_fs || !sdf_vs || !sdf_fs)
      return 2;

   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;
   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return 1;
   const EGLint pbuffer_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 2,
      EGL_CONTEXT_MINOR_VERSION, 1,
      EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attrs);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                         context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   const char *version = (const char *)glGetString(GL_VERSION);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("F50 renderer: %s; version: %s\n", renderer, version);

   GLuint native_program = make_program(native_vs, native_fs, "captured mask");
   GLuint sdf_program = make_program(sdf_vs, sdf_fs, "distance field");
   free(native_vs);
   free(native_fs);
   free(sdf_vs);
   free(sdf_fs);
   if (!native_program || !sdf_program)
      return 1;

   GLuint vbo, ebo;
   glGenBuffers(1, &vbo);
   glGenBuffers(1, &ebo);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glEnableVertexAttribArray(2);
   static const uint16_t quad[6] = {0, 2, 3, 3, 1, 0};
   uint16_t indices[24];
   for (unsigned i = 0; i < 4; ++i)
      memcpy(&indices[i * 6], quad, sizeof(quad));

   static const float colors[4][4] = {
      {0.72f, 0.16f, 0.08f, 0.80f},
      {0.08f, 0.68f, 0.20f, 0.85f},
      {0.10f, 0.24f, 0.82f, 0.90f},
      {0.78f, 0.38f, 0.06f, 0.75f},
   };
   bool native_band[4] = {false, false, false, false};
   bool wrong_band[4] = {false, false, false, false};
   if (workload != SDF_ONLY) {
      native_band[0] = native_band[2] = true;
      if (workload == FORCED_WRONG_MASK)
         wrong_band[1] = wrong_band[3] = true;
   }

   GLuint deferred_native[CYCLES], deferred_sdf[CYCLES];
   glDisable(GL_DITHER);
   glEnable(GL_SCISSOR_TEST);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   for (unsigned cycle = 0; cycle < CYCLES; ++cycle) {
      deferred_native[cycle] = make_alpha_texture(mask_values, cycle);
      deferred_sdf[cycle] = make_alpha_texture(distance_values, cycle);
      struct vertex vertices[16];
      fill_geometry(vertices);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), NULL, GL_STREAM_DRAW);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), NULL, GL_STREAM_DRAW);
      glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(indices), indices);

      glDisable(GL_SCISSOR_TEST);
      glDisable(GL_BLEND);
      glClearColor(background[0], background[1], background[2], background[3]);
      glClear(GL_COLOR_BUFFER_BIT);
      glEnable(GL_SCISSOR_TEST);
      glEnable(GL_BLEND);
      for (unsigned band = 0; band < 4; ++band) {
         bool native = native_band[band] || wrong_band[band];
         GLuint program = native ? native_program : sdf_program;
         GLuint texture = native_band[band] ? deferred_native[cycle]
                                               : deferred_sdf[cycle];
         draw_band(program, texture, native, colors[band], vbo, ebo, band,
                   finish_between);
      }
   }

   glFinish();
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   bool ok = gl_ok("material sequence") &&
             validate(native_band, wrong_band, colors, CYCLES - 1, noop);
   glDeleteTextures(CYCLES, deferred_sdf);
   glDeleteTextures(CYCLES, deferred_native);
   glDeleteBuffers(1, &ebo);
   glDeleteBuffers(1, &vbo);
   glDeleteProgram(sdf_program);
   glDeleteProgram(native_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   const char *name = workload == SDF_ONLY ? "sdf-only"
      : workload == NATIVE_SDF ? "native-sdf" : "forced-wrong-mask";
   printf("F50 %s %s: %s\n", name,
          finish_between ? "finish-between" : "tight",
          noop ? "G17P submission validated; pixels skipped"
               : ok ? "pixels exact (+/-2)" : "FAILED");
   return ok ? 0 : 1;
}
