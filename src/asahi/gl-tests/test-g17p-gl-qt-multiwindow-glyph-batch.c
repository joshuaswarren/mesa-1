/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * F51: Qt Quick GL 2.1 multi-window glyph-cache/batching oracle.
 *
 * The failing QuickShell capture used Qt's threaded render loop and created
 * one QRhi/OpenGL context per window. On the GL backend a grown glyph cache
 * is rebuilt as texture allocation,
 * full zero upload, preserved CPU-image upload, then queued glyph subuploads.
 * This test reproduces that order in independent, unshared contexts before
 * alternating native alpha-mask, distance-field and BGRA image batches.
 */

#define GL_GLEXT_PROTOTYPES 1

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 128
#define BAND_H 16
#define BANDS 4
#define HEIGHT (BAND_H * BANDS)
#define GLYPHS_PER_BAND 16
#define VERTICES (BANDS * GLYPHS_PER_BAND * 4)
#define INDICES (BANDS * GLYPHS_PER_BAND * 6)
#define CYCLES 16
#define MAX_WINDOWS 3
#define SDF_CACHES 2
#define MAX_RETIRED 32

struct vertex {
   float x, y, s, t, order;
};

struct glyph_cache {
   GLuint texture;
   unsigned width;
   unsigned height;
   uint8_t *pixels;
};

struct retired_texture {
   GLuint texture;
   unsigned release_cycle;
};

struct scene {
   EGLContext context;
   EGLSurface surface;
   GLuint native_program;
   GLuint sdf_program;
   GLuint image_program;
   GLuint vbo;
   GLuint ebo;
   GLuint native_texture;
   GLuint image_texture;
   struct glyph_cache sdf[SDF_CACHES];
   struct retired_texture retired[MAX_RETIRED];
   unsigned retired_count;
   unsigned id;
};

enum workload {
   SINGLE_WINDOW_GROWTH,
   MULTI_WINDOW_STATIC,
   MULTI_WINDOW_GROWTH,
   MULTI_WINDOW_DRAW_ONLY,
};

static const uint8_t sdf_values[16] = {
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

static const float colors[BANDS][4] = {
   {0.72f, 0.16f, 0.08f, 0.80f},
   {0.08f, 0.68f, 0.20f, 0.85f},
   {0.10f, 0.24f, 0.82f, 0.90f},
   {0.78f, 0.38f, 0.06f, 0.75f},
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
      fprintf(stderr, "F51 %s shader compile failed: %s\n", label, log);
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
      fprintf(stderr, "F51 %s program link failed: %s\n", label, log);
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
      fprintf(stderr, "F51 %s GL error %#x\n", where, error);
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
set_native_uniforms(GLuint program, const float color[4])
{
   float model[16] = {0};
   float projection[16];
   model[0] = model[5] = model[10] = model[15] = 1.0f;
   ortho(projection);
   set_matrix(program, "_17.modelViewMatrix", model);
   set_matrix(program, "_14.modelViewMatrix", model);
   set_matrix(program, "_17.projectionMatrix", projection);
   set_matrix(program, "_14.projectionMatrix", projection);
   set_vec2(program, "_17.textureScale", 1.0f / WIDTH, 1.0f / BAND_H);
   set_vec2(program, "_14.textureScale", 1.0f / WIDTH, 1.0f / BAND_H);
   GLint dpr = glGetUniformLocation(program, "_17.dpr");
   if (dpr >= 0)
      glUniform1f(dpr, 1.0f);
   set_vec4(program, "_17.color", color);
   set_vec4(program, "_14.color", color);
}

static void
set_sdf_uniforms(GLuint program, const float color[4], unsigned cache,
                 unsigned height)
{
   float matrix[16];
   ortho(matrix);
   set_matrix(program, "_17.matrix", matrix);
   set_matrix(program, "_14.matrix", matrix);
   set_vec2(program, "_17.textureScale", 1.0f / WIDTH, 1.0f / height);
   set_vec2(program, "_14.textureScale", 1.0f / WIDTH, 1.0f / height);
   set_vec4(program, "_17.color", color);
   set_vec4(program, "_14.color", color);
   const float alpha_min = cache ? 0.42f : 0.44f;
   const float alpha_max = cache ? 0.58f : 0.56f;
   GLint location = glGetUniformLocation(program, "_14.alphaMin");
   if (location >= 0)
      glUniform1f(location, alpha_min);
   location = glGetUniformLocation(program, "_14.alphaMax");
   if (location >= 0)
      glUniform1f(location, alpha_max);
}

static void
set_image_uniforms(GLuint program)
{
   float matrix[16];
   ortho(matrix);
   set_matrix(program, "matrix", matrix);
   set_vec2(program, "textureScale", 1.0f / WIDTH, 1.0f / BAND_H);
}

static uint8_t
pattern(const uint8_t values[16], unsigned x, unsigned generation,
        unsigned scene_id, unsigned cache)
{
   return values[(x + generation + scene_id * 3u + cache * 5u) & 15u];
}

static GLuint
make_alpha_texture(unsigned height)
{
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, WIDTH, height, 0, GL_ALPHA,
                GL_UNSIGNED_BYTE, NULL);
   return texture;
}

static void
release_retired(struct scene *scene, unsigned cycle)
{
   unsigned out = 0;
   for (unsigned i = 0; i < scene->retired_count; ++i) {
      if (scene->retired[i].release_cycle <= cycle)
         glDeleteTextures(1, &scene->retired[i].texture);
      else
         scene->retired[out++] = scene->retired[i];
   }
   scene->retired_count = out;
}

static bool
replace_or_update_sdf(struct scene *scene, unsigned cache_index,
                      unsigned generation, bool growth)
{
   struct glyph_cache *cache = &scene->sdf[cache_index];
   unsigned stage = growth ? generation / (cache_index + 1u) : 0;
   if (stage > 7)
      stage = 7;
   const unsigned wanted_height = 32 + stage * 16;
   if (!cache->texture || wanted_height > cache->height) {
      uint8_t *new_pixels = calloc(WIDTH * wanted_height, 1);
      uint8_t *zeros = calloc(WIDTH * wanted_height, 1);
      if (!new_pixels || !zeros) {
         free(new_pixels);
         free(zeros);
         return false;
      }
      if (cache->pixels) {
         for (unsigned y = 0; y < cache->height; ++y)
            memcpy(new_pixels + y * WIDTH, cache->pixels + y * WIDTH, WIDTH);
      }
      GLuint new_texture = make_alpha_texture(wanted_height);
      glBindTexture(GL_TEXTURE_2D, new_texture);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, wanted_height,
                      GL_ALPHA, GL_UNSIGNED_BYTE, zeros);
      if (cache->pixels) {
         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, cache->height,
                         GL_ALPHA, GL_UNSIGNED_BYTE, new_pixels);
         if (scene->retired_count >= MAX_RETIRED) {
            free(new_pixels);
            free(zeros);
            return false;
         }
         scene->retired[scene->retired_count++] =
            (struct retired_texture){cache->texture, generation + 2};
      }
      free(cache->pixels);
      free(zeros);
      cache->texture = new_texture;
      cache->width = WIDTH;
      cache->height = wanted_height;
      cache->pixels = new_pixels;
   }

   glBindTexture(GL_TEXTURE_2D, cache->texture);
   for (unsigned glyph = 0; glyph < GLYPHS_PER_BAND; ++glyph) {
      uint8_t upload[8 * BAND_H];
      const unsigned left = glyph * 8;
      for (unsigned y = 0; y < BAND_H; ++y) {
         for (unsigned x = 0; x < 8; ++x) {
            uint8_t value = pattern(sdf_values, left + x, generation,
                                    scene->id, cache_index);
            upload[y * 8 + x] = value;
            cache->pixels[y * WIDTH + left + x] = value;
         }
      }
      glTexSubImage2D(GL_TEXTURE_2D, 0, left, 0, 8, BAND_H, GL_ALPHA,
                      GL_UNSIGNED_BYTE, upload);
   }
   return true;
}

static void
update_native_texture(struct scene *scene, unsigned generation)
{
   glBindTexture(GL_TEXTURE_2D, scene->native_texture);
   for (unsigned glyph = 0; glyph < GLYPHS_PER_BAND; ++glyph) {
      uint8_t upload[8 * BAND_H];
      const unsigned left = glyph * 8;
      for (unsigned y = 0; y < BAND_H; ++y) {
         for (unsigned x = 0; x < 8; ++x)
            upload[y * 8 + x] = pattern(mask_values, left + x, generation,
                                        scene->id, 0);
      }
      glTexSubImage2D(GL_TEXTURE_2D, 0, left, 0, 8, BAND_H, GL_ALPHA,
                      GL_UNSIGNED_BYTE, upload);
   }
}

static void
update_image_texture(struct scene *scene, unsigned generation)
{
   uint8_t pixels[WIDTH * BAND_H * 4];
   for (unsigned y = 0; y < BAND_H; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned shift = (x + generation + scene->id * 7u) & 31u;
         const uint8_t alpha = 224;
         uint8_t *p = &pixels[(y * WIDTH + x) * 4];
         p[0] = (uint8_t)((32u + shift * 3u) * alpha / 255u); /* B */
         p[1] = (uint8_t)((96u + shift * 2u) * alpha / 255u); /* G */
         p[2] = (uint8_t)((192u - shift * 2u) * alpha / 255u); /* R */
         p[3] = alpha;
      }
   }
   glBindTexture(GL_TEXTURE_2D, scene->image_texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, BAND_H, 0, GL_BGRA,
                GL_UNSIGNED_BYTE, pixels);
}

static void
fill_geometry(struct vertex vertices[VERTICES], uint16_t indices[INDICES])
{
   unsigned vi = 0;
   unsigned ii = 0;
   for (unsigned band = 0; band < BANDS; ++band) {
      const float y0 = band * BAND_H;
      const float y1 = y0 + BAND_H;
      for (unsigned glyph = 0; glyph < GLYPHS_PER_BAND; ++glyph) {
         const float x0 = glyph * 8;
         const float x1 = x0 + 8;
         vertices[vi + 0] = (struct vertex){x0, y0, x0, 0, 0.1f};
         vertices[vi + 1] = (struct vertex){x1, y0, x1, 0, 0.1f};
         vertices[vi + 2] = (struct vertex){x0, y1, x0, BAND_H, 0.1f};
         vertices[vi + 3] = (struct vertex){x1, y1, x1, BAND_H, 0.1f};
         const uint16_t quad[6] = {
            (uint16_t)(vi + 0), (uint16_t)(vi + 2), (uint16_t)(vi + 3),
            (uint16_t)(vi + 3), (uint16_t)(vi + 1), (uint16_t)(vi + 0),
         };
         memcpy(&indices[ii], quad, sizeof(quad));
         vi += 4;
         ii += 6;
      }
   }
}

static void
bind_geometry(const struct scene *scene)
{
   glBindBuffer(GL_ARRAY_BUFFER, scene->vbo);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)(2u * sizeof(float)));
   glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)(4u * sizeof(float)));
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebo);
}

static void
draw_batch(const struct scene *scene, unsigned band, GLuint program,
           GLuint texture, bool native, int sdf_cache)
{
   glUseProgram(program);
   if (native) {
      set_native_uniforms(program, colors[band]);
   } else if (sdf_cache >= 0) {
      set_sdf_uniforms(program, colors[band], (unsigned)sdf_cache,
                       scene->sdf[sdf_cache].height);
   } else {
      set_image_uniforms(program);
   }
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, texture);
   const GLint filter = native ? GL_NEAREST : GL_LINEAR;
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
   GLint sampler = glGetUniformLocation(program,
      sdf_cache < 0 && !native ? "image" : "_qt_texture");
   if (sampler >= 0)
      glUniform1i(sampler, 1);
   bind_geometry(scene);
   glScissor(0, band * BAND_H, WIDTH, BAND_H);
   const uintptr_t offset = band * GLYPHS_PER_BAND * 6u * sizeof(uint16_t);
   glDrawElements(GL_TRIANGLES, GLYPHS_PER_BAND * 6, GL_UNSIGNED_SHORT,
                  (const void *)offset);
}

static bool
render_scene(EGLDisplay display, struct scene *scene, unsigned generation,
             bool growth, bool finish_between, bool update_resources,
             bool submit_frame)
{
   if (!eglMakeCurrent(display, scene->surface, scene->surface, scene->context))
      return false;
   if (update_resources) {
      release_retired(scene, generation);
      update_native_texture(scene, generation);
      if (!replace_or_update_sdf(scene, 0, generation, growth) ||
          !replace_or_update_sdf(scene, 1, generation, growth))
         return false;
      update_image_texture(scene, generation);

      struct vertex vertices[VERTICES];
      uint16_t indices[INDICES];
      fill_geometry(vertices, indices);
      glBindBuffer(GL_ARRAY_BUFFER, scene->vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), NULL, GL_STREAM_DRAW);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), NULL, GL_STREAM_DRAW);
      glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(indices), indices);
   }

   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_BLEND);
   glClearColor(background[0], background[1], background[2], background[3]);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_SCISSOR_TEST);
   glEnable(GL_BLEND);
   draw_batch(scene, 0, scene->native_program, scene->native_texture, true, -1);
   draw_batch(scene, 1, scene->sdf_program, scene->sdf[0].texture, false, 0);
   draw_batch(scene, 2, scene->image_program, scene->image_texture, false, -1);
   draw_batch(scene, 3, scene->sdf_program, scene->sdf[1].texture, false, 1);
   if (finish_between)
      glFinish();
   else if (submit_frame)
      glFlush();
   return gl_ok("frame");
}

static float
smoothstep_reference(float x, unsigned cache)
{
   const float low = cache ? 0.42f : 0.44f;
   const float high = cache ? 0.58f : 0.56f;
   float t = fminf(fmaxf((x - low) / (high - low), 0.0f), 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static uint8_t
to_unorm(float x)
{
   x = fminf(fmaxf(x, 0.0f), 1.0f);
   return (uint8_t)(x * 255.0f + 0.5f);
}

static void
image_source(unsigned x, unsigned generation, unsigned scene_id,
             float out[4])
{
   const unsigned shift = (x + generation + scene_id * 7u) & 31u;
   const float alpha = 224.0f / 255.0f;
   out[0] = ((192u - shift * 2u) * 224u / 255u) / 255.0f;
   out[1] = ((96u + shift * 2u) * 224u / 255u) / 255.0f;
   out[2] = ((32u + shift * 3u) * 224u / 255u) / 255.0f;
   out[3] = alpha;
}

static bool
validate_scene(EGLDisplay display, struct scene *scene, unsigned generation,
               bool noop)
{
   if (!eglMakeCurrent(display, scene->surface, scene->surface, scene->context))
      return false;
   glFinish();
   if (noop)
      return gl_ok("no-op completion");
   uint8_t pixels[WIDTH * HEIGHT * 4];
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   GLenum error = glGetError();
   for (unsigned band = 0; band < BANDS; ++band) {
      for (unsigned y = 0; y < BAND_H; ++y) {
         for (unsigned x = 0; x < WIDTH; ++x) {
            float source[4];
            if (band == 2) {
               image_source(x, generation, scene->id, source);
            } else {
               unsigned cache = band == 3 ? 1 : 0;
               uint8_t raw = pattern(band == 0 ? mask_values : sdf_values,
                                     x, generation, scene->id, cache);
               float coverage = raw / 255.0f;
               if (band != 0)
                  coverage = smoothstep_reference(coverage, cache);
               for (unsigned c = 0; c < 4; ++c)
                  source[c] = colors[band][c] * coverage;
            }
            uint8_t expected[4];
            for (unsigned c = 0; c < 3; ++c) {
               expected[c] = to_unorm(source[c] +
                                      background[c] * (1.0f - source[3]));
            }
            expected[3] = 255;
            const uint8_t *got =
               &pixels[((band * BAND_H + y) * WIDTH + x) * 4];
            for (unsigned c = 0; c < 4; ++c) {
               int delta = (int)got[c] - (int)expected[c];
               if (delta < -2 || delta > 2) {
                  fprintf(stderr,
                          "F51 window%u band%u (%u,%u) got %u,%u,%u,%u "
                          "expected %u,%u,%u,%u (+/-2), error %#x\n",
                          scene->id, band, x, y, got[0], got[1], got[2], got[3],
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

static bool
init_scene(EGLDisplay display, EGLConfig config, struct scene *scene,
           unsigned id, const char *native_vs, const char *native_fs,
           const char *sdf_vs, const char *sdf_fs)
{
   const EGLint surface_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 2,
      EGL_CONTEXT_MINOR_VERSION, 1,
      EGL_NONE,
   };
   scene->id = id;
   scene->surface = eglCreatePbufferSurface(display, config, surface_attrs);
   scene->context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                     context_attrs);
   if (scene->surface == EGL_NO_SURFACE || scene->context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, scene->surface, scene->surface, scene->context))
      return false;

   static const char image_vs[] =
      "#version 120\n"
      "uniform mat4 matrix;uniform vec2 textureScale;"
      "attribute vec4 vCoord;attribute vec2 tCoord;attribute float _qt_order;"
      "varying vec2 uv;void main(){uv=tCoord*textureScale;"
      "gl_Position=matrix*vCoord;gl_Position.z=_qt_order*gl_Position.w;}";
   static const char image_fs[] =
      "#version 120\n"
      "uniform sampler2D image;varying vec2 uv;"
      "void main(){gl_FragData[0]=texture2D(image,uv);}";

   scene->native_program = make_program(native_vs, native_fs, "native mask");
   scene->sdf_program = make_program(sdf_vs, sdf_fs, "distance field");
   scene->image_program = make_program(image_vs, image_fs, "BGRA image");
   if (!scene->native_program || !scene->sdf_program || !scene->image_program)
      return false;
   glGenBuffers(1, &scene->vbo);
   glGenBuffers(1, &scene->ebo);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glEnableVertexAttribArray(2);
   scene->native_texture = make_alpha_texture(BAND_H);
   glGenTextures(1, &scene->image_texture);
   glBindTexture(GL_TEXTURE_2D, scene->image_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glDisable(GL_DITHER);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   return gl_ok("scene initialization");
}

static void
destroy_scene(EGLDisplay display, struct scene *scene)
{
   eglMakeCurrent(display, scene->surface, scene->surface, scene->context);
   for (unsigned i = 0; i < scene->retired_count; ++i)
      glDeleteTextures(1, &scene->retired[i].texture);
   for (unsigned i = 0; i < SDF_CACHES; ++i) {
      glDeleteTextures(1, &scene->sdf[i].texture);
      free(scene->sdf[i].pixels);
   }
   glDeleteTextures(1, &scene->image_texture);
   glDeleteTextures(1, &scene->native_texture);
   glDeleteBuffers(1, &scene->ebo);
   glDeleteBuffers(1, &scene->vbo);
   glDeleteProgram(scene->image_program);
   glDeleteProgram(scene->sdf_program);
   glDeleteProgram(scene->native_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, scene->context);
   eglDestroySurface(display, scene->surface);
}

int
main(int argc, char **argv)
{
   if (argc < 5 || argc > 7) {
      fprintf(stderr, "usage: %s native.vert native.frag sdf.vert sdf.frag "
                      "[--single-window-growth|--multi-window-static|"
                      "--multi-window-growth|--multi-window-draw-only] "
                      "[--tight|--finish-between]\n",
              argv[0]);
      return 2;
   }
   enum workload workload = MULTI_WINDOW_GROWTH;
   if (argc > 5 && strcmp(argv[5], "--single-window-growth") == 0)
      workload = SINGLE_WINDOW_GROWTH;
   else if (argc > 5 && strcmp(argv[5], "--multi-window-static") == 0)
      workload = MULTI_WINDOW_STATIC;
   else if (argc > 5 && strcmp(argv[5], "--multi-window-growth") == 0)
      workload = MULTI_WINDOW_GROWTH;
   else if (argc > 5 && strcmp(argv[5], "--multi-window-draw-only") == 0)
      workload = MULTI_WINDOW_DRAW_ONLY;
   else if (argc > 5)
      return 2;
   const bool finish_between = argc > 6 &&
      strcmp(argv[6], "--finish-between") == 0;
   if (argc > 6 && !finish_between && strcmp(argv[6], "--tight") != 0)
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

   const unsigned window_count = workload == SINGLE_WINDOW_GROWTH ? 1 : 3;
   struct scene scenes[MAX_WINDOWS] = {0};
   bool ok = true;
   for (unsigned i = 0; i < window_count; ++i) {
      if (!init_scene(display, config, &scenes[i], i, native_vs, native_fs,
                      sdf_vs, sdf_fs)) {
         ok = false;
         break;
      }
   }
   free(native_vs);
   free(native_fs);
   free(sdf_vs);
   free(sdf_fs);
   if (!ok)
      return 1;

   const char *version = (const char *)glGetString(GL_VERSION);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("F51 renderer: %s; version: %s; windows: %u\n", renderer, version,
          window_count);
   fflush(stdout);

   const bool draw_only = workload == MULTI_WINDOW_DRAW_ONLY;
   const bool growth = workload == SINGLE_WINDOW_GROWTH ||
                       workload == MULTI_WINDOW_GROWTH;
   if (draw_only) {
      for (unsigned i = 0; i < window_count && ok; ++i)
         ok = render_scene(display, &scenes[i], CYCLES - 1, false, true, true,
                           true);
      if (ok)
         printf("F51 draw-phase-begin: resources initialized and idle\n");
      fflush(stdout);
   }
   for (unsigned cycle = 0; cycle < CYCLES && ok; ++cycle) {
      for (unsigned step = 0; step < window_count; ++step) {
         const unsigned index = (cycle + step) % window_count;
         if (!render_scene(display, &scenes[index], cycle, growth,
                           finish_between, !draw_only, window_count > 1)) {
            ok = false;
            break;
         }
      }
   }

   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   for (unsigned i = 0; i < window_count && ok; ++i)
      ok = validate_scene(display, &scenes[i], CYCLES - 1, noop);
   for (unsigned i = 0; i < window_count; ++i)
      destroy_scene(display, &scenes[i]);
   eglTerminate(display);

   const char *name = workload == SINGLE_WINDOW_GROWTH
      ? "single-window-growth"
      : workload == MULTI_WINDOW_STATIC ? "multi-window-static"
      : workload == MULTI_WINDOW_GROWTH ? "multi-window-growth"
                                        : "multi-window-draw-only";
   printf("F51 %s %s: %s\n", name,
          finish_between ? "finish-between" : "tight",
          noop ? "G17P submission validated; pixels skipped"
               : ok ? "pixels exact (+/-2)" : "FAILED");
   return ok ? 0 : 1;
}
