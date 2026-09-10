/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Qt Quick distance-field cache growth/state-transition oracle.
 *
 * The static mode is the small-scene control.  The growth modes reconstruct
 * QSGRhiDistanceFieldGlyphCache's OpenGLES resize workaround: allocate a new
 * R8 texture, initialize it, upload the CPU copy of the old allocation, append
 * new glyph subimages, replace the sampled texture, and update textureScale in
 * the material UBO.  Existing and newly allocated glyph rectangles are then
 * drawn around another cache and an RGBA image material without a finish or
 * readback in the loop.
 */

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 128
#define BAND_H 16
#define HEIGHT (BAND_H * 4)
#define OLD_W WIDTH
#define OLD_H BAND_H
#define GROWN_W (WIDTH * 2)
#define GROWN_H (BAND_H * 2)
#define GLYPH_UPLOAD_W 8
#define CYCLES 64
#define MATERIALS 4

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

struct material {
   GLuint texture;
   float color[4];
   float alpha_min;
   float alpha_max;
   unsigned texture_width;
   unsigned texture_height;
   unsigned region;
};

enum workload {
   WORKLOAD_STATIC_SDF,
   WORKLOAD_GROWTH_SDF,
   WORKLOAD_GROWTH_MASK,
};

static const uint8_t base_coverage[16] = {
   0, 48, 80, 96, 108, 116, 122, 126,
   130, 134, 140, 152, 176, 208, 240, 255,
};

static const float background[4] = {
   17.0f / 255.0f, 23.0f / 255.0f, 31.0f / 255.0f, 1.0f,
};

static bool api_barrier;
static unsigned draw_count;

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
      fprintf(stderr, "F49 %s shader compile failed: %s\n", label, log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *vertex_source, const char *fragment_source,
             const char *label)
{
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_source, label);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_source, label);
   if (!vs || !fs) {
      glDeleteShader(vs);
      glDeleteShader(fs);
      return 0;
   }

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);

   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "F49 %s program link failed: %s\n", label, log);
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
      fprintf(stderr, "F49 %s GL error %#x\n", where, error);
      ok = false;
   }
   return ok;
}

static void
before_draw(void)
{
   if (api_barrier)
      glMemoryBarrier(GL_ALL_BARRIER_BITS);
   draw_count++;
}

static uint8_t
atlas_value(unsigned region, unsigned generation, unsigned x)
{
   switch (region) {
   case 0:
      return base_coverage[x & 15u];
   case 1:
      return (uint8_t)(255u -
                       base_coverage[(x + generation * 3u) & 15u]);
   case 2:
      return base_coverage[(x * 5u + 7u) & 15u];
   default:
      return 0;
   }
}

static void
fill_region(uint8_t *pixels, unsigned width, unsigned height,
            unsigned region, unsigned generation)
{
   for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x)
         pixels[y * width + x] = atlas_value(region, generation, x);
   }
}

static void
set_texture_parameters(void)
{
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_ZERO);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
}

static GLuint
make_static_atlas(unsigned region, unsigned generation)
{
   uint8_t pixels[OLD_W * OLD_H];
   fill_region(pixels, OLD_W, OLD_H, region, generation);

   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   set_texture_parameters();
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, OLD_W, OLD_H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OLD_W, OLD_H, GL_RED,
                   GL_UNSIGNED_BYTE, pixels);
   return texture;
}

static GLuint
make_grown_atlas(unsigned generation)
{
   uint8_t zero[GROWN_W * GROWN_H] = {0};
   uint8_t old_pixels[OLD_W * OLD_H];
   fill_region(old_pixels, OLD_W, OLD_H, 0, 0);

   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   set_texture_parameters();
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

   /* Qt creates a zero-filled replacement, then the OpenGLES workaround
    * restores the CPU-side old allocation and appends glyph subimages. */
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, GROWN_W, GROWN_H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GROWN_W, GROWN_H, GL_RED,
                   GL_UNSIGNED_BYTE, zero);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OLD_W, OLD_H, GL_RED,
                   GL_UNSIGNED_BYTE, old_pixels);

   /* QRhi receives one upload entry per newly rasterized glyph. Keep these
    * separate rather than collapsing the appended half into one transfer. */
   for (unsigned left = 0; left < OLD_W; left += GLYPH_UPLOAD_W) {
      uint8_t glyph[GLYPH_UPLOAD_W * OLD_H];
      for (unsigned y = 0; y < OLD_H; ++y) {
         for (unsigned x = 0; x < GLYPH_UPLOAD_W; ++x) {
            glyph[y * GLYPH_UPLOAD_W + x] =
               atlas_value(1, generation, left + x);
         }
      }
      glTexSubImage2D(GL_TEXTURE_2D, 0, OLD_W + left, 0,
                      GLYPH_UPLOAD_W, OLD_H, GL_RED, GL_UNSIGNED_BYTE, glyph);
   }
   return texture;
}

static void
fill_uniform(struct qt_uniforms *u, const struct material *material)
{
   memset(u, 0, sizeof(*u));
   u->matrix[0] = 2.0f / WIDTH;
   u->matrix[5] = 2.0f / HEIGHT;
   u->matrix[10] = 1.0f;
   u->matrix[12] = -1.0f;
   u->matrix[13] = -1.0f;
   u->matrix[15] = 1.0f;
   u->texture_scale[0] = 1.0f / material->texture_width;
   u->texture_scale[1] = 1.0f / material->texture_height;
   memcpy(u->color, material->color, sizeof(u->color));
   u->alpha_min = material->alpha_min;
   u->alpha_max = material->alpha_max;
}

static void
fill_quad(struct vertex *v, unsigned band, unsigned region)
{
   const float y0 = (float)(band * BAND_H);
   const float y1 = y0 + BAND_H;
   const float tx0 = region == 1 ? OLD_W : 0;
   const float tx1 = tx0 + OLD_W;
   const float ty0 = 0;
   const float ty1 = OLD_H;
   v[0] = (struct vertex){{0, y0}, {tx0, ty0}};
   v[1] = (struct vertex){{WIDTH, y0}, {tx1, ty0}};
   v[2] = (struct vertex){{0, y1}, {tx0, ty1}};
   v[3] = (struct vertex){{WIDTH, y1}, {tx1, ty1}};
}

static void
upload_dynamic_state(GLuint vbo, GLuint ebo, GLuint ubo,
                     GLintptr uniform_stride, const struct material *materials)
{
   struct vertex vertices[MATERIALS * 4];
   static const uint16_t one_quad[6] = {0, 2, 3, 3, 1, 0};
   uint16_t indices[MATERIALS * 6];
   uint8_t uniform_bytes[4 * 1024];

   if ((size_t)(uniform_stride * MATERIALS) > sizeof(uniform_bytes)) {
      fprintf(stderr, "F49 uniform alignment unexpectedly large: %ld\n",
              (long)uniform_stride);
      exit(EXIT_FAILURE);
   }

   memset(uniform_bytes, 0, sizeof(uniform_bytes));
   for (unsigned i = 0; i < MATERIALS; ++i) {
      struct qt_uniforms uniform;
      fill_quad(&vertices[i * 4], i, materials[i].region);
      memcpy(&indices[i * 6], one_quad, sizeof(one_quad));
      fill_uniform(&uniform, &materials[i]);
      memcpy(uniform_bytes + i * uniform_stride, &uniform, sizeof(uniform));
   }

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), NULL, GL_STREAM_DRAW);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), NULL, GL_STREAM_DRAW);
   glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(indices), indices);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo);
   glBufferData(GL_UNIFORM_BUFFER, uniform_stride * MATERIALS, NULL,
                GL_STREAM_DRAW);
   glBufferSubData(GL_UNIFORM_BUFFER, 0, uniform_stride * MATERIALS,
                   uniform_bytes);
}

static void
draw_scene_noise(GLuint program, GLuint texture)
{
   glViewport(0, 0, WIDTH, HEIGHT);
   glEnable(GL_SCISSOR_TEST);
   glScissor(7, 5, WIDTH - 14, HEIGHT - 10);
   glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_REVERSE_SUBTRACT, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR,
                       GL_ONE, GL_ZERO);
   glBlendColor(0.25f, 0.75f, 0.5f, 0.375f);
   glUseProgram(program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);
   glUniform1i(glGetUniformLocation(program, "image"), 0);
   before_draw();
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void
draw_material(GLuint program, GLuint vao, GLuint vbo, GLuint ebo, GLuint ubo,
              GLintptr uniform_stride, const struct material *material,
              unsigned band)
{
   glViewport(0, 0, WIDTH, HEIGHT);
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, band * BAND_H, WIDTH, BAND_H);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glUseProgram(program);
   glBindVertexArray(vao);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
   const uintptr_t vertex_offset = band * 4u * sizeof(struct vertex);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)vertex_offset);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (const void *)(vertex_offset + 2u * sizeof(float)));
   glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, band * uniform_stride,
                     sizeof(struct qt_uniforms));
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, material->texture);
   glUniform1i(glGetUniformLocation(program, "_qt_texture"), 1);
   before_draw();
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT,
                  (const void *)(uintptr_t)(band * 6u * sizeof(uint16_t)));
}

static float
smoothstep_reference(float edge0, float edge1, float x)
{
   float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static uint8_t
to_unorm(float value)
{
   value = fminf(fmaxf(value, 0.0f), 1.0f);
   return (uint8_t)(value * 255.0f + 0.5f);
}

static bool
near_byte(uint8_t actual, uint8_t expected)
{
   return actual + 2 >= expected && actual <= expected + 2;
}

static bool
validate_pixels(const struct material *materials, unsigned generation,
                bool mask, bool noop)
{
   if (noop)
      return true;

   uint8_t pixels[WIDTH * HEIGHT * 4];
   memset(pixels, 0xa5, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   GLenum error = glGetError();
   for (unsigned band = 0; band < MATERIALS; ++band) {
      const struct material *m = &materials[band];
      for (unsigned y = 0; y < BAND_H; ++y) {
         for (unsigned x = 0; x < WIDTH; ++x) {
            const float sample = atlas_value(m->region,
                                             m->region == 1 ? generation : 0,
                                             x) / 255.0f;
            const float coverage = mask
               ? sample
               : smoothstep_reference(m->alpha_min, m->alpha_max, sample);
            const float source_alpha = m->color[3] * coverage;
            uint8_t expected[4];
            for (unsigned c = 0; c < 3; ++c) {
               expected[c] = to_unorm(m->color[c] * coverage +
                                      background[c] * (1.0f - source_alpha));
            }
            expected[3] = 255;
            const uint8_t *got = &pixels[((band * BAND_H + y) * WIDTH + x) * 4];
            for (unsigned c = 0; c < 4; ++c) {
               if (!near_byte(got[c], expected[c])) {
                  fprintf(stderr,
                          "F49 band%u (%u,%u) got %u,%u,%u,%u expected "
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
   enum workload workload = WORKLOAD_GROWTH_SDF;
   if (argc > 1 && strcmp(argv[1], "--static-sdf") == 0)
      workload = WORKLOAD_STATIC_SDF;
   else if (argc > 1 && strcmp(argv[1], "--growth-sdf") == 0)
      workload = WORKLOAD_GROWTH_SDF;
   else if (argc > 1 && strcmp(argv[1], "--growth-mask") == 0)
      workload = WORKLOAD_GROWTH_MASK;
   else if (argc > 1) {
      fprintf(stderr, "usage: %s [--static-sdf|--growth-sdf|--growth-mask] "
                      "[--tight|--api-barrier]\n", argv[0]);
      return EXIT_FAILURE;
   }

   if (argc > 2 && strcmp(argv[2], "--api-barrier") == 0)
      api_barrier = true;
   else if (argc > 2 && strcmp(argv[2], "--tight") != 0) {
      fprintf(stderr, "unknown synchronization mode: %s\n", argv[2]);
      return EXIT_FAILURE;
   }
   if (argc > 3)
      return EXIT_FAILURE;

   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return EXIT_FAILURE;

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
      return EXIT_FAILURE;

   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1,
      EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config,
                                                surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                         context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   static const char qt_vertex[] =
      "#version 300 es\n"
      "layout(location=0) in vec4 vCoord;"
      "layout(location=1) in vec2 tCoord;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "out vec2 sampleCoord;"
      "void main(){sampleCoord=tCoord*textureScale;"
      "gl_Position=matrix*vCoord;}";
   static const char sdf_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=color*smoothstep(alphaMin,alphaMax,"
      "texture(_qt_texture,sampleCoord).r);}";
   static const char mask_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 sampleCoord;"
      "uniform sampler2D _qt_texture;"
      "layout(std140) uniform buf {mat4 matrix;vec2 textureScale;"
      "vec4 color;float alphaMin;float alphaMax;};"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=color*texture(_qt_texture,sampleCoord).r;}";
   static const char scene_vertex[] =
      "#version 300 es\n"
      "const vec2 p[3]=vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),"
      "vec2(-1.0,3.0));out vec2 uv;"
      "void main(){gl_Position=vec4(p[gl_VertexID],0.0,1.0);"
      "uv=p[gl_VertexID]*0.5+0.5;}";
   static const char scene_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D image;"
      "layout(location=0) out vec4 fragColor;"
      "void main(){fragColor=texture(image,uv);}";

   const bool mask = workload == WORKLOAD_GROWTH_MASK;
   GLuint glyph_program = make_program(qt_vertex,
                                        mask ? mask_fragment : sdf_fragment,
                                        mask ? "mask" : "distance-field");
   GLuint scene_program = make_program(scene_vertex, scene_fragment, "scene");
   if (!glyph_program || !scene_program)
      return EXIT_FAILURE;
   GLuint block = glGetUniformBlockIndex(glyph_program, "buf");
   if (block == GL_INVALID_INDEX)
      return EXIT_FAILURE;
   glUniformBlockBinding(glyph_program, block, 0);

   GLuint vao, vbo, ebo, ubo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glGenBuffers(1, &ebo);
   glGenBuffers(1, &ubo);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   GLint uniform_alignment;
   glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniform_alignment);
   GLintptr uniform_stride = (sizeof(struct qt_uniforms) +
                              uniform_alignment - 1) /
                             uniform_alignment * uniform_alignment;

   GLuint cache_b = make_static_atlas(2, 0);
   GLuint static_a = 0;
   if (workload == WORKLOAD_STATIC_SDF)
      static_a = make_grown_atlas(CYCLES - 1);

   static const uint8_t image_pixels[16] = {
      13, 71, 193, 255, 229, 43, 89, 191,
      31, 211, 107, 127, 251, 157, 17, 63,
   };
   GLuint image_texture;
   glGenTextures(1, &image_texture);
   glBindTexture(GL_TEXTURE_2D, image_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 2, 2);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, GL_RGBA,
                   GL_UNSIGNED_BYTE, image_pixels);

   GLuint deferred[CYCLES] = {0};
   GLuint current_a = static_a;
   struct material materials[MATERIALS] = {
      {0, {0.72f, 0.16f, 0.08f, 0.80f}, 0.44f, 0.56f,
       GROWN_W, GROWN_H, 0},
      {0, {0.08f, 0.68f, 0.20f, 0.85f}, 0.42f, 0.58f,
       GROWN_W, GROWN_H, 1},
      {cache_b, {0.10f, 0.24f, 0.82f, 0.90f}, 0.40f, 0.60f,
       OLD_W, OLD_H, 2},
      {0, {0.78f, 0.38f, 0.06f, 0.75f}, 0.46f, 0.54f,
       GROWN_W, GROWN_H, 0},
   };

   glDisable(GL_DITHER);
   for (unsigned cycle = 0; cycle < CYCLES; ++cycle) {
      if (workload != WORKLOAD_STATIC_SDF) {
         current_a = make_grown_atlas(cycle);
         deferred[cycle] = current_a;
      }
      materials[0].texture = current_a;
      materials[1].texture = current_a;
      materials[3].texture = current_a;
      upload_dynamic_state(vbo, ebo, ubo, uniform_stride, materials);

      draw_scene_noise(scene_program, image_texture);
      glDisable(GL_SCISSOR_TEST);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glDisable(GL_BLEND);
      glClearColor(background[0], background[1], background[2], background[3]);
      glClear(GL_COLOR_BUFFER_BIT);

      /* Existing cache A, its newly appended region, another cache B, then A
       * again.  The last transition catches stale B texture/scale state. */
      for (unsigned band = 0; band < MATERIALS; ++band)
         draw_material(glyph_program, vao, vbo, ebo, ubo, uniform_stride,
                       &materials[band], band);
   }

   glFinish();
   if (!gl_ok("after churn") ||
       !validate_pixels(materials, CYCLES - 1, mask, noop))
      return EXIT_FAILURE;

   glDeleteTextures(1, &cache_b);
   glDeleteTextures(1, &image_texture);
   if (static_a)
      glDeleteTextures(1, &static_a);
   else
      glDeleteTextures(CYCLES, deferred);
   glDeleteBuffers(1, &ubo);
   glDeleteBuffers(1, &ebo);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(scene_program);
   glDeleteProgram(glyph_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   const char *workload_name = workload == WORKLOAD_STATIC_SDF ? "static-sdf"
      : workload == WORKLOAD_GROWTH_SDF ? "growth-sdf" : "growth-mask";
   printf("F49 %s %s: %u mixed draws %s\n", workload_name,
          api_barrier ? "api-barrier" : "tight", draw_count,
          noop ? "submitted; pixels skipped" : "pixels exact (+/-2)");
   return EXIT_SUCCESS;
}
