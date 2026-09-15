/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.ext_texture_shadow_lod.texture.sampler2darrayshadow_vertex fails
 * while its textureLod, textureOffset and fragment siblings all pass, so the
 * broken shape is narrow: a depth comparison against an array layer, in the
 * vertex stage, at the explicit level zero a GLSL texture() lowers to there.
 *
 * Give each layer its own depth and sweep the comparison reference across all
 * of them. The layer whose depth the comparison actually used is then readable
 * from the result, so a failure says whether the layer was ignored, the
 * reference was lost, or the comparison itself is wrong.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYERS 4
#define SIZE   4
#define STEPS  64
#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CLAMP_INDEX(i) ((i) < 0 ? 0 : ((i) > SIZE - 1 ? SIZE - 1 : (i)))

/* Layer L holds depth (L + 1) / 8, so a reference between two of them
 * identifies the layer the comparison read. */
static float
layer_depth(unsigned layer)
{
   return (float)(layer + 1) / 8.0f;
}

/* A depth ramp across the texels of a layer, so a filtered comparison has a
 * transition whose position is readable. */
static float
ramp_depth(unsigned x)
{
   return ((float)x + 0.5f) / (float)SIZE;
}

static const char *vs_template =
   "#version 330 core\n"
   "#extension GL_EXT_texture_shadow_lod : enable\n"
   "uniform sampler2DArrayShadow tex;\n"
   "uniform float layer;\n"
   "uniform float reference;\n"
   "uniform float u;\n"
   "uniform float s;\n"
   "uniform float t;\n"
   "out float result;\n"
   "void main() {\n"
   "  result = %s;\n"
   "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
   "  gl_PointSize = 1.0;\n"
   "}\n";

static const char *fs_passthrough =
   "#version 330 core\n"
   "in float result;\n"
   "out vec4 colour;\n"
   "void main() { colour = vec4(result, 0.0, 0.0, 1.0); }\n";

/* The same comparison in the fragment stage, which passes conformance and so
 * says whether the array comparison itself is sound. */
static const char *vs_passthrough =
   "#version 330 core\n"
   "void main() {\n"
   "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
   "  gl_PointSize = 1.0;\n"
   "}\n";

static const char *fs_template =
   "#version 330 core\n"
   "#extension GL_EXT_texture_shadow_lod : enable\n"
   "uniform sampler2DArrayShadow tex;\n"
   "uniform float layer;\n"
   "uniform float reference;\n"
   "uniform float u;\n"
   "uniform float s;\n"
   "uniform float t;\n"
   "out vec4 colour;\n"
   "void main() { colour = vec4(%s, 0.0, 0.0, 1.0); }\n";

static GLuint
compile(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint ok = 0;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024] = {0};
      glGetShaderInfoLog(shader, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "array shadow: compile failed: %s\n", log);
      return 0;
   }
   return shader;
}

static GLuint
build(const char *lookup, bool in_vertex)
{
   char source[1024];
   snprintf(source, sizeof(source),
            in_vertex ? vs_template : fs_template, lookup);

   GLuint vs = compile(GL_VERTEX_SHADER, in_vertex ? source : vs_passthrough);
   GLuint fs =
      compile(GL_FRAGMENT_SHADER, in_vertex ? fs_passthrough : source);
   if (!vs || !fs)
      return 0;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);

   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = {0};
      glGetProgramInfoLog(program, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "array shadow: link failed: %s\n", log);
      return 0;
   }
   return program;
}

static bool
run(const char *what, const char *lookup, bool in_vertex)
{
   GLuint program = build(lookup, in_vertex);
   if (!program)
      return false;

   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   const GLint layer_location = glGetUniformLocation(program, "layer");
   const GLint reference_location = glGetUniformLocation(program, "reference");

   bool ok = true;
   for (unsigned layer = 0; layer < LAYERS; ++layer) {
      /* Straddle this layer's depth, so the pair of results identifies which
       * layer's depth the comparison read. */
      /* The exact depth is the interesting one: GL_LEQUAL must answer one,
       * so a reference that loses precision on its way to the sampler
       * answers zero instead. */
      const float references[] = {layer_depth(layer) - 1.0f / 64.0f,
                                  layer_depth(layer),
                                  layer_depth(layer) + 1.0f / 64.0f,
                                  layer_depth(layer) - 1.0f / 4096.0f,
                                  layer_depth(layer) + 1.0f / 4096.0f};

      for (unsigned r = 0; r < ARRAY_COUNT(references); ++r) {
         glUniform1f(layer_location, (float)layer);
         glUniform1f(reference_location, references[r]);

         glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_POINTS, 0, 1);

         float pixel[4] = {0};
         glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
         if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "array shadow: %s raised a GL error\n", what);
            return false;
         }

         /* GL_LEQUAL against this layer's own depth. */
         const float want = references[r] <= layer_depth(layer) ? 1.0f : 0.0f;
         const bool match = pixel[0] == want;
         ok &= match;

         if (!match) {
            /* Name the layer whose depth would explain the result, so an
             * ignored layer reads as such instead of as a wrong number. */
            int explained = -1;
            for (unsigned l = 0; l < LAYERS; ++l) {
               const float from_l =
                  references[r] <= layer_depth(l) ? 1.0f : 0.0f;
               if (from_l == pixel[0] && from_l != want)
                  explained = (int)l;
            }
            printf("  %s layer %u reference %.5f -> %.3f, wanted %.3f%s\n",
                   what, layer, references[r], pixel[0], want,
                   explained >= 0 ? " (matches another layer's depth)" : "");
         }
      }
   }

   printf("%-40s %s\n", what, ok ? "match" : "MISMATCH");
   glDeleteProgram(program);
   return ok;
}

/* The comparison the hardware should make: a bilinear blend of the per-texel
 * comparisons, which for a ramp constant in y is a blend of two. */
static float
expected_filtered(float u, float reference)
{
   const float coord = u * (float)SIZE - 0.5f;
   const float floor_coord = floorf(coord);
   const float fraction = coord - floor_coord;
   int left = (int)floor_coord;
   int right = left + 1;

   left = CLAMP_INDEX(left);
   right = CLAMP_INDEX(right);

   const float a = reference <= ramp_depth((unsigned)left) ? 1.0f : 0.0f;
   const float b = reference <= ramp_depth((unsigned)right) ? 1.0f : 0.0f;
   return a + (b - a) * fraction;
}

/* Walk the sample across the ramp and report where the transition sits. A
 * displaced transition shows up as the first position whose blend disagrees. */
static bool
sweep(const char *what, const char *lookup, bool in_vertex)
{
   GLuint program = build(lookup, in_vertex);
   if (!program)
      return false;

   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   glUniform1f(glGetUniformLocation(program, "layer"), 1.0f);
   glUniform1f(glGetUniformLocation(program, "reference"), ramp_depth(2));
   const GLint u_location = glGetUniformLocation(program, "u");

   bool ok = true;
   float worst = 0.0f;
   float worst_u = 0.0f;
   for (unsigned step = 0; step <= STEPS; ++step) {
      const float u = (float)step / (float)STEPS;
      glUniform1f(u_location, u);

      glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_POINTS, 0, 1);

      float pixel[4] = {0};
      glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
      if (glGetError() != GL_NO_ERROR) {
         fprintf(stderr, "array shadow: %s raised a GL error\n", what);
         return false;
      }

      const float want = expected_filtered(u, ramp_depth(2));
      const float error = fabsf(pixel[0] - want);
      if (error > worst) {
         worst = error;
         worst_u = u;
      }
   }

   /* One step of the sampler's own coordinate precision, no more. */
   ok = worst <= 1.0f / 64.0f;
   printf("%-40s worst %.4f at u %.4f %s\n", what, worst, worst_u,
          ok ? "match" : "MISMATCH");
   glDeleteProgram(program);
   return ok;
}

/* The conformance case's own vertex shader, whose coordinate and comparison
 * reference arrive together in one attribute rather than from uniforms. */
static const char *cts_vs =
   "#version 130\n"
   "#extension GL_EXT_texture_shadow_lod : require\n"
   "in highp vec4 a_position;\n"
   "in highp vec4 a_in0;\n"
   "out mediump vec4 v_color;\n"
   "uniform highp sampler2DArrayShadow u_sampler;\n"
   "void main() {\n"
   "  gl_Position = a_position;\n"
   "  v_color = vec4(texture(u_sampler, a_in0), 0.0, 0.0, 1.0);\n"
   "  gl_PointSize = 1.0;\n"
   "}\n";

static const char *cts_fs =
   "#version 130\n"
   "out mediump vec4 o_color;\n"
   "in mediump vec4 v_color;\n"
   "void main() { o_color = v_color; }\n";

static bool
run_attribute_shape(void)
{
   GLuint vs = compile(GL_VERTEX_SHADER, cts_vs);
   GLuint fs = compile(GL_FRAGMENT_SHADER, cts_fs);
   if (!vs || !fs)
      return false;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "a_position");
   glBindAttribLocation(program, 1, "a_in0");
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = {0};
      glGetProgramInfoLog(program, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "array shadow: attribute shape link failed: %s\n", log);
      return false;
   }
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "u_sampler"), 0);

   GLuint buffer;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_ARRAY_BUFFER, buffer);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), NULL);
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                         (const void *)(4 * sizeof(float)));

   bool ok = true;
   for (unsigned layer = 0; layer < LAYERS; ++layer) {
      const float references[] = {layer_depth(layer) - 1.0f / 64.0f,
                                  layer_depth(layer),
                                  layer_depth(layer) + 1.0f / 64.0f,
                                  layer_depth(layer) - 1.0f / 4096.0f,
                                  layer_depth(layer) + 1.0f / 4096.0f};

      for (unsigned r = 0; r < ARRAY_COUNT(references); ++r) {
         const float vertex[8] = {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.5f, 0.5f, (float)layer, references[r],
         };
         glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STREAM_DRAW);

         glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_POINTS, 0, 1);

         float pixel[4] = {0};
         glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
         if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "array shadow: attribute shape raised an error\n");
            return false;
         }

         const float want = references[r] <= layer_depth(layer) ? 1.0f : 0.0f;
         if (pixel[0] != want) {
            printf("  attribute shape layer %u reference %.5f -> %.3f, "
                   "wanted %.3f\n",
                   layer, references[r], pixel[0], want);
            ok = false;
         }
      }
   }

   glDisableVertexAttribArray(0);
   glDisableVertexAttribArray(1);
   glDeleteBuffers(1, &buffer);
   glDeleteProgram(program);
   printf("%-40s %s\n", "vertex texture() from an attribute",
          ok ? "match" : "MISMATCH");
   return ok;
}

/* Level L of every layer holds depth (L + 1) / 16, so the comparison result
 * says which mip level the lookup read. A GLSL texture() in the vertex stage
 * has no derivatives and must read level zero. */
static bool
run_mip_levels(void)
{
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

   const unsigned levels = 4;
   for (unsigned level = 0; level < levels; ++level) {
      const unsigned size = (1u << (levels - 1)) >> level;
      float *depths = malloc(size * size * LAYERS * sizeof(float));
      if (!depths)
         return false;
      for (unsigned i = 0; i < size * size * LAYERS; ++i)
         depths[i] = (float)(level + 1) / 16.0f;
      glTexImage3D(GL_TEXTURE_2D_ARRAY, level, GL_DEPTH_COMPONENT32F, size,
                   size, LAYERS, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depths);
      free(depths);
   }
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, levels - 1);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
                   GL_LINEAR_MIPMAP_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE,
                   GL_COMPARE_REF_TO_TEXTURE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

   GLuint program = build("texture(tex, vec4(0.5, 0.5, layer, reference))", true);
   if (!program)
      return false;
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   const GLint layer_location = glGetUniformLocation(program, "layer");
   const GLint reference_location = glGetUniformLocation(program, "reference");

   bool ok = true;
   for (unsigned level = 0; level < levels; ++level) {
      /* Straddle this level's depth. Only level zero should ever answer. */
      const float depth = (float)(level + 1) / 16.0f;
      const float references[] = {depth - 1.0f / 64.0f, depth + 1.0f / 64.0f};

      for (unsigned r = 0; r < 2; ++r) {
         glUniform1f(layer_location, 1.0f);
         glUniform1f(reference_location, references[r]);

         glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_POINTS, 0, 1);

         float pixel[4] = {0};
         glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
         if (glGetError() != GL_NO_ERROR)
            return false;

         const float want = references[r] <= 1.0f / 16.0f ? 1.0f : 0.0f;
         if (pixel[0] != want) {
            printf("  mip levels reference %.5f (straddles level %u) -> %.3f, "
                   "wanted %.3f from level 0\n",
                   references[r], level, pixel[0], want);
            ok = false;
         }
      }
   }

   glDeleteProgram(program);
   glDeleteTextures(1, &texture);
   printf("%-40s %s\n", "vertex texture() reads level zero",
          ok ? "match" : "MISMATCH");
   return ok;
}

/* The array layer arrives as a float and the spec rounds it to nearest before
 * clamping, so a lookup at 1.6 reads layer 2. A driver that truncates reads
 * layer 1, and the error appears wherever the coordinate crosses a half. */
static bool
run_fractional_layer(void)
{
   GLuint program = build("texture(tex, vec4(0.5, 0.5, layer, reference))", true);
   if (!program)
      return false;

   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   const GLint layer_location = glGetUniformLocation(program, "layer");
   const GLint reference_location = glGetUniformLocation(program, "reference");

   static const float coordinates[] = {-0.6f, -0.4f, 0.4f, 0.6f, 1.4f, 1.6f,
                                       2.4f,  2.6f,  3.4f, 3.6f, 4.4f};
   bool ok = true;
   for (unsigned i = 0; i < ARRAY_COUNT(coordinates); ++i) {
      const float coordinate = coordinates[i];
      int expected = (int)(coordinate < 0.0f ? coordinate - 0.5f
                                             : coordinate + 0.5f);
      expected = CLAMP_INDEX(expected);

      /* A reference just above this layer's depth answers zero, and just
       * below answers one, so the pair identifies the layer that was read. */
      for (unsigned r = 0; r < 2; ++r) {
         const float reference =
            layer_depth((unsigned)expected) + (r ? 1.0f : -1.0f) / 64.0f;
         glUniform1f(layer_location, coordinate);
         glUniform1f(reference_location, reference);

         glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_POINTS, 0, 1);

         float pixel[4] = {0};
         glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
         if (glGetError() != GL_NO_ERROR)
            return false;

         const float want = r ? 0.0f : 1.0f;
         if (pixel[0] != want) {
            int read = -1;
            for (unsigned l = 0; l < LAYERS; ++l) {
               const float from_l = reference <= layer_depth(l) ? 1.0f : 0.0f;
               if (from_l == pixel[0])
                  read = (int)l;
            }
            printf("  layer coordinate %+.2f -> read layer %d, wanted %d\n",
                   coordinate, read, expected);
            ok = false;
         }
      }
   }

   glDeleteProgram(program);
   printf("%-40s %s\n", "fractional layer rounds to nearest",
          ok ? "match" : "MISMATCH");
   return ok;
}

/* Sweep the whole input box and print every result, so the same binary run
 * against two drivers can be diffed. The conformance case's inputs are not
 * observable from outside it; this is the net cast wide enough to catch them. */
static bool
run_dump(void)
{
   GLuint program = build("texture(tex, vec4(s, t, layer, reference))", true);
   if (!program)
      return false;

   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   const GLint s_location = glGetUniformLocation(program, "s");
   const GLint t_location = glGetUniformLocation(program, "t");
   const GLint layer_location = glGetUniformLocation(program, "layer");
   const GLint reference_location = glGetUniformLocation(program, "reference");

   for (int li = 0; li < (int)LAYERS; ++li) {
      for (int si = 0; si <= 96; ++si) {
         /* A third of a texel at a time, across three texels. */
         const float s = (float)si / (32.0f * 128.0f);
         for (int ti = 0; ti <= 4; ++ti) {
            const float tc = (float)ti / (4.0f * 128.0f);
            for (int ri = 0; ri <= 24; ++ri) {
               const float reference = (float)ri / 128.0f;
               glUniform1f(s_location, s);
               glUniform1f(t_location, tc);
               glUniform1f(layer_location, (float)li);
               glUniform1f(reference_location, reference);

               glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
               glClear(GL_COLOR_BUFFER_BIT);
               glDrawArrays(GL_POINTS, 0, 1);

               float pixel[4] = {0};
               glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
               printf("s %+.6f t %+.6f layer %d ref %+.6f -> %.6f\n", s, tc,
                      li, reference, pixel[0]);
            }
         }
      }
   }

   glDeleteProgram(program);
   return glGetError() == GL_NO_ERROR;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint surface_attributes[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16,
                                        EGL_NONE};
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   if (getenv("ASAHI_GL_NOOP_SHIM") != NULL)
      return 0;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glEnable(GL_PROGRAM_POINT_SIZE);

   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
   glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, SIZE, SIZE,
                LAYERS, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
   for (unsigned layer = 0; layer < LAYERS; ++layer) {
      float depths[SIZE * SIZE];
      for (unsigned i = 0; i < SIZE * SIZE; ++i)
         depths[i] = layer_depth(layer);
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, SIZE, SIZE, 1,
                      GL_DEPTH_COMPONENT, GL_FLOAT, depths);
   }
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE,
                   GL_COMPARE_REF_TO_TEXTURE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

   GLuint target;
   glGenTextures(1, &target);
   glBindTexture(GL_TEXTURE_2D, target);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          target, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "array shadow: no float render target\n");
      return 1;
   }
   glViewport(0, 0, 1, 1);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

   /* A second texture ramps the depth across the texels of each layer and
    * filters the comparison, so the result is a percentage-closer blend whose
    * transitions move with the sample position. The conformance failure is a
    * displaced transition, not a wrong layer, so this is the sensitive
    * measurement; the nearest-filtered sweep above is the coarse one. */
   GLuint ramp;
   glGenTextures(1, &ramp);
   glBindTexture(GL_TEXTURE_2D_ARRAY, ramp);
   glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, SIZE, SIZE,
                LAYERS, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
   for (unsigned layer = 0; layer < LAYERS; ++layer) {
      float depths[SIZE * SIZE];
      for (unsigned y = 0; y < SIZE; ++y)
         for (unsigned x = 0; x < SIZE; ++x)
            depths[y * SIZE + x] = ramp_depth(x);
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, SIZE, SIZE, 1,
                      GL_DEPTH_COMPONENT, GL_FLOAT, depths);
   }
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE,
                   GL_COMPARE_REF_TO_TEXTURE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

   /* Creating the ramp left it bound; the coarse sweep wants the first
    * texture back. */
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

   if (getenv("ASAHI_ARRAY_SHADOW_DUMP") != NULL) {
      /* The conformance case's own texture, as the driver's resource log
       * reports it: z16_unorm, 128x128, four layers, one level. Give it a
       * depth that varies per texel so the sample position is observable. */
      const unsigned big = 128;
      GLuint storage;
      glGenTextures(1, &storage);
      glBindTexture(GL_TEXTURE_2D_ARRAY, storage);
      glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT16, big, big,
                   LAYERS, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
      float *depths = malloc(big * big * sizeof(float));
      if (!depths)
         return 1;
      for (unsigned layer = 0; layer < LAYERS; ++layer) {
         for (unsigned y = 0; y < big; ++y)
            for (unsigned x = 0; x < big; ++x)
               depths[y * big + x] =
                  (float)((x + 2 * y + 16 * layer) % big) / (float)big;
         glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, big, big, 1,
                         GL_DEPTH_COMPONENT, GL_FLOAT, depths);
      }
      free(depths);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE,
                      GL_COMPARE_REF_TO_TEXTURE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
      return run_dump() ? 0 : 1;
   }

   bool ok = true;
   /* The failing conformance shape, then the siblings that pass, so a run
    * says which of them the driver still gets right. */
   ok &= run("vertex texture()",
             "texture(tex, vec4(0.5, 0.5, layer, reference))", true);
   ok &= run("vertex textureLod(0)",
             "textureLod(tex, vec4(0.5, 0.5, layer, reference), 0.0)", true);
   ok &= run("vertex textureOffset()",
             "textureOffset(tex, vec4(0.5, 0.5, layer, reference), ivec2(0))",
             true);
   ok &= run("fragment texture()",
             "texture(tex, vec4(0.5, 0.5, layer, reference))", false);

   ok &= run_fractional_layer();
   ok &= run_attribute_shape();
   ok &= run_mip_levels();
   glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

   glBindTexture(GL_TEXTURE_2D_ARRAY, ramp);
   ok &= sweep("vertex texture() filtered",
               "texture(tex, vec4(u, 0.5, layer, reference))", true);
   ok &= sweep("vertex textureLod(0) filtered",
               "textureLod(tex, vec4(u, 0.5, layer, reference), 0.0)", true);
   ok &= sweep("fragment texture() filtered",
               "texture(tex, vec4(u, 0.5, layer, reference))", false);

   printf("array shadow: %s\n", ok ? "match" : "MISMATCH");
   return ok ? 0 : 1;
}
