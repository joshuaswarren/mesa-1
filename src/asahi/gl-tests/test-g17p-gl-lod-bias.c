/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.texture_lod_bias.texture_lod_bias_all fails one combination of
 * sampler bias and shader bias out of many, which says the level of detail is
 * slightly wrong rather than wholly wrong. Reading the failure's colours back
 * cannot say by how much, so measure the level directly: give mip level L the
 * float value L, sample with trilinear filtering into a float target, and the
 * result IS the level of detail the hardware used.
 *
 * Run against llvmpipe for the reference and against the driver for the
 * answer; the difference between the two is the quantisation to explain.
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

#define LEVELS 8
#define BASE   (1 << (LEVELS - 1))

static const char *vs_source =
   "#version 330 core\n"
   "out vec2 tc;\n"
   "void main() {\n"
   "  vec2 p = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0,\n"
   "                (gl_VertexID & 2) == 0 ? -1.0 : 1.0);\n"
   "  tc = vec2(p.x * 0.5 + 0.5, 0.0);\n"
   "  gl_Position = vec4(p, 0.0, 1.0);\n"
   "}\n";

/* The conformance test's own fragment shader, which is where the level is
 * computed: one coordinate scaled across the primitive, the other pinned. */
static const char *fs_source =
   "#version 330 core\n"
   "in vec2 tc;\n"
   "out vec4 frag;\n"
   "uniform float biasshader;\n"
   "uniform float scale;\n"
   "uniform sampler2D texture0;\n"
   "void main() {\n"
   "  frag = vec4(texture(texture0, vec2(scale * tc.x, 0.0), biasshader).r,\n"
   "              0.0, 0.0, 1.0);\n"
   "}\n";

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
      fprintf(stderr, "lod bias: shader failed to compile: %s\n", log);
      return 0;
   }
   return shader;
}

/* Sample once and return the level of detail the sample landed on. The
 * coordinate spans the whole target, so the derivative is one texel per pixel
 * and the implicit level is zero: the result is the two biases summed. */
static bool
measure(GLint bias_location, float texture_bias, float shader_bias,
        float *level)
{
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, texture_bias);
   glUniform1f(bias_location, shader_bias);

   glClearColor(-1.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

   float pixel[4] = {0};
   glReadPixels(BASE / 2, 1, 1, 1, GL_RGBA, GL_FLOAT, pixel);

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "lod bias: sampling raised a GL error\n");
      return false;
   }

   *level = pixel[0];
   return true;
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
   const EGLint surface_attributes[] = {EGL_WIDTH, BASE, EGL_HEIGHT, 4,
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

   /* Level L holds the value L, so a trilinear sample between levels reads
    * back the fractional level itself. */
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   for (unsigned l = 0; l < LEVELS; ++l) {
      const unsigned size = BASE >> l;
      float *texels = malloc(size * size * sizeof(float));
      if (!texels)
         return 1;
      for (unsigned i = 0; i < size * size; ++i)
         texels[i] = (float)l;
      glTexImage2D(GL_TEXTURE_2D, l, GL_R32F, size, size, 0, GL_RED, GL_FLOAT,
                   texels);
      free(texels);
   }
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, LEVELS - 1);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                   GL_LINEAR_MIPMAP_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   GLuint target;
   glGenTextures(1, &target);
   glBindTexture(GL_TEXTURE_2D, target);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, BASE, 4, 0, GL_RGBA, GL_FLOAT,
                NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          target, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "lod bias: no float render target\n");
      return 1;
   }
   glViewport(0, 0, BASE, 4);

   GLuint vs = compile(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 1;
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = {0};
      glGetProgramInfoLog(program, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "lod bias: link failed: %s\n", log);
      return 1;
   }
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "texture0"), 0);
   /* One texel per pixel across the primitive, so the implicit level is 0. */
   glUniform1f(glGetUniformLocation(program, "scale"), 1.0f);
   const GLint bias_location = glGetUniformLocation(program, "biasshader");
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);

   /* The combination the conformance test reports, its neighbours on the same
    * 1/127 ladder, and a few exact multiples of 1/64 as a control. */
   static const struct {
      float texture_bias, shader_bias;
   } cases[] = {
      {80.0f / 127.0f, 240.0f / 127.0f}, {80.0f / 127.0f, 239.0f / 127.0f},
      {80.0f / 127.0f, 241.0f / 127.0f}, {79.0f / 127.0f, 240.0f / 127.0f},
      {81.0f / 127.0f, 240.0f / 127.0f}, {40.0f / 64.0f, 120.0f / 64.0f},
      {0.0f, 2.519684f},                 {2.519684f, 0.0f},
      {0.0f, 2.5f},                      {2.5f, 0.0f},
      {0.0f, 0.0f},                      {1.0f, 1.0f},
   };

   bool ok = true;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      float level;
      if (!measure(bias_location, cases[i].texture_bias, cases[i].shader_bias,
                   &level))
         return 1;

      const float want = cases[i].texture_bias + cases[i].shader_bias;
      const float error = level - want;
      /* One step of the sampler descriptor's 1/64 level field, plus a little
       * for the filter's own interpolation precision. */
      const bool near = fabsf(error) <= 1.0f / 64.0f + 1.0f / 512.0f;
      ok &= near;

      printf("texture %8.6f + shader %8.6f = %8.6f -> %8.6f  error %+9.6f "
             "(%+7.3f/64) %s\n",
             cases[i].texture_bias, cases[i].shader_bias, want, level, error,
             error * 64.0f, near ? "" : "OUT OF RANGE");
   }

   printf("lod bias: %s\n", ok ? "match" : "MISMATCH");
   return ok ? 0 : 1;
}
