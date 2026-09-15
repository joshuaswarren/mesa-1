/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL33.texture_swizzle.functional_format_* fails for every texture target
 * the suite can build one on -- 363 cases, independent of the format, so the
 * per-texture swizzle itself is what is wrong rather than any one format.
 *
 * This oracle reads a single known texel back through every swizzle the GL
 * enumerates, once per access method, so a failure names the channel and the
 * source rather than only "found pixel with wrong value".
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A shim control run must not silently answer from a different chip. */
static bool
renderer_is_expected(void)
{
   const char *expect = getenv("ASAHI_GL_EXPECT_RENDERER");
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   if (expect && (!renderer || !strstr(renderer, expect))) {
      fprintf(stderr,
              "texture swizzle: expected a renderer containing '%s', got "
              "'%s'\n", expect, renderer ? renderer : "(null)");
      return false;
   }
   return true;
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
      fprintf(stderr, "texture swizzle: shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

/* The CTS draws a four-vertex triangle strip from a shader that derives the
 * corners from gl_VertexID, and samples either in the vertex shader (passing
 * the single channel through a flat varying) or in the fragment shader. Both
 * stages are reproduced, because vertex-stage texturing is its own path. */
static const char *const quad_body =
   "    switch (gl_VertexID)\n"
   "    {\n"
   "      case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"
   "      case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"
   "      case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"
   "      case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"
   "    }\n";

static GLuint
make_program(const char *fetch, bool in_vertex_shader)
{
   char vs_source[1024], fs_source[1024];

   if (in_vertex_shader) {
      snprintf(vs_source, sizeof(vs_source),
               "#version 330 core\n"
               "uniform sampler2D tex;\n"
               "flat out vec4 result;\n"
               "void main() {\n"
               "    result = %s;\n"
               "%s"
               "}\n", fetch, quad_body);
      snprintf(fs_source, sizeof(fs_source),
               "#version 330 core\n"
               "flat in vec4 result;\n"
               "out vec4 color;\n"
               "void main() {\n"
               "    color = result;\n"
               "}\n");
   } else {
      snprintf(vs_source, sizeof(vs_source),
               "#version 330 core\n"
               "void main() {\n"
               "%s"
               "}\n", quad_body);
      snprintf(fs_source, sizeof(fs_source),
               "#version 330 core\n"
               "uniform sampler2D tex;\n"
               "out vec4 color;\n"
               "void main() {\n"
               "    color = %s;\n"
               "}\n", fetch);
   }

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
      fprintf(stderr, "texture swizzle: link failed: %s\n", log);
      return 0;
   }
   return program;
}

/* The texel the texture holds, one distinguishable value per channel. */
static const GLubyte texel[4] = {0x11, 0x22, 0x33, 0x44};

/* What the texture's own format leaves in each channel before the user
 * swizzle runs: a one-channel format reads (r, 0, 0, 1), and the CTS builds
 * its sources exactly there, which is the composition an RGBA source never
 * exercises. */
struct source_format {
   const char *name;
   GLenum internal_format, format;
   unsigned channels;
};

static GLubyte
expected_channel(GLenum swizzle, const struct source_format *src)
{
   switch (swizzle) {
   case GL_RED:   return texel[0];
   case GL_GREEN: return src->channels >= 2 ? texel[1] : 0x00;
   case GL_BLUE:  return src->channels >= 3 ? texel[2] : 0x00;
   case GL_ALPHA: return src->channels >= 4 ? texel[3] : 0xff;
   case GL_ZERO:  return 0x00;
   case GL_ONE:   return 0xff;
   default:       return 0;
   }
}

static const char *
swizzle_name(GLenum swizzle)
{
   switch (swizzle) {
   case GL_RED:   return "R";
   case GL_GREEN: return "G";
   case GL_BLUE:  return "B";
   case GL_ALPHA: return "A";
   case GL_ZERO:  return "0";
   case GL_ONE:   return "1";
   default:       return "?";
   }
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
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
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
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE,
   };
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   printf("texture swizzle: renderer %s\n",
          (const char *)glGetString(GL_RENDERER));
   if (!renderer_is_expected())
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   static const struct source_format sources[] = {
      {"RGBA8", GL_RGBA8, GL_RGBA, 4},
      {"RG8",   GL_RG8,   GL_RG,   2},
      {"R8",    GL_R8,    GL_RED,  1},
   };

   /* Render one pixel into an RGBA8 target and read it straight back. */
   GLuint fbo, rt;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          rt, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "texture swizzle: incomplete framebuffer\n");
      return 1;
   }
   glViewport(0, 0, 8, 8);

   static const struct { const char *name, *fetch; bool vertex; } access[] = {
      {"fs texture",    "texture(tex, vec2(0.5))",    false},
      {"fs texelFetch", "texelFetch(tex, ivec2(0), 0)", false},
      {"vs texelFetch", "texelFetch(tex, ivec2(0), 0)", true},
   };
   /* Identity, a full reversal (the shape the CTS logs), constants, and one
    * broadcast. */
   static const GLenum swizzles[][4] = {
      {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA},
      {GL_ALPHA, GL_BLUE, GL_GREEN, GL_RED},
      {GL_GREEN, GL_RED, GL_ALPHA, GL_BLUE},
      {GL_ONE, GL_ZERO, GL_RED, GL_GREEN},
      {GL_BLUE, GL_BLUE, GL_BLUE, GL_BLUE},
   };

   /* The no-op shim submits nothing, so only a real GPU can be graded;
    * there the run still covers the links and the submitted stream. */
   const bool grade = getenv("ASAHI_GL_NOOP_SHIM") == NULL;
   bool ok = true;
   for (unsigned f = 0; f < sizeof(sources) / sizeof(sources[0]); ++f) {
   const struct source_format *src = &sources[f];

   GLuint tex;
   glGenTextures(1, &tex);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, (GLint)src->internal_format, 1, 1, 0,
                src->format, GL_UNSIGNED_BYTE, texel);

   for (unsigned a = 0; a < sizeof(access) / sizeof(access[0]); ++a) {
      GLuint program = make_program(access[a].fetch, access[a].vertex);
      if (!program)
         return 1;
      glUseProgram(program);
      glUniform1i(glGetUniformLocation(program, "tex"), 0);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, tex);

      for (unsigned s = 0; s < sizeof(swizzles) / sizeof(swizzles[0]); ++s) {
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R,
                         (GLint)swizzles[s][0]);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G,
                         (GLint)swizzles[s][1]);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B,
                         (GLint)swizzles[s][2]);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A,
                         (GLint)swizzles[s][3]);

         glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         glFinish();

         /* The CTS checks every texel of its 8x8 output, so report the first
          * that differs and how many did. */
         GLubyte image[8 * 8 * 4] = {0};
         glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, image);
         const GLubyte *got = image;
         unsigned differing = 0;
         for (unsigned t = 1; t < 64; ++t) {
            if (memcmp(image + t * 4, image, 4))
               ++differing;
         }

         /* The CTS reads its output with glGetTexImage, not glReadPixels.
          * Read both so a divergence names the readback rather than the
          * swizzle. */
         GLubyte fetched[8 * 8 * 4] = {0};
         glBindTexture(GL_TEXTURE_2D, rt);
         glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, fetched);
         glBindTexture(GL_TEXTURE_2D, tex);
         if (memcmp(image, fetched, sizeof(image))) {
            printf("  %-5s %-11s               glGetTexImage %02x%02x%02x%02x "
                   "!= glReadPixels %02x%02x%02x%02x  READBACK DIVERGES\n",
                   src->name, access[a].name, fetched[0], fetched[1],
                   fetched[2], fetched[3], got[0], got[1], got[2], got[3]);
            ok = grade ? false : ok;
         }

         GLubyte want[4];
         for (unsigned c = 0; c < 4; ++c)
            want[c] = expected_channel(swizzles[s][c], src);

         const bool row_ok = !memcmp(got, want, 4) && differing == 0;
         ok &= row_ok || !grade;
         printf("  %-5s %-13s [%s%s%s%s]  got %02x%02x%02x%02x  want %02x%02x%02x%02x  %s%s\n",
                src->name, access[a].name, swizzle_name(swizzles[s][0]),
                swizzle_name(swizzles[s][1]), swizzle_name(swizzles[s][2]),
                swizzle_name(swizzles[s][3]),
                got[0], got[1], got[2], got[3],
                want[0], want[1], want[2], want[3],
                row_ok ? "" : "WRONG",
                differing ? " (texels differ across the 8x8)" : "");
      }
      glDeleteProgram(program);
   }
   glDeleteTextures(1, &tex);
   }

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "texture swizzle: a GL error was raised\n");
      ok = false;
   }

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &rt);
   glDeleteVertexArrays(1, &vao);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (ok)
      printf("texture swizzle: every swizzle reads back correctly\n");
   return ok ? 0 : 1;
}
