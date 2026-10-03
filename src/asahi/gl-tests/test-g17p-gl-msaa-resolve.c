/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_*, separated
 * into its two halves. A full-coverage draw into a four-sample target should
 * write every sample; resolving it should then reproduce the drawn colour.
 * The conformance case only reports that the resolved pixel is wrong, which
 * does not say whether the draw or the resolve is at fault. This reads each
 * sample back individually with texelFetch first, then resolves.
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

#define SIZE    16
#define SAMPLES 4

/* The conformance case walks every renderable colour format. A tile layout is
 * admitted per physical format, so check each one the conformance cases use.
 * GL 3.3 reaches further than GL 3.0 did: KHR-GL33.texture_size_promotion
 * renders to the signed normalized families as well, which is what added the
 * snorm rows below. */
/* An integer target needs an integer fragment output and an integer sampler:
 * a float draw writes nothing to one, which the sweep used to report as the
 * draw failing to reach any sample. */
enum channel_kind {
   CHANNEL_FLOAT,
   CHANNEL_UINT,
   CHANNEL_SINT,
};

static const struct {
   const char *name;
   GLenum internal_format;
   enum channel_kind kind;
} formats[] = {
   {"rgba8", GL_RGBA8},     {"rgb565", GL_RGB565},
   {"rgba4", GL_RGBA4},     {"rgb5_a1", GL_RGB5_A1},
   {"rgb10_a2", GL_RGB10_A2}, {"r8", GL_R8},
   {"rg8", GL_RG8},         {"r16f", GL_R16F},
   {"rgba16f", GL_RGBA16F}, {"r32f", GL_R32F},
   {"rgba32f", GL_RGBA32F}, {"srgb8a8", GL_SRGB8_ALPHA8},
   {"r11f", GL_R11F_G11F_B10F}, {"rgb9e5", GL_RGB9_E5},
   {"rg16f", GL_RG16F},     {"rgb10a2ui", GL_RGB10_A2UI, CHANNEL_UINT},
   {"r16", GL_R16},         {"rg16", GL_RG16},
   {"rgba16", GL_RGBA16},   {"r8i", GL_R8I, CHANNEL_SINT},
   {"rgba8i", GL_RGBA8I, CHANNEL_SINT},   {"r32ui", GL_R32UI, CHANNEL_UINT},
   {"r8_snorm", GL_R8_SNORM},     {"rg8_snorm", GL_RG8_SNORM},
   {"rgba8_snorm", GL_RGBA8_SNORM}, {"r16_snorm", GL_R16_SNORM},
   {"rg16_snorm", GL_RG16_SNORM}, {"rgba16_snorm", GL_RGBA16_SNORM},
};

static GLuint
compile(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "msaa resolve: compile failed: %s\n", log);
      return 0;
   }
   return shader;
}

static GLuint
link_program(const char *vs_source, const char *fs_source)
{
   GLuint vs = compile(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 0;
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glDeleteShader(vs);
   glDeleteShader(fs);
   glLinkProgram(program);
   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "msaa resolve: link failed: %s\n", log);
      return 0;
   }
   return program;
}

static const char vs_quad[] =
   "#version 330 core\n"
   "void main()\n"
   "{\n"
   "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
   "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
   "}\n";

/* Exercise GL 3.3's API sample mask together with late depth and discard.
 * Read all samples into one wide image, keeping the diagnostic independent
 * of colour/depth resolve behaviour and requiring only one readback per pass.
 */
static bool
run_sparse_depth_masks(void)
{
   if (getenv("ASAHI_GL_NOOP_SHIM")) {
      fprintf(stderr, "sparse depth masks require real readback, not a no-op shim\n");
      return false;
   }
   printf("sparse depth renderer: %s; GL: %s\n",
          (const char *)glGetString(GL_RENDERER),
          (const char *)glGetString(GL_VERSION));
   static const char write_fs[] =
      "#version 330 core\n"
      "uniform int phase;\n"
      "out vec4 color;\n"
      "void main() {\n"
      "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
      "  if (((p.x + p.y) & 1) != phase) discard;\n"
      "  gl_FragDepth = 0.25 + 0.125 * float(p.x & 1);\n"
      "  color = vec4(1.0, 0.0, 0.0, 1.0);\n"
      "}\n";
   static const char probe_fs[] =
      "#version 330 core\n"
      "out vec4 color;\n"
      "void main() { gl_FragDepth = 0.5; color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
   static const char inspect_fs[] =
      "#version 330 core\n"
      "uniform sampler2DMS colors;\n"
      "uniform sampler2DMS depths;\n"
      "uniform int extent;\n"
      "out vec4 color;\n"
      "void main() {\n"
      "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
      "  int sample_index = p.x / extent;\n"
      "  p.x %= extent;\n"
      "  color = vec4(texelFetch(colors, p, sample_index).rgb,\n"
      "               texelFetch(depths, p, sample_index).r);\n"
      "}\n";
   GLuint write = link_program(vs_quad, write_fs);
   GLuint probe = link_program(vs_quad, probe_fs);
   GLuint inspect = link_program(vs_quad, inspect_fs);
   if (!write || !probe || !inspect)
      return false;
   glUseProgram(inspect);
   glUniform1i(glGetUniformLocation(inspect, "colors"), 0);
   glUniform1i(glGetUniformLocation(inspect, "depths"), 1);
   glUniform1i(glGetUniformLocation(inspect, "extent"), SIZE);
   glDisable(GL_DITHER);

   unsigned checked = 0, sample_counts = 0;
   bool ok = true;
   for (unsigned samples = 2; samples <= 4 && ok; samples *= 2) {
      GLuint textures[3], fbos[2];
      glGenTextures(3, textures);
      glGenFramebuffers(2, fbos);
      glBindFramebuffer(GL_FRAMEBUFFER, fbos[0]);
      for (unsigned i = 0; i < 2; ++i) {
         glActiveTexture(GL_TEXTURE0 + i);
         glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textures[i]);
         glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples,
                                 i ? GL_DEPTH_COMPONENT32F : GL_RGBA8,
                                 SIZE, SIZE, GL_TRUE);
         glFramebufferTexture2D(GL_FRAMEBUFFER,
                                i ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D_MULTISAMPLE, textures[i], 0);
      }
      GLint actual_samples = 0;
      glGetIntegerv(GL_SAMPLES, &actual_samples);
      if (glGetError() != GL_NO_ERROR ||
          glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
         fprintf(stderr, "sparse depth: %ux FBO creation failed\n", samples);
         ok = false;
      } else if (actual_samples != (GLint)samples) {
         /* Some software drivers round a request for 2 up to 4. That is not
          * evidence for the 2x shader variant. Report it explicitly. */
         fprintf(stderr, "sparse depth: requested %ux, got %dx; not tested\n",
                 samples, actual_samples);
         glDeleteFramebuffers(2, fbos);
         glDeleteTextures(3, textures);
         continue;
      }
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, textures[2]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SIZE * samples, SIZE, 0,
                   GL_RGBA, GL_FLOAT, NULL);
      glBindFramebuffer(GL_FRAMEBUFFER, fbos[1]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, textures[2], 0);
      ok &= glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
      float pixels[SAMPLES * SIZE * SIZE * 4];
      unsigned masks_checked = 0;
      for (unsigned mask = 0; mask < (1u << samples) && ok; ++mask) {
         for (unsigned phase = 0; phase < 2 && ok; ++phase) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbos[0]);
            glViewport(0, 0, SIZE, SIZE);
            glDisable(GL_SAMPLE_MASK);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glClearDepth(1.0);
            glClearColor(0.0, 0.0, 1.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_SAMPLE_MASK);
            glSampleMaski(0, mask);
            glUseProgram(write);
            glUniform1i(glGetUniformLocation(write, "phase"), phase);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glDisable(GL_SAMPLE_MASK);
            for (unsigned pass = 0; pass < 2 && ok; ++pass) {
               if (pass) {
                  glBindFramebuffer(GL_FRAMEBUFFER, fbos[0]);
                  glViewport(0, 0, SIZE, SIZE);
                  glEnable(GL_DEPTH_TEST);
                  glUseProgram(probe);
                  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
               }
               glBindFramebuffer(GL_FRAMEBUFFER, fbos[1]);
               glViewport(0, 0, SIZE * samples, SIZE);
               glDisable(GL_DEPTH_TEST);
               glUseProgram(inspect);
               glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
               glReadPixels(0, 0, SIZE * samples, SIZE, GL_RGBA, GL_FLOAT, pixels);
               if (glGetError() != GL_NO_ERROR) {
                  fprintf(stderr, "sparse depth: %ux mask=%x phase=%u pass=%u GL error\n",
                          samples, mask, phase, pass);
                  ok = false;
                  break;
               }
               for (unsigned y = 0; y < SIZE && ok; ++y) {
                  for (unsigned x = 0; x < SIZE * samples && ok; ++x) {
                     unsigned sample = x / SIZE, px = x % SIZE;
                     bool written = (mask & (1u << sample)) &&
                                    ((px + y) & 1) == phase;
                     float expected[4] = {
                        written ? 1.0f : 0.0f,
                        !written && pass ? 1.0f : 0.0f,
                        !written && !pass ? 1.0f : 0.0f,
                        written ? 0.25f + 0.125f * (px & 1)
                                : pass ? 0.5f : 1.0f,
                     };
                     float *pixel = &pixels[4 * (y * SIZE * samples + x)];
                     for (unsigned c = 0; c < 4; ++c) {
                        if (!isfinite(pixel[c]) || fabsf(pixel[c] - expected[c]) > 0.00001f) {
                           fprintf(stderr, "sparse depth: %ux mask=%x phase=%u pass=%u sample=%u xy=%u,%u component=%u got=%g expected=%g\n",
                                   samples, mask, phase, pass, sample, px, y, c,
                                   pixel[c], expected[c]);
                           ok = false;
                           break;
                        }
                     }
                     checked++;
                  }
               }
            }
         }
         masks_checked++;
      }
      if (ok) {
         ++sample_counts;
         printf("sparse depth: %ux, %u masks, both discard phases and depth passes verified\n",
                samples, masks_checked);
      }
      glDeleteFramebuffers(2, fbos);
      glDeleteTextures(3, textures);
   }
   glDeleteProgram(write);
   glDeleteProgram(probe);
   glDeleteProgram(inspect);
   printf("sparse depth: %s, %u exact sample counts, %u sample observations\n",
          ok && sample_counts ? "PASS" : "FAIL", sample_counts, checked);
   return ok && sample_counts;
}

int
main(int argc, char **argv)
{
   const bool sparse_depth = argc == 2 && !strcmp(argv[1], "--sparse-depth");
   if (argc != 1 && !sparse_depth) {
      fprintf(stderr, "usage: %s [--sparse-depth]\n", argv[0]);
      return 1;
   }
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
   const EGLint surface_attributes[] = {EGL_WIDTH, 64, EGL_HEIGHT, 64,
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

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   if (sparse_depth)
      return run_sparse_depth_masks() ? 0 : 1;

   const char *only = getenv("ASAHI_GL_MSAA_FORMAT");
   bool all_ok = true;
   for (unsigned f = 0; f < sizeof(formats) / sizeof(formats[0]); ++f) {
   if (only && strcmp(only, formats[f].name))
      continue;
   /* A four-sample colour target, drawn to with full coverage. */
   GLuint ms_texture, ms_fbo;
   glGenTextures(1, &ms_texture);
   glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms_texture);
   glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES,
                           formats[f].internal_format, SIZE, SIZE, GL_TRUE);
   glGenFramebuffers(1, &ms_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, ms_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D_MULTISAMPLE, ms_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "msaa resolve: %-9s unsupported as a multisample "
                      "target\n", formats[f].name);
      continue;
   }

   const enum channel_kind kind = formats[f].kind;
   static const char *const fill_sources[] = {
      [CHANNEL_FLOAT] = "#version 330 core\n"
                        "out vec4 color;\n"
                        "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n",
      [CHANNEL_UINT] = "#version 330 core\n"
                       "out uvec4 color;\n"
                       "void main() { color = uvec4(255u, 0u, 0u, 255u); }\n",
      [CHANNEL_SINT] = "#version 330 core\n"
                       "out ivec4 color;\n"
                       "void main() { color = ivec4(127, 0, 0, 127); }\n",
   };
   GLuint fill = link_program(vs_quad, fill_sources[kind]);
   if (!fill)
      return 1;

   glViewport(0, 0, SIZE, SIZE);
   /* glClearColor is float state; an integer target takes its clear through
    * the typed entry point instead. */
   if (kind == CHANNEL_UINT) {
      const GLuint zero[4] = {0, 0, 0, 1};
      glClearBufferuiv(GL_COLOR, 0, zero);
   } else if (kind == CHANNEL_SINT) {
      const GLint zero[4] = {0, 0, 0, 1};
      glClearBufferiv(GL_COLOR, 0, zero);
   } else {
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
   }
   glUseProgram(fill);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "msaa resolve: the multisample draw raised an error\n");
      return 1;
   }

   if (getenv("ASAHI_GL_NOOP_SHIM") != NULL)
      continue;

   bool ok = true;

   /* Half one: did the draw reach every sample? Read each one back. */
   GLuint single_texture, single_fbo;
   glGenTextures(1, &single_texture);
   glBindTexture(GL_TEXTURE_2D, single_texture);
   glTexImage2D(GL_TEXTURE_2D, 0,
                kind == CHANNEL_UINT   ? GL_RGBA8UI
                : kind == CHANNEL_SINT ? GL_RGBA8I
                                       : GL_RGBA8,
                SIZE, SIZE, 0,
                kind == CHANNEL_FLOAT ? GL_RGBA : GL_RGBA_INTEGER,
                kind == CHANNEL_SINT ? GL_BYTE : GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &single_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, single_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          single_texture, 0);

   static const char *const read_sources[] = {
      [CHANNEL_FLOAT] =
         "#version 330 core\n"
         "uniform sampler2DMS source;\n"
         "uniform int sample_index;\n"
         "out vec4 color;\n"
         "void main()\n"
         "{\n"
         "    color = texelFetch(source, ivec2(gl_FragCoord.xy), sample_index);\n"
         "}\n",
      [CHANNEL_UINT] =
         "#version 330 core\n"
         "uniform usampler2DMS source;\n"
         "uniform int sample_index;\n"
         "out uvec4 color;\n"
         "void main()\n"
         "{\n"
         "    color = texelFetch(source, ivec2(gl_FragCoord.xy), sample_index);\n"
         "}\n",
      [CHANNEL_SINT] =
         "#version 330 core\n"
         "uniform isampler2DMS source;\n"
         "uniform int sample_index;\n"
         "out ivec4 color;\n"
         "void main()\n"
         "{\n"
         "    color = texelFetch(source, ivec2(gl_FragCoord.xy), sample_index);\n"
         "}\n",
   };
   GLuint read_sample = link_program(vs_quad, read_sources[kind]);
   if (!read_sample)
      return 1;
   glUseProgram(read_sample);
   glUniform1i(glGetUniformLocation(read_sample, "source"), 0);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms_texture);

   for (unsigned s = 0; s < SAMPLES; ++s) {
      glBindFramebuffer(GL_FRAMEBUFFER, single_fbo);
      if (kind == CHANNEL_UINT) {
         const GLuint green[4] = {0, 255, 0, 255};
         glClearBufferuiv(GL_COLOR, 0, green);
      } else if (kind == CHANNEL_SINT) {
         const GLint green[4] = {0, 127, 0, 127};
         glClearBufferiv(GL_COLOR, 0, green);
      } else {
         glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
      }
      glUniform1i(glGetUniformLocation(read_sample, "sample_index"), (GLint)s);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFinish();

      /* Read every kind back as signed 32-bit: it holds an unsigned byte and
       * a signed byte alike, so one comparison covers all three. */
      GLint pixel[4] = {0};
      glReadPixels(SIZE / 2, SIZE / 2, 1, 1,
                   kind == CHANNEL_FLOAT ? GL_RGBA : GL_RGBA_INTEGER,
                   kind == CHANNEL_FLOAT ? GL_UNSIGNED_BYTE : GL_INT, pixel);
      if (kind == CHANNEL_FLOAT) {
         const GLubyte *bytes = (const GLubyte *)pixel;
         pixel[3] = bytes[3]; pixel[2] = bytes[2];
         pixel[1] = bytes[1]; pixel[0] = bytes[0];
      }
      /* The drawn red is 255 unsigned, 127 signed; either way the red channel
       * is large and the green channel is not. */
      const GLint high = kind == CHANNEL_SINT ? 100 : 200;
      const bool red = pixel[0] >= high && pixel[1] < 60;
      if (!red)
         fprintf(stderr,
                 "msaa resolve: %-9s sample %u reads (%d, %d, %d, %d), the "
                 "draw did not reach it\n", formats[f].name, s, pixel[0],
                 pixel[1], pixel[2], pixel[3]);
      ok &= red;
   }

   /* Half two: does the resolve reproduce the drawn colour? Only the float
    * and normalized families have one to check -- GL defines no averaging
    * resolve for an integer target, and the conformance cases copy those with
    * a nearest blit between matching formats instead. Reaching every sample is
    * the whole question for them, and half one has answered it. */
   if (kind != CHANNEL_FLOAT) {
      fprintf(stderr, "msaa resolve: %-9s %s (samples only; an integer target "
                      "has no averaging resolve)\n",
              formats[f].name, ok ? "match" : "MISMATCH");
      all_ok &= ok;
      glDeleteFramebuffers(1, &single_fbo);
      glDeleteTextures(1, &single_texture);
      glDeleteFramebuffers(1, &ms_fbo);
      glDeleteTextures(1, &ms_texture);
      continue;
   }

   GLuint resolve_rb, resolve_fbo;
   glGenRenderbuffers(1, &resolve_rb);
   glBindRenderbuffer(GL_RENDERBUFFER, resolve_rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, SIZE, SIZE);
   glGenFramebuffers(1, &resolve_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, resolve_rb);
   glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   glBindFramebuffer(GL_READ_FRAMEBUFFER, ms_fbo);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo);
   glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE, GL_COLOR_BUFFER_BIT,
                     GL_NEAREST);
   glFinish();

   /* A format the driver will not resolve is a result, not a reason to stop:
    * the sweep exists to say which families reach every sample and resolve,
    * so record it and carry on to the next one. */
   const bool blit_ok = glGetError() == GL_NO_ERROR;
   if (!blit_ok)
      fprintf(stderr, "msaa resolve: %-9s the resolve blit raised an error\n",
              formats[f].name);

   bool resolved_red = false;
   if (blit_ok) {
      glBindFramebuffer(GL_READ_FRAMEBUFFER, resolve_fbo);
      GLubyte resolved[4] = {0};
      glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                   resolved);
      resolved_red = resolved[0] > 200 && resolved[1] < 60;
      if (!resolved_red)
         fprintf(stderr,
                 "msaa resolve: %-9s resolved pixel is (%u, %u, %u, %u), the "
                 "resolve lost it\n", formats[f].name, resolved[0],
                 resolved[1], resolved[2], resolved[3]);
   }
   ok &= resolved_red;

   fprintf(stderr, "msaa resolve: %-9s %s\n", formats[f].name,
           ok ? "match" : "MISMATCH");
   all_ok &= ok;

   glDeleteFramebuffers(1, &resolve_fbo);
   glDeleteRenderbuffers(1, &resolve_rb);
   glDeleteFramebuffers(1, &single_fbo);
   glDeleteTextures(1, &single_texture);
   glDeleteFramebuffers(1, &ms_fbo);
   glDeleteTextures(1, &ms_texture);
   glDeleteProgram(read_sample);
   }

   return all_ok ? 0 : 1;
}
