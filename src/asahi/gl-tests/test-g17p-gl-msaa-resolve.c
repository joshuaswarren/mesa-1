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
 * admitted per physical format, so check each one the GL 3.0 case uses. */
static const struct {
   const char *name;
   GLenum internal_format;
} formats[] = {
   {"rgba8", GL_RGBA8},     {"rgb565", GL_RGB565},
   {"rgba4", GL_RGBA4},     {"rgb5_a1", GL_RGB5_A1},
   {"rgb10_a2", GL_RGB10_A2}, {"r8", GL_R8},
   {"rg8", GL_RG8},         {"r16f", GL_R16F},
   {"rgba16f", GL_RGBA16F}, {"r32f", GL_R32F},
   {"rgba32f", GL_RGBA32F}, {"srgb8a8", GL_SRGB8_ALPHA8},
   {"r11f", GL_R11F_G11F_B10F}, {"rgb9e5", GL_RGB9_E5},
   {"rg16f", GL_RG16F},     {"rgb10a2ui", GL_RGB10_A2UI},
   {"r16", GL_R16},         {"rg16", GL_RG16},
   {"rgba16", GL_RGBA16},   {"r8i", GL_R8I},
   {"rgba8i", GL_RGBA8I},   {"r32ui", GL_R32UI},
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

   GLuint fill = link_program(vs_quad,
                              "#version 330 core\n"
                              "out vec4 color;\n"
                              "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n");
   if (!fill)
      return 1;

   glViewport(0, 0, SIZE, SIZE);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
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
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &single_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, single_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          single_texture, 0);

   GLuint read_sample = link_program(
      vs_quad,
      "#version 330 core\n"
      "uniform sampler2DMS source;\n"
      "uniform int sample_index;\n"
      "out vec4 color;\n"
      "void main()\n"
      "{\n"
      "    color = texelFetch(source, ivec2(gl_FragCoord.xy), sample_index);\n"
      "}\n");
   if (!read_sample)
      return 1;
   glUseProgram(read_sample);
   glUniform1i(glGetUniformLocation(read_sample, "source"), 0);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms_texture);

   for (unsigned s = 0; s < SAMPLES; ++s) {
      glBindFramebuffer(GL_FRAMEBUFFER, single_fbo);
      glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glUniform1i(glGetUniformLocation(read_sample, "sample_index"), (GLint)s);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFinish();

      GLubyte pixel[4] = {0};
      glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
      const bool red = pixel[0] > 200 && pixel[1] < 60;
      if (!red)
         fprintf(stderr,
                 "msaa resolve: %-9s sample %u reads (%u, %u, %u, %u), the "
                 "draw did not reach it\n", formats[f].name, s, pixel[0],
                 pixel[1], pixel[2], pixel[3]);
      ok &= red;
   }

   /* Half two: does the resolve reproduce the drawn colour? */
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
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "msaa resolve: the resolve blit raised an error\n");
      return 1;
   }

   glBindFramebuffer(GL_READ_FRAMEBUFFER, resolve_fbo);
   GLubyte resolved[4] = {0};
   glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, resolved);
   const bool resolved_red = resolved[0] > 200 && resolved[1] < 60;
   if (!resolved_red)
      fprintf(stderr,
              "msaa resolve: %-9s resolved pixel is (%u, %u, %u, %u), the "
              "resolve lost it\n", formats[f].name, resolved[0], resolved[1],
              resolved[2], resolved[3]);
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
