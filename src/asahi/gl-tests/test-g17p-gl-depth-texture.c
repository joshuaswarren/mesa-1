/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <GLES3/gl32.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 64
#define HEIGHT 64

static bool
gl_ok(const char *where)
{
   bool ok = true;
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "%s: GL error %#x\n", where, error);
      ok = false;
   }
   return ok;
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
      fprintf(stderr, "depth-texture shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *vs_source, const char *fs_source)
{
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
      fprintf(stderr, "depth-texture program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
pixel_ok(const uint8_t *pixel, bool noop)
{
   if (noop)
      return true;

   /* D24 quantization leaves 0.25 exact. LEQUAL(0.2, 0.25) is true. */
   const bool ok = pixel[0] >= 63 && pixel[0] <= 65 && pixel[1] == 255 &&
                   pixel[2] == 0 && pixel[3] == 255;
   if (!ok) {
      fprintf(stderr, "depth sample got {%u,%u,%u,%u}\n", pixel[0], pixel[1],
              pixel[2], pixel[3]);
   }
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
      return 1;

   const EGLint config_attributes[] = {
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
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;

   const EGLint surface_attributes[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   static const char depth_vs[] =
      "#version 300 es\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], -0.5, 1.0); }\n";
   static const char depth_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "void main() { }\n";
   static const char sample_vs[] =
      "#version 300 es\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
   static const char sample_fs[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform highp sampler2D raw_depth;\n"
      "uniform highp sampler2DShadow compare_depth;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() {\n"
      "  float raw = texture(raw_depth, vec2(0.5)).r;\n"
      "  float cmp = texture(compare_depth, vec3(0.5, 0.5, 0.2));\n"
      "  colour = vec4(raw, cmp, 0.0, 1.0);\n"
      "}\n";

   GLuint depth_program = make_program(depth_vs, depth_fs);
   GLuint sample_program = make_program(sample_vs, sample_fs);
   if (!depth_program || !sample_program)
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   GLuint depth_texture[2], depth_fbo[2];
   glGenTextures(2, depth_texture);
   glGenFramebuffers(2, depth_fbo);
   const GLenum none = GL_NONE;
   for (unsigned i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, depth_texture[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_TEXTURE_2D, depth_texture[i], 0);
      glDrawBuffers(1, &none);
      glReadBuffer(GL_NONE);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         return 1;
   }

   glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo[0]);
   glViewport(0, 0, WIDTH, HEIGHT);
   glClearDepthf(0.75f);
   glClearStencil(0x5a);
   glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_ALWAYS);
   glDepthMask(GL_TRUE);
   glUseProgram(depth_program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_DEPTH_TEST);
   glFinish();
   if (!gl_ok("depth texture render"))
      return 1;

   glBindFramebuffer(GL_READ_FRAMEBUFFER, depth_fbo[0]);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, depth_fbo[1]);
   glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT,
                     GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
   glFinish();
   if (!gl_ok("depth-stencil framebuffer blit"))
      return 1;

   GLuint colour_texture, colour_fbo;
   glGenTextures(1, &colour_texture);
   glBindTexture(GL_TEXTURE_2D, colour_texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &colour_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, colour_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          colour_texture, 0);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                          GL_TEXTURE_2D, depth_texture[1], 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 1;

   GLuint sampler[2];
   glGenSamplers(2, sampler);
   for (unsigned i = 0; i < 2; ++i) {
      glSamplerParameteri(sampler[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glSamplerParameteri(sampler[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glSamplerParameteri(sampler[i], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glSamplerParameteri(sampler[i], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   }
   glSamplerParameteri(sampler[0], GL_TEXTURE_COMPARE_MODE, GL_NONE);
   glSamplerParameteri(sampler[1], GL_TEXTURE_COMPARE_MODE,
                       GL_COMPARE_REF_TO_TEXTURE);
   glSamplerParameteri(sampler[1], GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
   for (unsigned i = 0; i < 2; ++i) {
      glActiveTexture(GL_TEXTURE0 + i);
      glBindTexture(GL_TEXTURE_2D, depth_texture[1]);
      glBindSampler(i, sampler[i]);
   }

   glUseProgram(sample_program);
   glUniform1i(glGetUniformLocation(sample_program, "raw_depth"), 0);
   glUniform1i(glGetUniformLocation(sample_program, "compare_depth"), 1);
   /* Make the copied stencil plane observable without relying on a
    * depth/stencil glReadPixels format that GLES does not guarantee.  The
    * colour remains red unless the destination retained the source's 0x5a
    * stencil value; a passing stencil test replaces it with the raw/shadow
    * depth sample checked below. */
   glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_STENCIL_TEST);
   glStencilFunc(GL_EQUAL, 0x5a, 0xff);
   glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
   glStencilMask(0x00);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_STENCIL_TEST);
   glFinish();
   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   glFinish();
   if (!gl_ok("depth texture raw/compare sample") || !pixel_ok(pixel, noop))
      return 1;

   glDeleteSamplers(2, sampler);
   glDeleteFramebuffers(1, &colour_fbo);
   glDeleteTextures(1, &colour_texture);
   glDeleteFramebuffers(2, depth_fbo);
   glDeleteTextures(2, depth_texture);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(sample_program);
   glDeleteProgram(depth_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "G17P WebGL2 depth texture render/blit/sample submitted"
             : "G17P WebGL2 depth texture render/blit/sample pixels exact");
   return 0;
}
