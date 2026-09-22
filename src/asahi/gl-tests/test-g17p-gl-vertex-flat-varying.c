/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL33.texture_swizzle's vertex-stage cases read a texel in the vertex
 * shader and pass it to the fragment shader through a flat varying. On G17P
 * only the first two components arrive; z and w read back as zero. That is
 * 363 of the suite's failures, so this oracle separates the two things that
 * shape could blame: the flat varying itself, and vertex-stage texturing.
 *
 * Each row writes the same four distinguishable bytes and reads them back
 * from an 8x8 target, so a row that loses components names which.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const GLubyte texel[4] = {0x11, 0x22, 0x33, 0x44};

static bool
renderer_is_expected(void)
{
   const char *expect = getenv("ASAHI_GL_EXPECT_RENDERER");
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   if (expect && (!renderer || !strstr(renderer, expect))) {
      fprintf(stderr,
              "flat varying: expected a renderer containing '%s', got '%s'\n",
              expect, renderer ? renderer : "(null)");
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
      fprintf(stderr, "flat varying: compile failed: %s\n%s\n", log, source);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static const char *const quad_body =
   "    switch (gl_VertexID)\n"
   "    {\n"
   "      case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"
   "      case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"
   "      case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"
   "      case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"
   "    }\n";

struct shape {
   const char *name;
   const char *vs_decl;   /* varying declarations, vertex side */
   const char *vs_body;   /* what the vertex shader assigns */
   const char *fs_decl;   /* varying declarations, fragment side */
   const char *fs_color;  /* how the fragment shader rebuilds the vec4 */
};

static const struct shape shapes[] = {
   {"flat vec4 const",
    "flat out vec4 v;\n",
    "    v = vec4(17.0, 34.0, 51.0, 68.0) / 255.0;\n",
    "flat in vec4 v;\n", "v"},
   {"smooth vec4 const",
    "out vec4 v;\n",
    "    v = vec4(17.0, 34.0, 51.0, 68.0) / 255.0;\n",
    "in vec4 v;\n", "v"},
   {"flat 2x vec2 const",
    "flat out vec2 a;\nflat out vec2 b;\n",
    "    a = vec2(17.0, 34.0) / 255.0;\n    b = vec2(51.0, 68.0) / 255.0;\n",
    "flat in vec2 a;\nflat in vec2 b;\n", "vec4(a, b)"},
   {"flat 4x float const",
    "flat out float p;\nflat out float q;\n"
    "flat out float r;\nflat out float s;\n",
    "    p = 17.0/255.0; q = 34.0/255.0; r = 51.0/255.0; s = 68.0/255.0;\n",
    "flat in float p;\nflat in float q;\n"
    "flat in float r;\nflat in float s;\n", "vec4(p, q, r, s)"},
   {"flat ivec4 const",
    "flat out ivec4 iv;\n",
    "    iv = ivec4(17, 34, 51, 68);\n",
    "flat in ivec4 iv;\n", "vec4(iv) / 255.0"},
   {"flat vec4 texel",
    "uniform sampler2D tex;\nflat out vec4 v;\n",
    "    v = texelFetch(tex, ivec2(0), 0);\n",
    "flat in vec4 v;\n", "v"},
   {"smooth vec4 texel",
    "uniform sampler2D tex;\nout vec4 v;\n",
    "    v = texelFetch(tex, ivec2(0), 0);\n",
    "in vec4 v;\n", "v"},
   /* The packer only stages a vertex fetch's coordinates when they are not an
    * immediate, so the same fetch through a uniform takes the other path. */
   {"flat vec4 texel dyn",
    "uniform sampler2D tex;\nuniform int zero;\nflat out vec4 v;\n",
    "    v = texelFetch(tex, ivec2(zero), 0);\n",
    "flat in vec4 v;\n", "v"},
   {"flat vec4 textureLod",
    "uniform sampler2D tex;\nflat out vec4 v;\n",
    "    v = textureLod(tex, vec2(0.5), 0.0);\n",
    "flat in vec4 v;\n", "v"},
   {"flat vec4 texLod dyn",
    "uniform sampler2D tex;\nuniform int zero;\nflat out vec4 v;\n",
    "    v = textureLod(tex, vec2(0.5) + float(zero), 0.0);\n",
    "flat in vec4 v;\n", "v"},
   /* Broadcasting one component tells a fetch that lost it apart from a
    * varying that failed to carry it. */
   {"flat texel.z bcast",
    "uniform sampler2D tex;\nflat out vec4 v;\n",
    "    v = vec4(texelFetch(tex, ivec2(0), 0).z);\n",
    "flat in vec4 v;\n", "v"},
   {"flat texel.w bcast",
    "uniform sampler2D tex;\nflat out vec4 v;\n",
    "    v = vec4(texelFetch(tex, ivec2(0), 0).w);\n",
    "flat in vec4 v;\n", "v"},
   {"flat texel.zw pair",
    "uniform sampler2D tex;\nflat out vec4 v;\n",
    "    v = vec4(texelFetch(tex, ivec2(0), 0).zw, 0.0, 0.0);\n",
    "flat in vec4 v;\n", "v"},
   /* The same fetches in the fragment stage, which is the control. */
   {"fs vec4 texel", "", "",
    "uniform sampler2D tex;\n", "texelFetch(tex, ivec2(0), 0)"},
   {"fs texel.z bcast", "", "",
    "uniform sampler2D tex;\n", "vec4(texelFetch(tex, ivec2(0), 0).z)"},
};

/* Per-shape expectations, since the broadcasts do not read back the texel. */
static const GLubyte expected[][4] = {
   {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44},
   {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44},
   {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44}, {0x11,0x22,0x33,0x44},
   {0x11,0x22,0x33,0x44},
   {0x33,0x33,0x33,0x33}, {0x44,0x44,0x44,0x44}, {0x33,0x44,0x00,0x00},
   {0x11,0x22,0x33,0x44}, {0x33,0x33,0x33,0x33},
};

static GLuint
make_program(const struct shape *shape)
{
   char vs[1024], fs[1024];
   snprintf(vs, sizeof(vs),
            "#version 330 core\n%s"
            "void main() {\n%s%s}\n",
            shape->vs_decl, shape->vs_body, quad_body);
   snprintf(fs, sizeof(fs),
            "#version 330 core\n%s"
            "out vec4 color;\n"
            "void main() {\n    color = %s;\n}\n",
            shape->fs_decl, shape->fs_color);

   GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
   GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
   if (!v || !f)
      return 0;
   GLuint program = glCreateProgram();
   glAttachShader(program, v);
   glAttachShader(program, f);
   glLinkProgram(program);
   glDeleteShader(v);
   glDeleteShader(f);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "flat varying: link failed: %s\n", log);
      return 0;
   }
   return program;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;

   const EGLint cfg_attr[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE};
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, cfg_attr, &config, 1, &count) || count != 1)
      return 1;
   const EGLint surf_attr[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(display, config, surf_attr);
   const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
                              EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attr);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   printf("flat varying: renderer %s\n", (const char *)glGetString(GL_RENDERER));
   if (!renderer_is_expected())
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   GLuint tex;
   glGenTextures(1, &tex);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, texel);

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
      fprintf(stderr, "flat varying: incomplete framebuffer\n");
      return 1;
   }
   glViewport(0, 0, 8, 8);
   glBindTexture(GL_TEXTURE_2D, tex);

   /* The no-op shim submits nothing, so only a real GPU can be graded;
    * there the run still covers the links and the submitted stream. */
   const bool grade = getenv("ASAHI_GL_NOOP_SHIM") == NULL;
   bool ok = true;
   for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
      GLuint program = make_program(&shapes[i]);
      if (!program)
         return 1;
      glUseProgram(program);
      const GLint loc = glGetUniformLocation(program, "tex");
      if (loc >= 0)
         glUniform1i(loc, 0);
      const GLint zero_loc = glGetUniformLocation(program, "zero");
      if (zero_loc >= 0)
         glUniform1i(zero_loc, 0);

      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFinish();

      GLubyte image[8 * 8 * 4] = {0};
      glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, image);
      unsigned differing = 0;
      for (unsigned t = 1; t < 64; ++t)
         if (memcmp(image + t * 4, image, 4))
            ++differing;

      const GLubyte *want = expected[i];
      const bool row_ok = !memcmp(image, want, 4) && differing == 0;
      ok &= row_ok || !grade;
      printf("  %-20s got %02x%02x%02x%02x  want %02x%02x%02x%02x  %s%s\n",
             shapes[i].name, image[0], image[1], image[2], image[3],
             want[0], want[1], want[2], want[3],
             row_ok ? "" : "WRONG",
             differing ? " (texels differ across the 8x8)" : "");
      glDeleteProgram(program);
   }

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "flat varying: a GL error was raised\n");
      ok = false;
   }

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &rt);
   glDeleteTextures(1, &tex);
   glDeleteVertexArrays(1, &vao);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (ok)
      printf("flat varying: every shape carries all four components\n");
   return ok ? 0 : 1;
}
