/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL33.transform_feedback.{capture,query,discard}_geometry_interleaved_test
 * link a geometry shader that writes gl_Position plus as many vec4 varyings as
 * GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS allows, behind a vertex
 * shader that writes nothing at all. On G17P all three report
 *
 *   asahi: the AGX backend cannot encode the default geometry shader variant
 *
 * from glLinkProgram(), so the cases never reach a draw. The widest shape the
 * limits admit is the one that refuses, so the test walks the varying count up
 * and reports the first width that fails rather than only the maximum.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The CTS emits "points", "line_strip" and "triangle_strip" from one template;
 * the output primitive does not change the varying shape, so the reproducer
 * keeps the first case and varies the width instead. */
static const char vs_source[] =
   "#version 130\n"
   "\n"
   "void main()\n"
   "{\n"
   "}\n";

static const char fs_source[] =
   "#version 130\n"
   "\n"
   "out vec4 color;\n"
   "\n"
   "void main()\n"
   "{\n"
   "    color = vec4(0.5);\n"
   "}\n";

/* A shim control run must not silently answer from a different chip. */
static bool
renderer_is_expected(void)
{
   const char *expect = getenv("ASAHI_GL_EXPECT_RENDERER");
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   if (expect && (!renderer || !strstr(renderer, expect))) {
      fprintf(stderr,
              "xfb geometry: expected a renderer containing '%s', got '%s'\n",
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
      fprintf(stderr, "xfb geometry: shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

/* Build the CTS's geometry shader for a given user varying count. */
static char *
geometry_source(unsigned varyings)
{
   size_t size = 4096 + varyings * 256;
   char *src = calloc(1, size);
   if (!src)
      return NULL;

   char *decls = calloc(1, varyings * 32 + 1);
   char *setters = calloc(1, varyings * 64 + 1);
   if (!decls || !setters) {
      free(src);
      free(decls);
      free(setters);
      return NULL;
   }

   for (unsigned i = 0; i < varyings; ++i) {
      char line[64];
      snprintf(line, sizeof(line), "out vec4 result_%u;\n", i);
      strcat(decls, line);
      snprintf(line, sizeof(line), "    result_%u = vec4(%u.0, %u.0, %u.0, %u.0);\n",
               i, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3);
      strcat(setters, line);
   }

   snprintf(src, size,
            "#version 150\n"
            "\n"
            "layout(points) in;\n"
            "layout(points, max_vertices = 4) out;\n"
            "\n"
            "%s"
            "\n"
            "void main()\n"
            "{\n"
            "    gl_Position = vec4(-0.9375,  0.9375, 0.0, 1.0);\n"
            "%s"
            "    EmitVertex();\n"
            "\n"
            "    gl_Position = vec4( 0.9375,  0.9375, 0.0, 1.0);\n"
            "%s"
            "    EmitVertex();\n"
            "\n"
            "    gl_Position = vec4(-0.9375, -0.9375, 0.0, 1.0);\n"
            "%s"
            "    EmitVertex();\n"
            "\n"
            "    gl_Position = vec4( 0.9375, -0.9375, 0.0, 1.0);\n"
            "%s"
            "    EmitVertex();\n"
            "}\n",
            decls, setters, setters, setters, setters);

   free(decls);
   free(setters);
   return src;
}

/* Returns true when the program links. On failure the info log is copied out
 * so the caller can tell a compiler refusal apart from a limits rejection.
 * On success the program is returned through out_program when asked for, and
 * deleted otherwise. */
static bool
try_link(unsigned varyings, char *log, size_t log_size, GLuint *out_program)
{
   char *gs_source = geometry_source(varyings);
   if (!gs_source) {
      snprintf(log, log_size, "out of memory");
      return false;
   }

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   free(gs_source);
   if (!vs || !gs || !fs)
      return false;

   /* The CTS names result_0..result_n-1 first and gl_Position last. */
   const char **names = calloc(varyings + 1, sizeof(*names));
   char *storage = calloc(varyings + 1, 32);
   if (!names || !storage) {
      snprintf(log, log_size, "out of memory");
      free(names);
      free(storage);
      return false;
   }
   for (unsigned i = 0; i < varyings; ++i) {
      names[i] = storage + i * 32;
      snprintf(storage + i * 32, 32, "result_%u", i);
   }
   names[varyings] = "gl_Position";

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, gs);
   glAttachShader(program, fs);
   glTransformFeedbackVaryings(program, varyings + 1, names,
                               GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(gs);
   glDeleteShader(fs);
   free(names);
   free(storage);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   log[0] = '\0';
   if (!linked)
      glGetProgramInfoLog(program, (GLsizei)log_size, NULL, log);

   if (linked && out_program)
      *out_program = program;
   else
      glDeleteProgram(program);
   return linked;
}

/* Capture one point through the widest program and check what the geometry
 * shader said it wrote. Four vertices are emitted per input point, each with
 * gl_Position last, so the capture is (varyings + 1) * 4 floats per vertex. */
static bool
capture(GLuint program, unsigned varyings)
{
   const unsigned components = (varyings + 1) * 4;
   const unsigned vertices = 4;

   GLuint vao, xfb_buffer;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &xfb_buffer);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb_buffer);
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
                (GLsizeiptr)(vertices * components * sizeof(GLfloat)), NULL,
                GL_DYNAMIC_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb_buffer);

   glUseProgram(program);
   glEnable(GL_RASTERIZER_DISCARD);
   glBeginTransformFeedback(GL_POINTS);
   glDrawArrays(GL_POINTS, 0, 1);
   glEndTransformFeedback();
   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();

   bool ok = true;
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb geometry: the capture draw raised a GL error\n");
      ok = false;
   } else if (getenv("ASAHI_GL_NOOP_SHIM") == NULL) {
      /* The no-op shim submits nothing, so only a real GPU can be checked. */
      GLfloat *got = calloc(vertices * components, sizeof(GLfloat));
      glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                         (GLsizeiptr)(vertices * components * sizeof(GLfloat)),
                         got);
      for (unsigned v = 0; v < vertices && ok; ++v) {
         for (unsigned i = 0; i < varyings && ok; ++i) {
            for (unsigned c = 0; c < 4; ++c) {
               const GLfloat want = (GLfloat)(i * 4 + c);
               const GLfloat have = got[v * components + i * 4 + c];
               if (have != want) {
                  fprintf(stderr,
                          "xfb geometry: vertex %u result_%u[%u]: %f, "
                          "expected %f\n", v, i, c, have, want);
                  ok = false;
                  break;
               }
            }
         }
      }
      free(got);
   }

   glDeleteBuffers(1, &xfb_buffer);
   glDeleteVertexArrays(1, &vao);
   return ok;
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
      EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE,
   };
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

   printf("xfb geometry: renderer %s\n", (const char *)glGetString(GL_RENDERER));

   if (!renderer_is_expected())
      return 1;

   /* The CTS's own limit arithmetic, verbatim. */
   GLint interleaved = 0, geometry_total = 0;
   glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS,
                 &interleaved);
   glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS, &geometry_total);
   if (interleaved <= 0 || geometry_total <= 0) {
      fprintf(stderr, "xfb geometry: limits are zero (%d, %d)\n", interleaved,
              geometry_total);
      return 1;
   }
   if (interleaved * 4 > geometry_total)
      interleaved = geometry_total / 4;

   const int cts_varyings = interleaved / 4 - 1;
   if (cts_varyings < 1) {
      fprintf(stderr, "xfb geometry: limits leave no user varyings\n");
      return 1;
   }
   printf("xfb geometry: interleaved %d, geometry total %d, CTS width %d\n",
          interleaved, geometry_total, cts_varyings);

   int first_refused = -1;
   for (int varyings = 1; varyings <= cts_varyings; ++varyings) {
      char log[2048];
      if (try_link((unsigned)varyings, log, sizeof(log), NULL)) {
         printf("  %2d user varyings: links\n", varyings);
         continue;
      }
      printf("  %2d user varyings: REFUSED: %s\n", varyings, log);
      if (first_refused < 0)
         first_refused = varyings;
      break;
   }

   /* The widest program is the one the CTS builds. Run its capture, so the
    * oracle covers the submitted stream and not only the link. */
   bool captured = false;
   if (first_refused < 0) {
      char log[2048];
      GLuint program = 0;
      if (try_link((unsigned)cts_varyings, log, sizeof(log), &program)) {
         captured = capture(program, (unsigned)cts_varyings);
         glDeleteProgram(program);
      } else {
         fprintf(stderr, "xfb geometry: relink refused: %s\n", log);
      }
   }

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (first_refused >= 0) {
      fprintf(stderr,
              "xfb geometry: the geometry shader stops encoding at %d user "
              "varyings; the CTS needs %d\n",
              first_refused, cts_varyings);
      return 1;
   }
   printf("xfb geometry: every width up to the CTS's %d encodes\n",
          cts_varyings);
   return captured ? 0 : 1;
}
