/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.transform_feedback.capture_vertex_interleaved_test, reduced to a
 * standalone oracle. The vertex shader writes fifteen constant vec4 varyings
 * plus a position chosen by gl_VertexID; all of them, with gl_Position last,
 * are captured interleaved into one buffer while the same draw renders into a
 * one-channel float framebuffer. The conformance case checks both and reports
 * only "Fail", so this oracle reports which half is wrong and where.
 *
 * KHR-GL30.transform_feedback.discard_vertex_test reuses the same program.
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

#define COMPONENTS   64
#define VARYINGS     ((COMPONENTS / 4) - 1)
#define FB_SIZE      2
#define EPSILON      0.0625f
#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint compiled = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
   if (!compiled) {
      char log[4096] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "xfb interleaved: shader compile failed: %s\n", log);
      return 0;
   }
   return shader;
}

static GLuint
make_program(void)
{
   char vs_source[16384];
   int n = snprintf(vs_source, sizeof(vs_source), "#version 130\n\n");

   for (unsigned i = 0; i < VARYINGS; ++i)
      n += snprintf(vs_source + n, sizeof(vs_source) - n,
                    "out vec4 result_%u;\n", i);

   n += snprintf(vs_source + n, sizeof(vs_source) - n, "\nvoid main()\n{\n");

   for (unsigned i = 0; i < VARYINGS; ++i)
      n += snprintf(vs_source + n, sizeof(vs_source) - n,
                    "    result_%u = vec4(%u.0, %u.0, %u.0, %u.0);\n", i,
                    i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3);

   n += snprintf(
      vs_source + n, sizeof(vs_source) - n,
      "\n"
      "    vec4 position = vec4(0.0);\n"
      "    switch (gl_VertexID) {\n"
      "    case 0: position = vec4(-1.0 + %f,  1.0 - %f, 0.0, 1.0); break;\n"
      "    case 1: position = vec4( 1.0 - %f,  1.0 - %f, 0.0, 1.0); break;\n"
      "    case 2: position = vec4(-1.0 + %f, -1.0 + %f, 0.0, 1.0); break;\n"
      "    case 3: position = vec4( 1.0 - %f, -1.0 + %f, 0.0, 1.0); break;\n"
      "    }\n"
      "    gl_Position = position;\n"
      "}\n",
      EPSILON, EPSILON, EPSILON, EPSILON, EPSILON, EPSILON, EPSILON, EPSILON);

   static const char fs_source[] =
      "#version 130\n"
      "\n"
      "out vec4 color;\n"
      "\n"
      "void main()\n"
      "{\n"
      "    color = vec4(0.5);\n"
      "}\n";

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 0;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glDeleteShader(vs);
   glDeleteShader(fs);

   /* The conformance case captures gl_Position last, so it lands in the final
    * four components of every vertex's stride. */
   char names[VARYINGS + 1][16];
   const char *varyings[VARYINGS + 1];
   for (unsigned i = 0; i < VARYINGS; ++i) {
      snprintf(names[i], sizeof(names[i]), "result_%u", i);
      varyings[i] = names[i];
   }
   varyings[VARYINGS] = "gl_Position";

   glTransformFeedbackVaryings(program, VARYINGS + 1, varyings,
                               GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[4096] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "xfb interleaved: link failed: %s\n", log);
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

   GLint max_components = 0;
   glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS,
                 &max_components);
   GLint max_varying = 0;
   glGetIntegerv(GL_MAX_VARYING_COMPONENTS, &max_varying);
   if (max_components < COMPONENTS || max_varying < COMPONENTS) {
      fprintf(stderr,
              "xfb interleaved: limits too small (interleaved %d, varying %d)\n",
              max_components, max_varying);
      return 1;
   }

   GLuint program = make_program();
   if (!program)
      return 1;

   GLuint renderbuffer, framebuffer;
   glGenRenderbuffers(1, &renderbuffer);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_R32F, FB_SIZE, FB_SIZE);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, renderbuffer);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "xfb interleaved: the framebuffer is incomplete\n");
      return 1;
   }
   glViewport(0, 0, FB_SIZE, FB_SIZE);

   GLuint vao, xfb_buffer;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &xfb_buffer);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb_buffer);
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
                8 * COMPONENTS * sizeof(GLfloat), NULL, GL_DYNAMIC_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb_buffer);

   /* The conformance case walks every primitive the capture supports, drawn
    * with indices, and captures the decomposed points, lines or triangles. */
   static const struct {
      const char *name;
      GLenum mode, captured;
      GLuint count, vertices;
      GLuint indices[6];
   } cases[] = {
      {"points", GL_POINTS, GL_POINTS, 4, 4, {0, 1, 2, 3}},
      {"lines", GL_LINES, GL_LINES, 4, 4, {0, 1, 2, 3}},
      {"line loop", GL_LINE_LOOP, GL_LINES, 4, 8, {0, 1, 3, 2}},
      {"line strip", GL_LINE_STRIP, GL_LINES, 4, 6, {0, 1, 3, 2}},
      {"triangles", GL_TRIANGLES, GL_TRIANGLES, 6, 6, {2, 0, 1, 2, 1, 3}},
      {"triangle strip", GL_TRIANGLE_STRIP, GL_TRIANGLES, 4, 6, {0, 1, 2, 3}},
      {"triangle fan", GL_TRIANGLE_FAN, GL_TRIANGLES, 4, 6, {2, 0, 1, 3}},
   };

   GLuint elements;
   glGenBuffers(1, &elements);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements);
   glUseProgram(program);

   bool ok = true;
   for (unsigned c = 0; c < ARRAY_LENGTH(cases); ++c) {
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cases[c].indices),
                   cases[c].indices, GL_STATIC_DRAW);
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glBeginTransformFeedback(cases[c].captured);
      glDrawElements(cases[c].mode, cases[c].count, GL_UNSIGNED_INT, NULL);
      glEndTransformFeedback();
      glFinish();
      if (glGetError() != GL_NO_ERROR) {
         fprintf(stderr, "xfb interleaved: %s raised a GL error\n",
                 cases[c].name);
         return 1;
      }

      /* The no-op shim submits nothing, so only a real GPU has results. */
      if (getenv("ASAHI_GL_NOOP_SHIM") != NULL)
         continue;

      bool case_ok = true;
      GLfloat pixels[FB_SIZE * FB_SIZE];
      glReadPixels(0, 0, FB_SIZE, FB_SIZE, GL_RED, GL_FLOAT, pixels);
      for (unsigned i = 0; i < FB_SIZE * FB_SIZE; ++i) {
         if (fabsf(pixels[i] - 0.5f) > 0.0625f) {
            fprintf(stderr, "xfb interleaved: %s framebuffer pixel %u is %f, "
                            "expected 0.5\n", cases[c].name, i, pixels[i]);
            case_ok = false;
            break;
         }
      }

      const GLfloat *captured =
         glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
      if (!captured) {
         fprintf(stderr, "xfb interleaved: %s capture did not map\n",
                 cases[c].name);
         return 1;
      }
      if (getenv("ASAHI_GL_XFB_DUMP")) {
         for (unsigned v = 0; v < cases[c].vertices; ++v) {
            fprintf(stderr, "xfb interleaved: %s vertex %u:", cases[c].name, v);
            for (unsigned i = 0; i < COMPONENTS; ++i)
               fprintf(stderr, " %g", captured[v * COMPONENTS + i]);
            fprintf(stderr, "\n");
         }
      }
      for (unsigned v = 0; v < cases[c].vertices && case_ok; ++v) {
         const GLfloat *vertex = captured + v * COMPONENTS;
         for (unsigned i = 0; i < COMPONENTS - 4; ++i) {
            if (fabsf(vertex[i] - (GLfloat)i) > 0.125f) {
               fprintf(stderr,
                       "xfb interleaved: %s vertex %u component %u is %f, "
                       "expected %u\n", cases[c].name, v, i, vertex[i], i);
               case_ok = false;
               break;
            }
         }
         const GLfloat *position = vertex + COMPONENTS - 4;
         if (case_ok &&
             (fabsf(fabsf(position[0]) - 1.0f + EPSILON) > 0.125f ||
              fabsf(fabsf(position[1]) - 1.0f + EPSILON) > 0.125f ||
              fabsf(position[2]) > 0.125f ||
              fabsf(position[3] - 1.0f) > 0.125f)) {
            fprintf(stderr,
                    "xfb interleaved: %s vertex %u position is "
                    "(%f, %f, %f, %f)\n", cases[c].name, v, position[0],
                    position[1], position[2], position[3]);
            case_ok = false;
         }
      }
      glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);

      fprintf(stderr, "xfb interleaved: %-15s %s\n", cases[c].name,
              case_ok ? "match" : "MISMATCH");
      ok &= case_ok;
   }

   return ok ? 0 : 1;
}
