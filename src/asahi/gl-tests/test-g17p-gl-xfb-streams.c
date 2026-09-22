/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.transform_feedback3.multiple_streams links a vertex and a
 * two-stream geometry shader with no fragment shader, then captures both
 * streams under GL_RASTERIZER_DISCARD. The geometry copy program still runs
 * as a hardware vertex shader, so its varying stores must not depend on a
 * fragment shader that can never be attached.
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
      fprintf(stderr, "xfb streams: shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(void)
{
   static const char vs_source[] =
      "#version 400 core\n"
      "in vec4 vertex;\n"
      "void main(void)\n"
      "{\n"
      "   gl_Position = vertex;\n"
      "}\n";
   static const char gs_source[] =
      "#version 400 core\n"
      "layout(points) in;\n"
      "layout(points, max_vertices = 8) out;\n"
      "layout(stream = 0) out vec4 pos0;\n"
      "layout(stream = 1) out vec4 pos1;\n"
      "void main(void)\n"
      "{\n"
      "   pos0 = vec4(0.1, 0.2, 0.3, 0.4) * gl_in[0].gl_Position;\n"
      "   gl_Position = vec4(0.9, 0.9, 0.0, 1.0) * gl_in[0].gl_Position;\n"
      "   EmitStreamVertex(0);\n"
      "   EndStreamPrimitive(0);\n"
      "   pos1 = vec4(-0.1, -0.2, -0.3, -0.4) * gl_in[0].gl_Position;\n"
      "   gl_Position = vec4(-0.9, -0.9, 0.0, 1.0) * gl_in[0].gl_Position;\n"
      "   EmitStreamVertex(1);\n"
      "   EndStreamPrimitive(1);\n"
      "}\n";
   static const char *varyings[] = {
      "pos0", "gl_SkipComponents4", "gl_NextBuffer", "gl_SkipComponents4",
      "pos1",
   };

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source);
   if (!vs || !gs)
      return 0;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, gs);
   glTransformFeedbackVaryings(program, 5, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(gs);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "xfb streams: link failed: %s\n", log);
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
      EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE,
   };
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   GLint max_streams = 0;
   glGetIntegerv(GL_MAX_VERTEX_STREAMS, &max_streams);
   if (max_streams < 2) {
      fprintf(stderr, "xfb streams: only %d vertex streams\n", max_streams);
      return 1;
   }

   GLuint program = make_program();
   if (!program)
      return 1;

   static const GLfloat vertices[3][4] = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {-4.0f, 3.0f, -2.0f, 1.0f},
      {0.5f, -0.5f, 8.0f, -8.0f},
   };
   GLuint vao, vbo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   const GLint attribute = glGetAttribLocation(program, "vertex");
   glEnableVertexAttribArray(attribute);
   glVertexAttribPointer(attribute, 4, GL_FLOAT, GL_FALSE, 0, NULL);

   /* Each buffer holds one captured vec4 and four skipped components per
    * vertex. */
   const GLsizeiptr stride = 8 * sizeof(GLfloat);
   GLuint xfb_buffers[2];
   glGenBuffers(2, xfb_buffers);
   for (unsigned i = 0; i < 2; ++i) {
      glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb_buffers[i]);
      glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3 * stride, NULL,
                   GL_DYNAMIC_READ);
      glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, xfb_buffers[i]);
   }

   glUseProgram(program);
   glEnable(GL_RASTERIZER_DISCARD);
   glBeginTransformFeedback(GL_POINTS);
   glDrawArrays(GL_POINTS, 0, 3);
   glEndTransformFeedback();
   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb streams: the capture draw raised a GL error\n");
      return 1;
   }

   /* The no-op shim submits nothing, so only a real GPU can be checked. */
   if (getenv("ASAHI_GL_NOOP_SHIM") == NULL) {
      static const GLfloat scale[2][4] = {
         {0.1f, 0.2f, 0.3f, 0.4f}, {-0.1f, -0.2f, -0.3f, -0.4f}};
      for (unsigned buffer = 0; buffer < 2; ++buffer) {
         GLfloat captured[3][8];
         glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb_buffers[buffer]);
         glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(captured),
                            captured);
         for (unsigned v = 0; v < 3; ++v) {
            /* Buffer 0 captures pos0 then skips; buffer 1 skips then pos1. */
            const GLfloat *got = captured[v] + (buffer ? 4 : 0);
            for (unsigned c = 0; c < 4; ++c) {
               const GLfloat want = scale[buffer][c] * vertices[v][c];
               if (fabsf(got[c] - want) > 1e-5f) {
                  fprintf(stderr,
                          "xfb streams: buffer %u vertex %u component %u: "
                          "%f, expected %f\n",
                          buffer, v, c, got[c], want);
                  return 1;
               }
            }
         }
      }
   }

   glDeleteBuffers(2, xfb_buffers);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
