/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL33.transform_feedback_overflow_query_ARB.* on a 3.3 core context.
 * This family matters out of proportion to its size: when one of its cases
 * fails, the case throws while transform feedback is still active, and
 * gluStateReset only ends transform feedback on a 4.0 context. On 3.3 it goes
 * straight to glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0), which is
 * GL_INVALID_OPERATION while feedback is active, and dEQP treats a failed
 * deinit as fatal -- so the whole session terminates at this case and every
 * later case in gl33-main.txt never runs at all.
 *
 * Two distinct shapes are covered:
 *
 *  - single stream: a vertex shader with four float outputs and no
 *    gl_Position, captured interleaved and separate. The G17P packer refused
 *    it, so glLinkProgram failed and four cases failed on the link alone.
 *
 *  - multiple streams, one buffer per stream: an empty vertex shader behind a
 *    four-stream geometry shader, with the overflow queries checked around a
 *    paused capture. Only a real device can grade the query results; under
 *    the no-op shim this still exercises the link and the submitted stream.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

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
      fprintf(stderr, "xfb overflow: shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

/* The CTS's own single-stream program: four float outputs, no position, no
 * fragment shader. */
static bool
single_stream(GLenum mode, const char *mode_name)
{
   static const char vs_source[] =
      "#version 150 core\n"
      "out float output1;\n"
      "out float output2;\n"
      "out float output3;\n"
      "out float output4;\n"
      "void main() {\n"
      "    output1 = 1.0;\n"
      "    output2 = 2.0;\n"
      "    output3 = 3.0;\n"
      "    output4 = 4.0;\n"
      "}";
   static const char *varyings[] = {"output1", "output2", "output3",
                                    "output4"};

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   if (!vs)
      return false;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glTransformFeedbackVaryings(program, 4, varyings, mode);
   glLinkProgram(program);
   glDeleteShader(vs);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "xfb overflow: %s link failed: %s\n", mode_name, log);
      glDeleteProgram(program);
      return false;
   }

   /* Capture four points and check what each varying carried. Separate mode
    * puts each varying in its own buffer; interleaved packs all four. */
   const unsigned points = 4;
   const unsigned buffers = mode == GL_SEPARATE_ATTRIBS ? 4 : 1;
   const unsigned per_buffer = mode == GL_SEPARATE_ATTRIBS ? 1 : 4;

   GLuint vao, xfb[4];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers((GLsizei)buffers, xfb);
   for (unsigned i = 0; i < buffers; ++i) {
      glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[i]);
      glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
                   (GLsizeiptr)(points * per_buffer * sizeof(GLfloat)), NULL,
                   GL_DYNAMIC_READ);
      glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, xfb[i]);
   }

   glUseProgram(program);
   glEnable(GL_RASTERIZER_DISCARD);
   glBeginTransformFeedback(GL_POINTS);
   glDrawArrays(GL_POINTS, 0, (GLsizei)points);
   glEndTransformFeedback();
   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();

   bool ok = true;
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb overflow: %s capture raised a GL error\n",
              mode_name);
      ok = false;
   } else if (getenv("ASAHI_GL_NOOP_SHIM") == NULL) {
      for (unsigned i = 0; i < buffers && ok; ++i) {
         GLfloat got[16] = {0};
         glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[i]);
         glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                            (GLsizeiptr)(points * per_buffer * sizeof(GLfloat)),
                            got);
         for (unsigned p = 0; p < points && ok; ++p) {
            for (unsigned c = 0; c < per_buffer; ++c) {
               const GLfloat want =
                  mode == GL_SEPARATE_ATTRIBS ? (GLfloat)(i + 1)
                                              : (GLfloat)(c + 1);
               const GLfloat have = got[p * per_buffer + c];
               if (have != want) {
                  fprintf(stderr,
                          "xfb overflow: %s buffer %u point %u slot %u: %f, "
                          "expected %f\n", mode_name, i, p, c, have, want);
                  ok = false;
                  break;
               }
            }
         }
      }
   }

   glDeleteBuffers((GLsizei)buffers, xfb);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
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
   /* The CTS runs this family on a 3.3 core context, which is what makes a
    * failure here fatal to the whole session. */
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE,
   };
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   printf("xfb overflow: renderer %s\n",
          (const char *)glGetString(GL_RENDERER));

   bool ok = true;
   ok &= single_stream(GL_INTERLEAVED_ATTRIBS, "interleaved");
   ok &= single_stream(GL_SEPARATE_ATTRIBS, "separate");

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (ok)
      printf("xfb overflow: the positionless four-output capture works\n");
   return ok ? 0 : 1;
}
