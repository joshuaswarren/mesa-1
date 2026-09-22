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

/* A shim control run must not silently answer from a different chip. */
static bool
renderer_is_expected(void)
{
   const char *expect = getenv("ASAHI_GL_EXPECT_RENDERER");
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   if (expect && (!renderer || !strstr(renderer, expect))) {
      fprintf(stderr,
              "xfb overflow: expected a renderer containing '%s', got '%s'\n",
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



/* The single-stream cases above reach the overflow query through a plain
 * vertex shader. This one reaches it through a geometry shader and never
 * pauses, which separates "the geometry path never reports overflow" from
 * "pause and resume lose the running offset". */
/* `dynamic` makes the geometry shader's vertex count depend on a uniform.
 * A statically known count lets poly_previous_xfb_primitives() compute the
 * base arithmetically; a dynamic one sends it through the count buffer and
 * the libagx_prefix_sum_geom kernel, which is a different machine entirely
 * (threadgroup memory, a barrier and a subgroup scan). The CTS's
 * multiple-streams case emits conditionally, so it takes the second path --
 * this row isolates that from the stream count. */
static bool
geometry_overflow(bool dynamic)
{
   static const char vs_source[] =
      "#version 150 core\n"
      "void main() {\n"
      "}";
   static const char gs_static[] =
      "#version 150 core\n"
      "layout(points) in;\n"
      "layout(points, max_vertices = 1) out;\n"
      "out float output1;\n"
      "void main() {\n"
      "    output1 = 1.0;\n"
      "    EmitVertex();\n"
      "    EndPrimitive();\n"
      "}";
   static const char gs_dynamic[] =
      "#version 150 core\n"
      "layout(points) in;\n"
      "layout(points, max_vertices = 1) out;\n"
      "out float output1;\n"
      "uniform uint emit;\n"
      "void main() {\n"
      "    if (emit != 0u) {\n"
      "        output1 = 1.0;\n"
      "        EmitVertex();\n"
      "        EndPrimitive();\n"
      "    }\n"
      "}";
   const char *gs_source = dynamic ? gs_dynamic : gs_static;
   static const char *varyings[] = {"output1"};

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source);
   if (!vs || !gs)
      return false;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, gs);
   glTransformFeedbackVaryings(program, 1, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(gs);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "xfb overflow: geometry link failed: %s\n", log);
      return false;
   }

   GLuint vao, xfb, overflow_query, written_query;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &xfb);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 10 * sizeof(GLfloat), NULL,
                GL_DYNAMIC_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);
   glGenQueries(1, &overflow_query);
   glGenQueries(1, &written_query);

   glUseProgram(program);
   {
      const GLint emit = glGetUniformLocation(program, "emit");
      if (emit >= 0)
         glUniform1ui((GLuint)emit, 1);
   }
   glEnable(GL_RASTERIZER_DISCARD);

   bool ok = true;
   /* Ten points fit exactly; eleven do not. Both run from a fresh capture, so
    * neither depends on a running offset surviving anything. */
   static const struct { GLsizei points; GLuint overflow, written; } cases[] = {
      {10, 0, 10}, {11, 1, 10},
   };
   for (unsigned i = 0; i < 2; ++i) {
      glBeginTransformFeedback(GL_POINTS);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW, overflow_query);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, written_query);
      glDrawArrays(GL_POINTS, 0, cases[i].points);
      glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
      glEndQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW);
      glEndTransformFeedback();

      GLuint over = 0, written = 0;
      glGetQueryObjectuiv(overflow_query, GL_QUERY_RESULT, &over);
      glGetQueryObjectuiv(written_query, GL_QUERY_RESULT, &written);
      const bool step_ok =
         over == cases[i].overflow && written == cases[i].written;
      ok &= step_ok;
      printf("  geometry%s %2d points %s overflow %u/%u  written %2u/%2u  %s\n",
             dynamic ? " dyn" : "    ", cases[i].points,
             dynamic ? "  " : "      ", over, cases[i].overflow, written,
             cases[i].written, step_ok ? "ok" : "WRONG");
   }

   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb overflow: geometry run raised a GL error\n");
      ok = false;
   }

   glDeleteQueries(1, &written_query);
   glDeleteQueries(1, &overflow_query);
   glDeleteBuffers(1, &xfb);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   return ok;
}

/* KHR-GL33.transform_feedback_overflow_query_ARB.multiple-streams-one-buffer-
 * per-stream, step for step. Four vertex streams each capture one float into
 * their own ten-float buffer, and the overflow queries are read around
 * PauseTransformFeedback/ResumeTransformFeedback. Every step is reported, so a
 * divergence names which draw the driver got wrong rather than only that the
 * case failed. */
struct stream_step {
   const char *what;
   GLsizei count[4];
   GLboolean any;
   GLboolean stream[4];
};

static GLuint ms_program;
static GLuint ms_overflow_query;
static GLuint ms_stream_query[4];
static GLint ms_stream_uniform;
static unsigned ms_streams;

static void
ms_begin_queries(void)
{
   glBeginQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW, ms_overflow_query);
   for (unsigned i = 0; i < ms_streams; ++i)
      glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW, i,
                          ms_stream_query[i]);
}

static void
ms_end_queries(void)
{
   glEndQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW);
   for (unsigned i = 0; i < ms_streams; ++i)
      glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW, i);
}

static void
ms_draw(const GLsizei count[4])
{
   ms_begin_queries();
   for (unsigned stream = 0; stream < 4; ++stream) {
      glUniform1ui((GLuint)ms_stream_uniform, stream);
      glDrawArrays(GL_POINTS, 0, count[stream]);
   }
   ms_end_queries();
}

static GLuint
ms_result(GLuint query)
{
   GLuint value = 0;
   glGetQueryObjectuiv(query, GL_QUERY_RESULT, &value);
   return value;
}


static bool ms_build_program(void);

/* Two cuts through the CTS's multiple-streams case, each dropping one of its
 * two ingredients, so a failure names which one is responsible.
 *
 *  - pause_resume(): one stream, one buffer, but paused and resumed exactly
 *    as the CTS does it.
 *  - streams_without_pause(): four streams into four buffers with the same
 *    non-identity buffer mapping, but capture never pauses.
 */
static bool
pause_resume(void)
{
   static const char vs_source[] =
      "#version 150 core\n"
      "void main() {\n"
      "}";
   static const char gs_source[] =
      "#version 150 core\n"
      "layout(points) in;\n"
      "layout(points, max_vertices = 1) out;\n"
      "out float output1;\n"
      "void main() {\n"
      "    output1 = 1.0;\n"
      "    EmitVertex();\n"
      "    EndPrimitive();\n"
      "}";
   static const char *varyings[] = {"output1"};

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source);
   if (!vs || !gs)
      return false;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, gs);
   glTransformFeedbackVaryings(program, 1, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(gs);

   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked)
      return false;

   GLuint vao, xfb, query;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &xfb);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 10 * sizeof(GLfloat), NULL,
                GL_DYNAMIC_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);
   glGenQueries(1, &query);

   glUseProgram(program);
   glEnable(GL_RASTERIZER_DISCARD);

   /* Eight fit, the paused eleven must not count at all, then three more
    * cannot fit beside the eight already captured. */
   static const struct { const char *what; GLsizei points; GLuint want; }
      steps[] = {
         {"8 points",            8,  0},
         {"11 points, paused",   11, 0},
         {"3 points, resumed",   3,  1},
      };

   bool ok = true;
   glBeginTransformFeedback(GL_POINTS);
   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2)
         glResumeTransformFeedback();

      glBeginQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW, query);
      glDrawArrays(GL_POINTS, 0, steps[i].points);
      glEndQuery(GL_TRANSFORM_FEEDBACK_OVERFLOW);

      if (i == 0)
         glPauseTransformFeedback();
      else if (i == 2)
         glEndTransformFeedback();

      GLuint got = 0;
      glGetQueryObjectuiv(query, GL_QUERY_RESULT, &got);
      const bool step_ok = got == steps[i].want;
      ok &= step_ok;
      printf("  pause/resume %-18s overflow %u/%u  %s\n", steps[i].what, got,
             steps[i].want, step_ok ? "ok" : "WRONG");
   }

   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb overflow: pause/resume run raised a GL error\n");
      ok = false;
   }

   glDeleteQueries(1, &query);
   glDeleteBuffers(1, &xfb);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   return ok;
}

static bool
streams_without_pause(void)
{
   GLint max_streams = 0;
   glGetIntegerv(GL_MAX_VERTEX_STREAMS, &max_streams);
   if (max_streams < 4)
      return false;
   ms_streams = 4;

   if (!ms_build_program())
      return false;

   GLuint vao, xfb[4];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(4, xfb);
   for (unsigned i = 0; i < 4; ++i) {
      glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[i]);
      glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 10 * sizeof(GLfloat), NULL,
                   GL_DYNAMIC_READ);
      glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, xfb[i]);
   }
   glGenQueries(1, &ms_overflow_query);
   glGenQueries(4, ms_stream_query);
   GLuint written_query[4];
   glGenQueries(4, written_query);

   glUseProgram(ms_program);
   ms_stream_uniform = glGetUniformLocation(ms_program, "stream");
   glEnable(GL_RASTERIZER_DISCARD);

   static const struct stream_step steps[] = {
      {"8 to every stream",    {8, 8, 8, 8}, GL_FALSE,
       {GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE}},
      {"3 to streams 1 and 3", {0, 3, 0, 3}, GL_TRUE,
       {GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE}},
   };
   /* Each buffer holds ten points. The second group asks streams 1 and 3 for
    * three more on top of eight, so only two of the three may be recorded --
    * which is the same clamp the overflow flag is derived from. */
   static const GLuint want_written[2][4] = {{8, 8, 8, 8}, {0, 2, 0, 2}};

   bool ok = true;
   glBeginTransformFeedback(GL_POINTS);
   for (unsigned i = 0; i < 2; ++i) {
      for (unsigned s = 0; s < 4; ++s)
         glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, s,
                             written_query[s]);
      ms_draw(steps[i].count);
      for (unsigned s = 0; s < 4; ++s)
         glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, s);

      GLuint written[4];
      for (unsigned s = 0; s < 4; ++s)
         glGetQueryObjectuiv(written_query[s], GL_QUERY_RESULT, &written[s]);
      printf("  no-pause %-24s written %u%u%u%u/%u%u%u%u\n", steps[i].what,
             written[0], written[1], written[2], written[3],
             want_written[i][0], want_written[i][1], want_written[i][2],
             want_written[i][3]);

      const GLuint any = ms_result(ms_overflow_query);
      GLuint per[4];
      for (unsigned s = 0; s < 4; ++s)
         per[s] = ms_result(ms_stream_query[s]);

      const bool step_ok = any == (GLuint)steps[i].any &&
         per[0] == (GLuint)steps[i].stream[0] &&
         per[1] == (GLuint)steps[i].stream[1] &&
         per[2] == (GLuint)steps[i].stream[2] &&
         per[3] == (GLuint)steps[i].stream[3];
      ok &= step_ok;
      printf("  no-pause %-24s any %u/%u  streams %u%u%u%u/%u%u%u%u  %s\n",
             steps[i].what, any, steps[i].any, per[0], per[1], per[2], per[3],
             steps[i].stream[0], steps[i].stream[1], steps[i].stream[2],
             steps[i].stream[3], step_ok ? "ok" : "WRONG");
   }
   glEndTransformFeedback();
   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb overflow: no-pause run raised a GL error\n");
      ok = false;
   }

   glDeleteQueries(4, written_query);
   glDeleteQueries(4, ms_stream_query);
   glDeleteQueries(1, &ms_overflow_query);
   glDeleteBuffers(4, xfb);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(ms_program);
   return ok;
}

static bool
ms_build_program(void)
{
   static const char vs_source[] =
      "#version 150 core\n"
      "void main() {\n"
      "}";
   static const char gs_source[] =
      "#version 150 core\n"
      "#extension GL_ARB_gpu_shader5 : require\n"
      "layout(points) in;\n"
      "layout(points, max_vertices = 1) out;\n"
      "layout(stream=0) out float output1;\n"
      "layout(stream=1) out float output2;\n"
      "layout(stream=2) out float output3;\n"
      "layout(stream=3) out float output4;\n"
      "uniform uint stream;\n"
      "void main() {\n"
      "    if (stream == 0) {\n"
      "        output1 = 1.0;\n"
      "        EmitStreamVertex(0);\n"
      "        EndStreamPrimitive(0);\n"
      "    }\n"
      "    if (stream == 1) {\n"
      "        output2 = 2.0;\n"
      "        EmitStreamVertex(1);\n"
      "        EndStreamPrimitive(1);\n"
      "    }\n"
      "    if (stream == 2) {\n"
      "        output3 = 3.0;\n"
      "        EmitStreamVertex(2);\n"
      "        EndStreamPrimitive(2);\n"
      "    }\n"
      "    if (stream == 3) {\n"
      "        output4 = 4.0;\n"
      "        EmitStreamVertex(3);\n"
      "        EndStreamPrimitive(3);\n"
      "    }\n"
      "}";
   static const char *varyings[] = {"output4", "gl_NextBuffer", "output3",
                                    "gl_NextBuffer", "output2",
                                    "gl_NextBuffer", "output1"};

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source);
   if (!vs || !gs)
      return false;

   ms_program = glCreateProgram();
   glAttachShader(ms_program, vs);
   glAttachShader(ms_program, gs);
   glTransformFeedbackVaryings(ms_program, 7, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(ms_program);
   glDeleteShader(vs);
   glDeleteShader(gs);

   GLint linked = GL_FALSE;
   glGetProgramiv(ms_program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[2048] = {0};
      glGetProgramInfoLog(ms_program, sizeof(log), NULL, log);
      fprintf(stderr, "xfb overflow: multi-stream link failed: %s\n", log);
      return false;
   }
   return true;
}

static bool
multiple_streams(void)
{
   GLint max_streams = 0;
   glGetIntegerv(GL_MAX_VERTEX_STREAMS, &max_streams);
   if (max_streams < 4) {
      fprintf(stderr, "xfb overflow: only %d vertex streams\n", max_streams);
      return false;
   }
   ms_streams = 4;

   if (!ms_build_program())
      return false;

   GLuint vao, xfb[4];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(4, xfb);
   for (unsigned i = 0; i < 4; ++i) {
      glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[i]);
      glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 10 * sizeof(GLfloat), NULL,
                   GL_DYNAMIC_READ);
      glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, xfb[i]);
   }

   glGenQueries(1, &ms_overflow_query);
   glGenQueries(4, ms_stream_query);

   glUseProgram(ms_program);
   ms_stream_uniform = glGetUniformLocation(ms_program, "stream");
   glEnable(GL_RASTERIZER_DISCARD);

   /* Each buffer holds ten points. The steps below run the CTS's exact
    * sequence: fill eight, overflow while paused (which must not count),
    * overflow streams 1 and 3, exactly fill streams 0 and 2, then overflow
    * streams 2 and 3. */
   static const struct stream_step steps[] = {
      {"8 to every stream",        {8, 8, 8, 8},     GL_FALSE,
       {GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE}},
      {"11 to every stream, paused", {11, 11, 11, 11}, GL_FALSE,
       {GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE}},
      {"3 to streams 1 and 3",     {0, 3, 0, 3},     GL_TRUE,
       {GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE}},
      {"2 to streams 0 and 2",     {2, 0, 2, 0},     GL_FALSE,
       {GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE}},
      {"1 to streams 2 and 3",     {0, 0, 1, 1},     GL_TRUE,
       {GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE}},
   };

   bool ok = true;
   glBeginTransformFeedback(GL_POINTS);
   for (unsigned i = 0; i < 5; ++i) {
      /* Capture is paused from the end of the first step until the third, so
       * the overflowing second step must leave every query FALSE. */
      if (i >= 2)
         glResumeTransformFeedback();

      ms_draw(steps[i].count);

      if (i == 4)
         glEndTransformFeedback();
      else if (i != 1)
         glPauseTransformFeedback();

      const GLuint any = ms_result(ms_overflow_query);
      GLuint per[4];
      for (unsigned s = 0; s < 4; ++s)
         per[s] = ms_result(ms_stream_query[s]);

      const bool step_ok = any == (GLuint)steps[i].any &&
         per[0] == (GLuint)steps[i].stream[0] &&
         per[1] == (GLuint)steps[i].stream[1] &&
         per[2] == (GLuint)steps[i].stream[2] &&
         per[3] == (GLuint)steps[i].stream[3];
      ok &= step_ok;

      printf("  %-28s any %u/%u  streams %u%u%u%u/%u%u%u%u  %s\n",
             steps[i].what, any, steps[i].any, per[0], per[1], per[2], per[3],
             steps[i].stream[0], steps[i].stream[1], steps[i].stream[2],
             steps[i].stream[3], step_ok ? "ok" : "WRONG");
   }
   glDisable(GL_RASTERIZER_DISCARD);
   glFinish();

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "xfb overflow: multi-stream run raised a GL error\n");
      ok = false;
   }

   glDeleteQueries(4, ms_stream_query);
   glDeleteQueries(1, &ms_overflow_query);
   glDeleteBuffers(4, xfb);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(ms_program);
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

   if (!renderer_is_expected())
      return 1;

   bool ok = true;
   ok &= single_stream(GL_INTERLEAVED_ATTRIBS, "interleaved");
   ok &= single_stream(GL_SEPARATE_ATTRIBS, "separate");

   /* The query results only mean anything where something actually ran. */
   if (getenv("ASAHI_GL_NOOP_SHIM") == NULL) {
      ok &= geometry_overflow(false);
      ok &= geometry_overflow(true);
      ok &= pause_resume();
      ok &= streams_without_pause();
      ok &= multiple_streams();
   }
   else
      printf("xfb overflow: no-op shim, skipping the query results\n");

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   if (ok)
      printf("xfb overflow: the positionless four-output capture works\n");
   return ok ? 0 : 1;
}
