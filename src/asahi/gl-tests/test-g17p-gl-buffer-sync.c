/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 32
#define HEIGHT 32
#define GUARD_SIZE 32
#define GUARD_BYTE 0xa5
#define WAIT_NS UINT64_C(5000000000)

struct guarded_positions {
   uint8_t before[GUARD_SIZE];
   float data[8];
   uint8_t after[GUARD_SIZE];
};

struct guarded_colours {
   uint8_t before[GUARD_SIZE];
   float data[16];
   uint8_t after[GUARD_SIZE];
};

struct guarded_pixels {
   uint8_t before[GUARD_SIZE];
   uint8_t data[WIDTH * HEIGHT * 4];
   uint8_t after[GUARD_SIZE];
};

static bool
gl_ok(const char *where)
{
   bool ok = true;
   GLenum error;

   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "%s: GL error 0x%04x\n", where, error);
      ok = false;
   }

   return ok;
}

static bool
gl_error_is(const char *where, GLenum expected)
{
   const GLenum error = glGetError();
   if (error == expected && glGetError() == GL_NO_ERROR)
      return true;

   fprintf(stderr, "%s: GL error 0x%04x, expected 0x%04x\n", where, error,
           expected);
   return false;
}

static bool
guard_ok(const uint8_t guard[GUARD_SIZE])
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (guard[i] != GUARD_BYTE)
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
      fprintf(stderr, "shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

static GLuint
make_program(void)
{
   static const char vs_source[] =
      "#version 300 es\n"
      "layout(location=0) in vec2 a_pos;\n"
      "layout(location=1) in vec4 a_colour;\n"
      "out vec4 v_colour;\n"
      "void main() {\n"
      "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
      "  v_colour = a_colour;\n"
      "}\n";
   static const char fs_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec4 v_colour;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = v_colour; }\n";

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
      fprintf(stderr, "program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }

   return program;
}

static bool
sync_field(GLsync sync, GLenum pname, GLint expected)
{
   GLsizei length = 0;
   GLint value = 0;
   glGetSynciv(sync, pname, 1, &length, &value);
   if (!gl_ok("glGetSynciv") || length != 1 || value != expected) {
      fprintf(stderr, "sync field 0x%x: length %d value 0x%x expected 0x%x\n",
              pname, length, value, expected);
      return false;
   }

   return true;
}

static bool
wait_fence(GLsync sync)
{
   if (!glIsSync(sync) ||
       !sync_field(sync, GL_OBJECT_TYPE, GL_SYNC_FENCE) ||
       !sync_field(sync, GL_SYNC_CONDITION, GL_SYNC_GPU_COMMANDS_COMPLETE) ||
       !sync_field(sync, GL_SYNC_FLAGS, 0))
      return false;

   if (glClientWaitSync(sync, 2, 0) != GL_WAIT_FAILED ||
       !gl_error_is("invalid client wait flags", GL_INVALID_VALUE))
      return false;

   GLenum result = glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, WAIT_NS);
   if (result != GL_ALREADY_SIGNALED && result != GL_CONDITION_SATISFIED) {
      fprintf(stderr, "glClientWaitSync returned 0x%x\n", result);
      return false;
   }

   if (!sync_field(sync, GL_SYNC_STATUS, GL_SIGNALED))
      return false;

   glWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
   return gl_ok("fence wait");
}

static bool
buffer_matches(GLuint buffer, const void *expected, size_t size,
               void *guarded, uint8_t *before, uint8_t *data, uint8_t *after)
{
   memset(guarded, GUARD_BYTE, GUARD_SIZE + size + GUARD_SIZE);
   glBindBuffer(GL_ARRAY_BUFFER, buffer);
   const void *map = glMapBufferRange(GL_ARRAY_BUFFER, 0, size, GL_MAP_READ_BIT);
   if (!map)
      return false;

   memcpy(data, map, size);
   if (!glUnmapBuffer(GL_ARRAY_BUFFER))
      return false;

   return gl_ok("read map") && guard_ok(before) && guard_ok(after) &&
          memcmp(data, expected, size) == 0;
}

static void
draw(GLuint program, GLuint positions, GLuint colours)
{
   glViewport(0, 0, WIDTH, HEIGHT);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(program);

   glBindBuffer(GL_ARRAY_BUFFER, positions);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
   glBindBuffer(GL_ARRAY_BUFFER, colours);
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void *)0);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static bool
read_colour(struct guarded_pixels *pixels, const uint8_t expected[4],
            bool noop)
{
   memset(pixels, GUARD_BYTE, sizeof(*pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels->data);
   if (!gl_ok("read pixels") || !guard_ok(pixels->before) ||
       !guard_ok(pixels->after))
      return false;

   if (noop)
      return true;

   const uint8_t *got = &pixels->data[((HEIGHT / 2) * WIDTH + WIDTH / 2) * 4];
   if (memcmp(got, expected, 4) == 0)
      return true;

   fprintf(stderr, "centre pixel {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
           got[0], got[1], got[2], got[3], expected[0], expected[1],
           expected[2], expected[3]);
   return false;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "EGL surfaceless initialization failed: 0x%04x\n",
              eglGetError());
      return 1;
   }
   if (!eglBindAPI(EGL_OPENGL_ES_API))
      return 1;

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
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

   const EGLint pbuffer_attributes[] = {
      EGL_WIDTH, WIDTH,
      EGL_HEIGHT, HEIGHT,
      EGL_NONE,
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, pbuffer_attributes);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   printf("EGL %d.%d; GL %s; renderer %s\n", major, minor,
          (const char *)glGetString(GL_VERSION),
          (const char *)glGetString(GL_RENDERER));

   GLuint program = make_program();
   if (!program)
      return 1;

   struct guarded_positions positions;
   memset(&positions, GUARD_BYTE, sizeof(positions));
   const float position_data[8] = {
      -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
   };
   memcpy(positions.data, position_data, sizeof(position_data));

   struct guarded_colours red;
   struct guarded_colours blue;
   memset(&red, GUARD_BYTE, sizeof(red));
   memset(&blue, GUARD_BYTE, sizeof(blue));
   for (unsigned i = 0; i < 4; ++i) {
      red.data[4 * i + 0] = 1.0f;
      red.data[4 * i + 1] = 0.0f;
      red.data[4 * i + 2] = 0.0f;
      red.data[4 * i + 3] = 1.0f;
      blue.data[4 * i + 0] = 0.0f;
      blue.data[4 * i + 1] = 0.0f;
      blue.data[4 * i + 2] = 1.0f;
      blue.data[4 * i + 3] = 1.0f;
   }

   GLuint position_bo, colour_bo;
   glGenBuffers(1, &position_bo);
   glBindBuffer(GL_ARRAY_BUFFER, position_bo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions.data), NULL, GL_DYNAMIC_DRAW);
   void *write_map = glMapBufferRange(
      GL_ARRAY_BUFFER, 0, sizeof(positions.data),
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
   if (!write_map)
      return 1;
   memcpy(write_map, positions.data, sizeof(positions.data));
   if (!glUnmapBuffer(GL_ARRAY_BUFFER))
      return 1;

   glGenBuffers(1, &colour_bo);
   glBindBuffer(GL_ARRAY_BUFFER, colour_bo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(red.data), NULL, GL_DYNAMIC_DRAW);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(red.data), red.data);

   GLuint colour_texture, fbo;
   glGenTextures(1, &colour_texture);
   glBindTexture(GL_TEXTURE_2D, colour_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          colour_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !gl_ok("resource creation"))
      return 1;

   if (!guard_ok(positions.before) || !guard_ok(positions.after) ||
       !guard_ok(red.before) || !guard_ok(red.after) ||
       !guard_ok(blue.before) || !guard_ok(blue.after))
      return 1;

   GLsync invalid = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 1);
   (void)invalid;
   if (!gl_error_is("invalid fence flags", GL_INVALID_VALUE))
      return 1;

   draw(program, position_bo, colour_bo);
   GLsync first = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
   glFlush();
   if (!first || !gl_ok("first fence"))
      return 1;

   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;
   if (!glIsProgram(program) || !glIsBuffer(position_bo) ||
       !glIsBuffer(colour_bo) || !glIsTexture(colour_texture) ||
       !glIsFramebuffer(fbo) || !glIsSync(first) || !wait_fence(first))
      return 1;

   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   struct guarded_pixels pixels;
   const uint8_t expected_red[4] = {255, 0, 0, 255};
   if (!read_colour(&pixels, expected_red, noop))
      return 1;

   struct guarded_positions position_read;
   struct guarded_colours colour_read;
   if (!buffer_matches(position_bo, position_data, sizeof(position_data),
                       &position_read, position_read.before,
                       (uint8_t *)position_read.data, position_read.after) ||
       !buffer_matches(colour_bo, red.data, sizeof(red.data), &colour_read,
                       colour_read.before, (uint8_t *)colour_read.data,
                       colour_read.after))
      return 1;

   glDeleteSync(first);
   if (glIsSync(first) || !gl_ok("first fence delete"))
      return 1;

   glBindBuffer(GL_ARRAY_BUFFER, colour_bo);
   write_map = glMapBufferRange(
      GL_ARRAY_BUFFER, 0, sizeof(blue.data),
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
   if (!write_map)
      return 1;
   memcpy(write_map, blue.data, sizeof(blue.data));
   if (!glUnmapBuffer(GL_ARRAY_BUFFER) || !gl_ok("blue write map"))
      return 1;

   draw(program, position_bo, colour_bo);
   GLsync second = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
   glFlush();
   if (!second || !wait_fence(second))
      return 1;

   const uint8_t expected_blue[4] = {0, 0, 255, 255};
   if (!read_colour(&pixels, expected_blue, noop) ||
       !buffer_matches(colour_bo, blue.data, sizeof(blue.data), &colour_read,
                       colour_read.before, (uint8_t *)colour_read.data,
                       colour_read.after))
      return 1;

   glDeleteSync(second);
   if (glIsSync(second) || !gl_ok("second fence delete") ||
       !guard_ok(positions.before) || !guard_ok(positions.after) ||
       !guard_ok(red.before) || !guard_ok(red.after) ||
       !guard_ok(blue.before) || !guard_ok(blue.after) ||
       memcmp(positions.data, position_data, sizeof(position_data)) != 0)
      return 1;

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &colour_texture);
   glDeleteBuffers(1, &colour_bo);
   glDeleteBuffers(1, &position_bo);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL04 buffer/sync lifecycle exact; no-op pixels skipped"
             : "GL04 buffer/sync lifecycle, retained context and pixels exact");
   return 0;
}
