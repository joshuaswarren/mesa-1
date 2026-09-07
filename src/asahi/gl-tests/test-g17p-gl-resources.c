/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 64
#define HEIGHT 64
#define GUARD_SIZE 32
#define GUARD_BYTE 0xa5

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
      "attribute vec2 a_pos;\n"
      "attribute vec2 a_uv;\n"
      "uniform vec2 u_offset;\n"
      "varying vec2 v_uv;\n"
      "void main() {\n"
      "  gl_Position = vec4(a_pos + u_offset, 0.0, 1.0);\n"
      "  v_uv = a_uv;\n"
      "}\n";
   static const char fs_source[] =
      "precision highp float;\n"
      "uniform sampler2D u_tex;\n"
      "uniform vec4 u_scale;\n"
      "varying vec2 v_uv;\n"
      "void main() { gl_FragColor = texture2D(u_tex, v_uv) * u_scale; }\n";

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 0;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "a_pos");
   glBindAttribLocation(program, 1, "a_uv");
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

struct guarded_vertices {
   uint8_t before[GUARD_SIZE];
   float data[16];
   uint8_t after[GUARD_SIZE];
};

struct guarded_indices {
   uint8_t before[GUARD_SIZE];
   uint16_t data[6];
   uint8_t after[GUARD_SIZE];
};

struct guarded_texture {
   uint8_t before[GUARD_SIZE];
   uint8_t data[16];
   uint8_t after[GUARD_SIZE];
};

struct guarded_readback {
   uint8_t before[GUARD_SIZE];
   uint8_t data[WIDTH * HEIGHT * 4];
   uint8_t after[GUARD_SIZE];
};

static bool
pixel_ok(const struct guarded_readback *readback, unsigned x, unsigned y,
         const uint8_t expected[4])
{
   const uint8_t *got = &readback->data[(y * WIDTH + x) * 4];
   if (memcmp(got, expected, 4) == 0)
      return true;

   fprintf(stderr,
           "pixel (%u,%u): got {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
           x, y, got[0], got[1], got[2], got[3], expected[0], expected[1],
           expected[2], expected[3]);
   return false;
}

static bool
draw_and_read(GLuint program, GLuint vbo, GLuint ibo,
              struct guarded_readback *readback, float red_scale)
{
   memset(readback, GUARD_BYTE, sizeof(*readback));

   glViewport(0, 0, WIDTH, HEIGHT);
   glClearColor(0.0f, 0.25f, 0.5f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(program);
   glUniform2f(glGetUniformLocation(program, "u_offset"), 0.0f, 0.0f);
   glUniform4f(glGetUniformLocation(program, "u_scale"), red_scale, 1.0f,
               1.0f, 1.0f);
   glUniform1i(glGetUniformLocation(program, "u_tex"), 0);

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                         (void *)(2 * sizeof(float)));
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)0);
   glFinish();

   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE,
                readback->data);
   glFinish();

   const uint8_t red = red_scale == 1.0f ? 255 : 102;
   const uint8_t bottom_left[4] = {red, 0, 0, 255};
   const uint8_t bottom_right[4] = {0, 255, 0, 255};
   const uint8_t top_left[4] = {0, 0, 255, 255};
   const uint8_t top_right[4] = {red, 255, 255, 255};

   return gl_ok("draw/readback") && guard_ok(readback->before) &&
          guard_ok(readback->after) &&
          pixel_ok(readback, WIDTH / 4, HEIGHT / 4, bottom_left) &&
          pixel_ok(readback, 3 * WIDTH / 4, HEIGHT / 4, bottom_right) &&
          pixel_ok(readback, WIDTH / 4, 3 * HEIGHT / 4, top_left) &&
          pixel_ok(readback, 3 * WIDTH / 4, 3 * HEIGHT / 4, top_right);
}

int
main(void)
{
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
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
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
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, pbuffer_attributes);
   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE,
   };
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

   struct guarded_vertices vertices;
   memset(&vertices, GUARD_BYTE, sizeof(vertices));
   const float vertex_data[16] = {
      -1.0f, -1.0f, 0.0f, 0.0f,
       1.0f, -1.0f, 1.0f, 0.0f,
       1.0f,  1.0f, 1.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 1.0f,
   };
   memcpy(vertices.data, vertex_data, sizeof(vertex_data));

   struct guarded_indices indices;
   memset(&indices, GUARD_BYTE, sizeof(indices));
   const uint16_t index_data[6] = {0, 1, 2, 0, 2, 3};
   memcpy(indices.data, index_data, sizeof(index_data));

   struct guarded_texture texture;
   memset(&texture, GUARD_BYTE, sizeof(texture));
   const uint8_t texture_data[16] = {
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
   };
   memcpy(texture.data, texture_data, sizeof(texture_data));

   GLuint vbo, ibo, sampled, colour, fbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices.data), vertices.data,
                GL_STATIC_DRAW);
   glGenBuffers(1, &ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices.data), indices.data,
                GL_STATIC_DRAW);

   glGenTextures(1, &sampled);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, sampled);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, texture.data);

   glGenTextures(1, &colour);
   glBindTexture(GL_TEXTURE_2D, colour);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          colour, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 1;

   glBindTexture(GL_TEXTURE_2D, sampled);
   if (!gl_ok("resource creation") || !guard_ok(vertices.before) ||
       !guard_ok(vertices.after) || !guard_ok(indices.before) ||
       !guard_ok(indices.after) || !guard_ok(texture.before) ||
       !guard_ok(texture.after))
      return 1;

   struct guarded_readback readback;
   if (!draw_and_read(program, vbo, ibo, &readback, 1.0f))
      return 1;

   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT) ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   if (!glIsProgram(program) || !glIsBuffer(vbo) || !glIsBuffer(ibo) ||
       !glIsTexture(sampled) || !glIsTexture(colour) ||
       !glIsFramebuffer(fbo))
      return 1;

   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, sampled);
   if (!draw_and_read(program, vbo, ibo, &readback, 0.4f))
      return 1;

   if (!guard_ok(vertices.before) || !guard_ok(vertices.after) ||
       !guard_ok(indices.before) || !guard_ok(indices.after) ||
       !guard_ok(texture.before) || !guard_ok(texture.after) ||
       memcmp(vertices.data, vertex_data, sizeof(vertex_data)) != 0 ||
       memcmp(indices.data, index_data, sizeof(index_data)) != 0 ||
       memcmp(texture.data, texture_data, sizeof(texture_data)) != 0) {
      fprintf(stderr, "upload source or guard changed after context reuse\n");
      return 1;
   }

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &colour);
   glDeleteTextures(1, &sampled);
   glDeleteBuffers(1, &ibo);
   glDeleteBuffers(1, &vbo);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts("GL03 resource sequence completed twice; pixels, sources and guards "
        "exact");
   return 0;
}
