/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH  8
#define HEIGHT 8

/*
 * Which driver actually ran.
 *
 * A G17P GL run can reach EGL 1.5 / GL 4.6 / GLES 3.2 and report an Apple G17P
 * renderer through a Gallium that has none of this project's G17P work in it --
 * the system Mesa's own asahi driver does exactly that.  Such a driver never
 * binds the fixed render context at 0x1000000000, so the kernel refuses its
 * first render job ("job context view 0 ... is not bound") and the failure
 * reads as a driver bug in code that was never loaded.
 *
 * Two independent discriminators are printed, and either can be made fatal:
 * the mapped libgallium object, and the renderer string (this driver names the
 * exact product, "Apple A18 Pro (G17P ..)", where a driver without the G17P
 * product table falls back to "Apple M5 Unknown (G17P ..)").
 */
static bool
check_driver_provenance(void)
{
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *version = (const char *)glGetString(GL_VERSION);

   printf("vertex-texture GL_RENDERER: %s\n", renderer ? renderer : "(null)");
   printf("vertex-texture GL_VERSION: %s\n", version ? version : "(null)");

   const char *expect_driver = getenv("ASAHI_GL_EXPECT_DRIVER");
   char previous[4096] = {0};
   unsigned mapped = 0;
   bool foreign = false;

   FILE *maps = fopen("/proc/self/maps", "r");
   char line[4096];
   while (maps && fgets(line, sizeof(line), maps)) {
      char *path = strchr(line, '/');
      if (!path || !strstr(path, "libgallium"))
         continue;

      char *newline = strchr(path, '\n');
      if (newline)
         *newline = '\0';
      if (!strcmp(path, previous))
         continue;

      snprintf(previous, sizeof(previous), "%s", path);
      printf("vertex-texture gallium object: %s\n", path);
      mapped++;

      if (expect_driver && !strstr(path, expect_driver))
         foreign = true;
   }
   if (maps)
      fclose(maps);

   if (expect_driver && (!mapped || foreign)) {
      fprintf(stderr,
              "vertex-texture: expected the Gallium driver at '%s'; %u "
              "libgallium object(s) mapped, foreign=%d\n",
              expect_driver, mapped, foreign);
      return false;
   }

   const char *expect_renderer = getenv("ASAHI_GL_EXPECT_RENDERER");
   if (expect_renderer &&
       (!renderer || !strstr(renderer, expect_renderer))) {
      fprintf(stderr,
              "vertex-texture: expected a renderer containing '%s', got '%s'\n",
              expect_renderer, renderer ? renderer : "(null)");
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
      fprintf(stderr, "vertex-texture shader compile failed: %s\n", log);
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
      "layout(location=0) in vec2 position;\n"
      "uniform highp sampler2D source;\n"
      "flat out highp vec4 sampled;\n"
      "void main() {\n"
      "  gl_Position = vec4(position, 0.0, 1.0);\n"
      "  sampled = texture(source, vec2(0.5));\n"
      "}\n";
   static const char fs_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "flat in highp vec4 sampled;\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = sampled; }\n";

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
      fprintf(stderr, "vertex-texture program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
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
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
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

   if (!check_driver_provenance())
      return 1;

   GLuint program = make_program();
   if (!program)
      return 1;

   static const uint8_t texel[4] = {17, 123, 201, 255};
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA,
                   GL_UNSIGNED_BYTE, texel);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   static const float positions[] = {
      -1.0f, -1.0f,
       3.0f, -1.0f,
      -1.0f,  3.0f,
   };
   GLuint vao, vbo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "source"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "vertex-texture draw/readback error %#x\n", error);
      return 1;
   }
   if (!noop && memcmp(pixel, texel, sizeof(texel)) != 0) {
      fprintf(stderr, "vertex-texture pixel {%u,%u,%u,%u}\n", pixel[0],
              pixel[1], pixel[2], pixel[3]);
      return 1;
   }

   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(1, &texture);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts(noop ? "G17P native vertex texture draw submitted"
             : "G17P native vertex texture pixel exact");
   return 0;
}
