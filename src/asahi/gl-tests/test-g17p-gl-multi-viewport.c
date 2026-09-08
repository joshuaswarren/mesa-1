/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdbool.h>
#include <stdio.h>

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
      fprintf(stderr, "multi-viewport shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

static GLuint
make_program(void)
{
   static const char vs_source[] =
      "#version 450 core\n"
      "#extension GL_ARB_shader_viewport_layer_array : require\n"
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(1.0, -1.0), "
      "vec2(0.0, 1.0));\n"
      "void main() {\n"
      "  gl_Position = vec4(p[gl_VertexID], 0.0, 1.0);\n"
      "  gl_ViewportIndex = 1;\n"
      "}\n";
   static const char fs_source[] =
      "#version 450 core\n"
      "layout(location=0) out vec4 colour;\n"
      "void main() { colour = vec4(0.25, 0.5, 0.75, 1.0); }\n";

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
      fprintf(stderr, "multi-viewport program link failed: %s\n", log);
      glDeleteProgram(program);
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
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   GLint max_viewports = 0;
   glGetIntegerv(GL_MAX_VIEWPORTS, &max_viewports);
   GLuint program = make_program();
   if (max_viewports != 16 || !program)
      return 1;

   GLfloat viewports[16][4];
   for (unsigned i = 0; i < 16; ++i) {
      viewports[i][0] = 0.0f;
      viewports[i][1] = 0.0f;
      viewports[i][2] = 64.0f;
      viewports[i][3] = 64.0f;
   }
   viewports[1][0] = 13.0f;
   viewports[1][1] = 17.0f;
   viewports[1][2] = 20.0f;
   viewports[1][3] = 18.0f;

   /* The API bound is part of the contract: first + count may not exceed
    * GL_MAX_VIEWPORTS, and the rejected call must not poison the legal draw. */
   glViewportArrayv(1, 16, &viewports[0][0]);
   if (glGetError() != GL_INVALID_VALUE)
      return 1;
   glViewportArrayv(0, 16, &viewports[0][0]);
   if (glGetError() != GL_NO_ERROR)
      return 1;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glUseProgram(program);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();
   if (glGetError() != GL_NO_ERROR)
      return 1;

   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts("G17P GL multi-viewport index 1 draw completed");
   return 0;
}
