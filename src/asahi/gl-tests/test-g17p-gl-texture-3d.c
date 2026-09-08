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

#define WIDTH  8
#define HEIGHT 8

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
      fprintf(stderr, "texture3d shader compile failed: %s\n", log);
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
      "const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), "
      "vec2(-1.0, 3.0));\n"
      "void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); }\n";
   /* This is the texture operation in Hyprland's applyIcc3DLut(), including
    * its texel-centre remap. It must remain implicit LOD: replacing texture()
    * with textureLod() would miss the production compiler path. */
   static const char fs_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "precision highp sampler3D;\n"
      "uniform sampler3D iccLut3D;\n"
      "uniform float iccLutSize;\n"
      "layout(location=0) out vec4 colour;\n"
      "vec3 applyIcc3DLut(vec3 linearRgb01) {\n"
      "  vec3 x = clamp(linearRgb01, 0.0, 1.0);\n"
      "  vec3 coord = (x * (iccLutSize - 1.0) + 0.5) / iccLutSize;\n"
      "  return texture(iccLut3D, coord).rgb;\n"
      "}\n"
      "void main() { colour = vec4(applyIcc3DLut(vec3(0.5, 0.5, "
      "0.6666667)), 1.0); }\n";

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
      fprintf(stderr, "texture3d program link failed: %s\n", log);
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

   GLuint program = make_program();
   if (!program)
      return 1;

   static const uint8_t slices[4][4] = {
      {1, 2, 3, 255}, {4, 5, 6, 255},
      {17, 123, 201, 255}, {7, 8, 9, 255},
   };
   uint8_t volume[4 * 4 * 4 * 4];
   for (unsigned z = 0; z < 4; ++z) {
      for (unsigned i = 0; i < 16; ++i) {
         for (unsigned c = 0; c < 4; ++c)
            volume[((z * 16 + i) * 4) + c] = slices[z][c];
      }
   }

   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_3D, texture);
   glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, 4, 4, 4);
   glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 4, 4, 4, GL_RGBA,
                   GL_UNSIGNED_BYTE, volume);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "iccLut3D"), 0);
   glUniform1f(glGetUniformLocation(program, "iccLutSize"), 4.0f);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glFinish();

   uint8_t pixel[4] = {0};
   glReadPixels(WIDTH / 2, HEIGHT / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "texture3d draw/readback error %#x\n", error);
      return 1;
   }
   if (!noop && (pixel[0] != 17 || pixel[1] != 123 || pixel[2] != 201 ||
                 pixel[3] != 255)) {
      fprintf(stderr, "texture3d ICC pixel {%u,%u,%u,%u}\n", pixel[0],
              pixel[1], pixel[2], pixel[3]);
      return 1;
   }

   glDeleteVertexArrays(1, &vao);
   glDeleteTextures(1, &texture);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts(noop ? "G17P Hyprland ICC implicit texture3d sample submitted"
             : "G17P Hyprland ICC implicit texture3d pixel exact");
   return 0;
}
