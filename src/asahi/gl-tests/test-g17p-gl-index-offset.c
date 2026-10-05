/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Qt image strips use U16 element-buffer offsets that are not dword aligned.
 * Check every pixel, including both triangles, with and without restart and
 * after CPU uploads or GPU compute writes. Aligned U32 draws are controls.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 64
#define HEIGHT 64

static GLuint
compile_shader(GLenum type, const char *source)
{
   GLuint shader = glCreateShader(type);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "shader compile failed: %s\n", log);
      exit(2);
   }
   return shader;
}

static GLuint
link_program(GLuint first, GLuint second)
{
   GLuint program = glCreateProgram();
   glAttachShader(program, first);
   if (second)
      glAttachShader(program, second);
   glLinkProgram(program);
   glDeleteShader(first);
   if (second)
      glDeleteShader(second);
   GLint ok;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "program link failed: %s\n", log);
      exit(2);
   }
   return program;
}

int
main(void)
{
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                              EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(display, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API))
      return 2;
   const EGLint attributes[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, attributes, &config, 1, &count) || !count)
      return 2;
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE,
   };
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                         context_attributes);
   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
      return 2;
   printf("Index-offset renderer: %s\n", glGetString(GL_RENDERER));

   GLuint graphics = link_program(
      compile_shader(GL_VERTEX_SHADER,
         "#version 310 es\nlayout(location=0) in vec2 v;"
         "void main(){gl_Position=vec4(v,0,1);}"),
      compile_shader(GL_FRAGMENT_SHADER,
         "#version 310 es\nprecision highp float;out vec4 c;"
         "void main(){c=vec4(1,0,1,1);}"));
   GLuint compute = link_program(compile_shader(GL_COMPUTE_SHADER,
      "#version 310 es\nlayout(local_size_x=1) in;"
      "layout(std430,binding=0) readonly buffer S{uint src[];};"
      "layout(std430,binding=1) writeonly buffer D{uint dst[];};"
      "void main(){uint i=gl_GlobalInvocationID.x;dst[i]=src[i];}"), 0);

   GLuint vao, vertex, indices, source, framebuffer, texture;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   const float vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
   glGenBuffers(1, &vertex);
   glBindBuffer(GL_ARRAY_BUFFER, vertex);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glGenBuffers(1, &indices);
   glGenBuffers(1, &source);
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D, texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 2;
   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_DITHER);

   const GLenum types[] = {GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT};
   uint8_t pixels[WIDTH * HEIGHT * 4];
   unsigned failures = 0, cases = 0;
   for (unsigned type = 0; type < 3; ++type) {
      const unsigned size = 1u << type;
      for (unsigned gpu = 0; gpu < 2; ++gpu) {
         for (unsigned restart = 0; restart < 2; ++restart) {
            for (unsigned offset = 0; offset < 4; ++offset) {
               uint8_t data[32];
               memset(data, 0xff, sizeof(data));
               for (unsigned i = 0; i < 4; ++i) {
                  for (unsigned byte = 0; byte < size; ++byte)
                     data[(offset + i) * size + byte] = (uint8_t)(i >> (8 * byte));
               }
               glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
               glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(data),
                             gpu ? NULL : data, GL_DYNAMIC_DRAW);
               if (gpu) {
                  glBindBuffer(GL_SHADER_STORAGE_BUFFER, source);
                  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data,
                                GL_DYNAMIC_DRAW);
                  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, source);
                  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indices);
                  glUseProgram(compute);
                  glDispatchCompute(sizeof(data) / 4, 1, 1);
                  glMemoryBarrier(GL_ELEMENT_ARRAY_BARRIER_BIT);
               }
               if (restart)
                  glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
               else
                  glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
               glUseProgram(graphics);
               glClearColor(0, 0, 0, 1);
               glClear(GL_COLOR_BUFFER_BIT);
               glDrawElements(GL_TRIANGLE_STRIP, 4, types[type],
                               (void *)(uintptr_t)(offset * size));
               glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
               unsigned correct = 0;
               for (unsigned i = 0; i < WIDTH * HEIGHT; ++i)
                  correct += pixels[i * 4] == 255 && pixels[i * 4 + 1] == 0 &&
                             pixels[i * 4 + 2] == 255 && pixels[i * 4 + 3] == 255;
               GLenum error = glGetError();
               bool ok = correct == WIDTH * HEIGHT && error == GL_NO_ERROR;
               if (!ok)
                  fprintf(stderr, "FAIL U%u offset=%u writer=%s restart=%u: "
                          "%u/%u pixels, GL error %#x\n", size * 8, offset,
                          gpu ? "GPU" : "CPU", restart, correct,
                          WIDTH * HEIGHT, error);
               failures += !ok;
               ++cases;
            }
         }
      }
   }
   printf("Index-offset pixel oracle: %u/%u cases passed\n",
          cases - failures, cases);
   glDeleteFramebuffers(1, &framebuffer);
   glDeleteTextures(1, &texture);
   glDeleteBuffers(1, &source);
   glDeleteBuffers(1, &indices);
   glDeleteBuffers(1, &vertex);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(compute);
   glDeleteProgram(graphics);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglTerminate(display);
   return failures ? 1 : 0;
}
