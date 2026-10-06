/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Chromium's glyph atlas uses R16G16_UINT vertex coordinates. AGX3 loads raw
 * packed lanes; the AGX2 prolog instead expects the load to widen each lane.
 * Check exact integer vertex values, including sign extension and defaults.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

static GLuint
shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
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

int
main(void)
{
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                              EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(display, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API))
      return 2;
   const EGLint attributes[] = {
      EGL_RENDERABLE_TYPE,
      EGL_OPENGL_ES3_BIT,
      EGL_SURFACE_TYPE,
      EGL_PBUFFER_BIT,
      EGL_RED_SIZE,
      8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, attributes, &config, 1, &count) || !count)
      return 2;
   const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                                        EGL_NONE};
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
      return 2;
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("renderer: %s\n", renderer);
   const char *expected = getenv("ASAHI_GL_EXPECT_RENDERER");
   if (expected && !strstr(renderer, expected))
      return 2;

   GLuint programs[2];
   for (unsigned sign = 0; sign < 2; ++sign) {
      char vs[1024];
      snprintf(vs, sizeof(vs),
               "#version 300 es\n"
               "precision highp float;precision highp int;"
               "layout(location=0) in %svec4 value;"
               "flat out highp uvec4 result;"
               "void main(){"
               "vec2 pos=vec2(float(gl_VertexID & 1),float(gl_VertexID >> 1));"
               "gl_Position=vec4(pos*2.0-1.0,0,1);result=uvec4(value);}",
               sign ? "i" : "u");
      GLuint fs =
         shader(GL_FRAGMENT_SHADER, "#version 300 es\nprecision highp int;"
                                    "flat in highp uvec4 result;"
                                    "layout(location=0) out highp uvec4 color;"
                                    "void main(){color=result;}");
      GLuint vertex = shader(GL_VERTEX_SHADER, vs);
      programs[sign] = glCreateProgram();
      glAttachShader(programs[sign], vertex);
      glAttachShader(programs[sign], fs);
      glLinkProgram(programs[sign]);
      GLint ok;
      glGetProgramiv(programs[sign], GL_LINK_STATUS, &ok);
      if (!ok)
         return 2;
      glDeleteShader(vertex);
      glDeleteShader(fs);
   }

   GLuint vao, vbo, texture, fbo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glEnableVertexAttribArray(0);
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, 8, 8);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 2;
   glViewport(0, 0, 8, 8);

   const GLenum types[2][3] = {
      {GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT},
      {GL_BYTE, GL_SHORT, GL_INT},
   };
   unsigned passed = 0, total = 0;
   for (unsigned sign = 0; sign < 2; ++sign) {
      glUseProgram(programs[sign]);
      for (unsigned width = 1; width <= 4; width *= 2) {
         for (unsigned channels = 1; channels <= 4; ++channels) {
            for (unsigned offset = 0; offset <= width; offset += width) {
               uint8_t data[128] = {0};
               uint32_t want[4] = {0, 0, 0, 1};
               for (unsigned c = 0; c < channels; ++c) {
                  int32_t value =
                     sign ? -(int32_t)(17 * (c + 1)) : (int32_t)(17 * (c + 1));
                  want[c] = value;
                  for (unsigned v = 0; v < 4; ++v)
                     memcpy(data + offset + (v * channels + c) * width, &value,
                            width);
               }
               glBufferData(GL_ARRAY_BUFFER, sizeof(data), data,
                            GL_STATIC_DRAW);
               glVertexAttribIPointer(0, channels, types[sign][width / 2],
                                      channels * width,
                                      (void *)(uintptr_t)offset);
               const GLuint clear[4] = {0x12345678, 0x12345678, 0x12345678,
                                        0x12345678};
               glClearBufferuiv(GL_COLOR, 0, clear);
               glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
               uint32_t pixels[8 * 8 * 4];
               glReadPixels(0, 0, 8, 8, GL_RGBA_INTEGER, GL_UNSIGNED_INT,
                            pixels);
               unsigned good = 0;
               for (unsigned p = 0; p < 64; ++p)
                  good += !memcmp(pixels + p * 4, want, sizeof(want));
               GLenum error = glGetError();
               bool ok = good == 64 && error == GL_NO_ERROR;
               ++total;
               passed += ok;
               printf("%s%u x%u offset=%u: %s (%u/64 pixels, error=%#x)\n",
                      sign ? "s" : "u", width * 8, channels, offset,
                      ok ? "PASS" : "FAIL", good, error);
            }
         }
      }
   }
   printf("%u/%u cases passed\n", passed, total);
   return passed == total ? 0 : 1;
}
