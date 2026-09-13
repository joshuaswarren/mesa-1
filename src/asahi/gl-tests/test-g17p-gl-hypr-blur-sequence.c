/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Hyprland's legacy layer-blur sequence through Zink: two BGRA8 workbuffer
 * passes share the monitor stencil attachment, an alpha-discarding client
 * mask is drawn with color writes disabled, the captured background is drawn
 * through stencil, and the premultiplied client is drawn after state restore.
 * Repeated overlapping boxes and a partial-damage pass expose stale FBO,
 * stencil, color-mask, blend, and damage state independently of blur math.
 */

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH         32
#define HEIGHT        16
#define CLIENT_WIDTH  8
#define CLIENT_HEIGHT 8
#define DATA_BYTES    (WIDTH * HEIGHT * 4)
#define GUARD_SIZE    64
#define GUARD_BYTE    0xa5

#ifndef GL_BGRA8_EXT
#define GL_BGRA8_EXT 0x93A1
#endif

struct guarded_pixels {
   uint8_t before[GUARD_SIZE];
   uint8_t data[DATA_BYTES];
   uint8_t after[GUARD_SIZE];
};

struct box {
   int x, y, width, height;
};

static uint8_t expected[DATA_BYTES];
static uint8_t client[CLIENT_WIDTH * CLIENT_HEIGHT * 4];

static bool
has_extension(const char *extensions, const char *name)
{
   const size_t length = strlen(name);
   const char *match = extensions;
   while ((match = strstr(match, name))) {
      if ((match == extensions || match[-1] == ' ') &&
          (match[length] == '\0' || match[length] == ' '))
         return true;
      match += length;
   }
   return false;
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
      fprintf(stderr, "GL43 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *fragment)
{
   static const char vertex[] =
      "#version 300 es\n"
      "const vec2 p[3]=vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),"
      "vec2(-1.0,3.0));\n"
      "out vec2 uv;\n"
      "void main(){gl_Position=vec4(p[gl_VertexID],0.0,1.0);"
      "uv=p[gl_VertexID]*0.5+0.5;}\n";
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment);
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
      fprintf(stderr, "GL43 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
guards_ok(const struct guarded_pixels *pixels)
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (pixels->before[i] != GUARD_BYTE ||
          pixels->after[i] != GUARD_BYTE)
         return false;
   }
   return true;
}

static bool
near_byte(uint8_t actual, uint8_t want)
{
   return actual + 2 >= want && actual <= want + 2;
}

static bool
read_expected(const char *phase, bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   GLenum error = glGetError();
   if (!guards_ok(&pixels)) {
      fprintf(stderr, "GL43 %s readback guard changed\n", phase);
      return false;
   }
   if (noop)
      return error == GL_NO_ERROR;

   for (unsigned pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
      const uint8_t *got = &pixels.data[pixel * 4];
      const uint8_t *want = &expected[pixel * 4];
      for (unsigned c = 0; c < 4; ++c) {
         if (!near_byte(got[c], want[c])) {
            fprintf(stderr,
                    "GL43 %s pixel %u got %02x%02x%02x%02x, expected "
                    "%02x%02x%02x%02x (+/-2), error %#x\n",
                    phase, pixel, got[0], got[1], got[2], got[3],
                    want[0], want[1], want[2], want[3], error);
            return false;
         }
      }
   }
   fprintf(stderr, "GL43 %s: exact sequenced pixels (+/-2)\n", phase);
   return error == GL_NO_ERROR;
}

static GLuint
make_texture(GLenum internal_format, int width, int height,
             const uint8_t *pixels)
{
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   if (internal_format == GL_BGRA8_EXT)
      glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height);
   else
      glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   return texture;
}

static GLuint
make_framebuffer(GLuint texture, GLuint stencil)
{
   GLuint framebuffer;
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, stencil);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "GL43 shared-stencil BGRA8 framebuffer incomplete\n");
      return 0;
   }
   return framebuffer;
}

static void
bind_sampler(GLuint program, GLuint texture)
{
   glUseProgram(program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
}

static void
draw_fullscreen(GLuint program, GLuint texture)
{
   glViewport(0, 0, WIDTH, HEIGHT);
   bind_sampler(program, texture);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static uint8_t
blend_byte(uint8_t src, uint8_t dst, uint8_t alpha)
{
   const float out = src / 255.0f +
                     (dst / 255.0f) * (1.0f - alpha / 255.0f);
   const unsigned byte = (unsigned)(out * 255.0f + 0.5f);
   return byte > 255 ? 255 : (uint8_t)byte;
}

static void
update_reference(struct box box, const struct box *damage)
{
   uint8_t snapshot[DATA_BYTES];
   memcpy(snapshot, expected, sizeof(snapshot));

   for (int local_y = 0; local_y < box.height; ++local_y) {
      for (int local_x = 0; local_x < box.width; ++local_x) {
         const int x = box.x + local_x;
         const int y = box.y + local_y;
         if (damage &&
             (x < damage->x || y < damage->y ||
              x >= damage->x + damage->width ||
              y >= damage->y + damage->height))
            continue;

         uint8_t *dst = &expected[(y * WIDTH + x) * 4];
         const uint8_t *src = &client[(local_y * CLIENT_WIDTH + local_x) * 4];
         if (src[3] >= 128) {
            const int shifted_x = x > 0 ? x - 1 : 0;
            memcpy(dst, &snapshot[(y * WIDTH + shifted_x) * 4], 4);
         }
         for (unsigned c = 0; c < 3; ++c)
            dst[c] = blend_byte(src[c], dst[c], src[3]);
         dst[3] = blend_byte(src[3], dst[3], src[3]);
      }
   }
}

static void
set_scissor(const struct box *damage, int expand_left)
{
   if (!damage) {
      glDisable(GL_SCISSOR_TEST);
      return;
   }
   glEnable(GL_SCISSOR_TEST);
   const int x = damage->x > expand_left ? damage->x - expand_left : 0;
   const int width = damage->width + damage->x - x;
   glScissor(x, damage->y, width, damage->height);
}

static bool
draw_blurred_client(unsigned index, struct box box, const struct box *damage,
                    GLuint main_fbo, GLuint main_texture,
                    GLuint work_a_fbo, GLuint work_a_texture,
                    GLuint work_b_fbo, GLuint work_b_texture,
                    GLuint client_texture, GLuint copy_program,
                    GLuint shift_program, GLuint screen_program,
                    GLuint mask_program,
                    GLuint client_program, bool noop)
{
   glDisable(GL_BLEND);
   glDisable(GL_STENCIL_TEST);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

   glBindFramebuffer(GL_FRAMEBUFFER, work_a_fbo);
   set_scissor(damage, 1);
   draw_fullscreen(copy_program, main_texture);

   glBindFramebuffer(GL_FRAMEBUFFER, work_b_fbo);
   set_scissor(damage, 0);
   draw_fullscreen(shift_program, work_a_texture);

   glBindFramebuffer(GL_FRAMEBUFFER, main_fbo);
   glViewport(box.x, box.y, box.width, box.height);
   set_scissor(damage, 0);
   glStencilMask(0xff);
   glClearStencil(0);
   glDisable(GL_SCISSOR_TEST);
   glClear(GL_STENCIL_BUFFER_BIT);
   set_scissor(damage, 0);
   glEnable(GL_STENCIL_TEST);
   glStencilFunc(GL_ALWAYS, 1, 0xff);
   glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
   glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
   bind_sampler(mask_program, client_texture);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glStencilFunc(GL_EQUAL, 1, 0xff);
   glStencilMask(0x00);
   glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
   bind_sampler(screen_program, work_b_texture);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   glDisable(GL_STENCIL_TEST);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   bind_sampler(client_program, client_texture);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);
   glFinish();

   update_reference(box, damage);
   char phase[96];
   snprintf(phase, sizeof(phase), "overlapping blur/client pass %u%s", index,
            damage ? " partial-damage" : "");
   return read_expected(phase, noop);
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      return EXIT_FAILURE;

   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_DEPTH_SIZE, 24,
      EGL_STENCIL_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return EXIT_FAILURE;
   const EGLint surface_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
   if (!extensions ||
       !has_extension(extensions, "GL_EXT_texture_format_BGRA8888")) {
      fprintf(stderr, "GL43 BGRA8 render attachment support missing\n");
      return EXIT_FAILURE;
   }

   static const char copy_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D tex;"
      "layout(location=0) out vec4 colour;"
      "void main(){colour=texture(tex,uv);}";
   static const char shift_fragment[] =
      "#version 300 es\n"
      "precision highp float;uniform sampler2D tex;"
      "layout(location=0) out vec4 colour;"
      "void main(){ivec2 p=ivec2(gl_FragCoord.xy);"
      "colour=texelFetch(tex,ivec2(max(p.x-1,0),p.y),0);}";
   static const char screen_fragment[] =
      "#version 300 es\n"
      "precision highp float;uniform sampler2D tex;"
      "layout(location=0) out vec4 colour;"
      "void main(){colour=texelFetch(tex,ivec2(gl_FragCoord.xy),0);}";
   static const char mask_fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D tex;"
      "layout(location=0) out vec4 colour;"
      "void main(){vec4 c=texture(tex,uv);if(c.a<0.5)discard;colour=c;}";
   GLuint copy_program = make_program(copy_fragment);
   GLuint shift_program = make_program(shift_fragment);
   GLuint screen_program = make_program(screen_fragment);
   GLuint mask_program = make_program(mask_fragment);
   GLuint client_program = copy_program;
   if (!copy_program || !shift_program || !screen_program || !mask_program)
      return EXIT_FAILURE;

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   for (unsigned y = 0; y < CLIENT_HEIGHT; ++y) {
      for (unsigned x = 0; x < CLIENT_WIDTH; ++x) {
         static const uint8_t alpha[CLIENT_WIDTH] = {
            0, 0, 64, 64, 128, 128, 255, 255,
         };
         uint8_t *pixel = &client[(y * CLIENT_WIDTH + x) * 4];
         pixel[0] = alpha[x];
         pixel[1] = (uint8_t)((alpha[x] + 2) / 4);
         pixel[2] = (uint8_t)((alpha[x] * 3 + 2) / 4);
         pixel[3] = alpha[x];
      }
   }

   GLuint client_texture =
      make_texture(GL_RGBA8, CLIENT_WIDTH, CLIENT_HEIGHT, client);
   GLuint main_texture = make_texture(GL_BGRA8_EXT, WIDTH, HEIGHT, NULL);
   GLuint work_a_texture = make_texture(GL_BGRA8_EXT, WIDTH, HEIGHT, NULL);
   GLuint work_b_texture = make_texture(GL_BGRA8_EXT, WIDTH, HEIGHT, NULL);

   GLuint stencil;
   glGenRenderbuffers(1, &stencil);
   glBindRenderbuffer(GL_RENDERBUFFER, stencil);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT);

   GLuint main_fbo = make_framebuffer(main_texture, stencil);
   GLuint work_a_fbo = make_framebuffer(work_a_texture, stencil);
   GLuint work_b_fbo = make_framebuffer(work_b_texture, stencil);
   if (!main_fbo || !work_a_fbo || !work_b_fbo)
      return EXIT_FAILURE;

   glBindFramebuffer(GL_FRAMEBUFFER, main_fbo);
   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_DITHER);
   glClearColor(32.0f / 255.0f, 128.0f / 255.0f, 224.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   for (unsigned pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
      expected[pixel * 4 + 0] = 32;
      expected[pixel * 4 + 1] = 128;
      expected[pixel * 4 + 2] = 224;
      expected[pixel * 4 + 3] = 255;
   }

   const struct box boxes[] = {
      {4, 2, CLIENT_WIDTH, CLIENT_HEIGHT},
      {8, 5, CLIENT_WIDTH, CLIENT_HEIGHT},
      {12, 1, CLIENT_WIDTH, CLIENT_HEIGHT},
   };
   for (unsigned i = 0; i < sizeof(boxes) / sizeof(boxes[0]); ++i) {
      if (!draw_blurred_client(i, boxes[i], NULL, main_fbo, main_texture,
                               work_a_fbo, work_a_texture, work_b_fbo,
                               work_b_texture, client_texture, copy_program,
                               shift_program, screen_program, mask_program,
                               client_program, noop))
         return EXIT_FAILURE;
   }

   const struct box damage = {14, 3, 4, 5};
   if (!draw_blurred_client(3, boxes[2], &damage, main_fbo, main_texture,
                            work_a_fbo, work_a_texture, work_b_fbo,
                            work_b_texture, client_texture, copy_program,
                            shift_program, screen_program, mask_program,
                            client_program, noop))
      return EXIT_FAILURE;

   glDeleteFramebuffers(1, &main_fbo);
   glDeleteFramebuffers(1, &work_a_fbo);
   glDeleteFramebuffers(1, &work_b_fbo);
   glDeleteRenderbuffers(1, &stencil);
   glDeleteTextures(1, &main_texture);
   glDeleteTextures(1, &work_a_texture);
   glDeleteTextures(1, &work_b_texture);
   glDeleteTextures(1, &client_texture);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(copy_program);
   glDeleteProgram(shift_program);
   glDeleteProgram(screen_program);
   glDeleteProgram(mask_program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   puts(noop ? "GL43 Hypr blur sequence submitted; pixels skipped"
             : "GL43 shared-stencil multi-window blur/client/damage sequence "
               "exact");
   return EXIT_SUCCESS;
}
