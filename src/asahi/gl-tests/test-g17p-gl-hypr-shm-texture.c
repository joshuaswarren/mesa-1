/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Foot -> Hyprland wl_shm texture workload discriminator.
 *
 * Hyprland uploads WL_SHM_FORMAT_ARGB8888 as a BGRA GL texture, composites it
 * with the Omarchy default window opacity (0.985), and reuses the texture for
 * same-size commits.  A size change instead allocates a new texture and drops
 * the old one after the preceding frame.  Keep those three workloads separate:
 *
 *   --static   one full BGRA upload and composite
 *   --updates  dirty row-band glTexSubImage2D uploads at 40 Hz
 *   --resize   two draw/delete/reallocate size transitions
 */

#define _POSIX_C_SOURCE 200809L

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80e1
#endif

#define OUTPUT_WIDTH 1840
#define OUTPUT_HEIGHT 960
#define UPDATE_FRAMES 120
#define UPDATE_HZ 40
#define BAND_HEIGHT 32
#define OPACITY 0.985f
#define TOLERANCE 2

enum mode {
   MODE_STATIC,
   MODE_UPDATES,
   MODE_RESIZE,
};

struct image {
   int width;
   int height;
   uint8_t *bgra;
   GLuint texture;
};

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
      fprintf(stderr, "GL46 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }

   return shader;
}

static GLuint
make_program(void)
{
   static const char vertex[] =
      "#version 300 es\n"
      "const vec2 p[3]=vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),"
      "vec2(-1.0,3.0));\n"
      "out vec2 uv;\n"
      "void main(){gl_Position=vec4(p[gl_VertexID],0.0,1.0);"
      "uv=p[gl_VertexID]*0.5+0.5;}\n";
   static const char fragment[] =
      "#version 300 es\n"
      "precision highp float;in vec2 uv;uniform sampler2D client_tex;"
      "uniform float opacity;layout(location=0) out vec4 colour;"
      "void main(){vec4 c=texture(client_tex,uv);"
      "colour=vec4(c.rgb*opacity,c.a*opacity);}\n";

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
      fprintf(stderr, "GL46 program link failed: %s\n", log);
      glDeleteProgram(program);
      return 0;
   }

   return program;
}

static void
pattern_pixel(uint8_t *pixel, int x, int y, unsigned generation)
{
   /* Memory order for DRM/WL ARGB8888 on little-endian hosts is B, G, R, A. */
   pixel[0] = (uint8_t)(17u * generation + 3u * (unsigned)x + (unsigned)y);
   pixel[1] = (uint8_t)(31u * generation + (unsigned)x + 5u * (unsigned)y);
   pixel[2] = (uint8_t)(47u * generation + 7u * (unsigned)x + 3u * (unsigned)y);
   pixel[3] = 255;
}

static void
solid_pixel(uint8_t *pixel, unsigned generation)
{
   static const uint8_t colours[3][4] = {
      {0x21, 0x58, 0xd3, 0xff},
      {0xc4, 0x39, 0x72, 0xff},
      {0x6b, 0xe1, 0x14, 0xff},
   };
   memcpy(pixel, colours[generation % 3], 4);
}

static bool
allocate_pixels(struct image *image, unsigned generation, bool solid)
{
   const size_t size = (size_t)image->width * image->height * 4;
   image->bgra = malloc(size);
   if (!image->bgra) {
      fprintf(stderr, "GL46 could not allocate %zu-byte client image\n", size);
      return false;
   }

   for (int y = 0; y < image->height; ++y) {
      for (int x = 0; x < image->width; ++x) {
         uint8_t *pixel = &image->bgra[((size_t)y * image->width + x) * 4];
         if (solid)
            solid_pixel(pixel, generation);
         else
            pattern_pixel(pixel, x, y, generation);
      }
   }

   return true;
}

static bool
create_texture(struct image *image)
{
   glGenTextures(1, &image->texture);
   glBindTexture(GL_TEXTURE_2D, image->texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, image->width);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA_EXT, image->width, image->height, 0,
                GL_BGRA_EXT, GL_UNSIGNED_BYTE, image->bgra);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "GL46 BGRA texture allocation failed: %#x\n", error);
      return false;
   }

   return true;
}

static void
destroy_image(struct image *image)
{
   glBindTexture(GL_TEXTURE_2D, 0);
   if (image->texture)
      glDeleteTextures(1, &image->texture);
   free(image->bgra);
   memset(image, 0, sizeof(*image));
}

static void
draw_frame(EGLDisplay display, EGLSurface surface, GLuint program,
           const struct image *image)
{
   glViewport(0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT);
   glDisable(GL_SCISSOR_TEST);
   glClearColor(32.0f / 255.0f, 64.0f / 255.0f, 96.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   glUseProgram(program);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, image->texture);
   glUniform1i(glGetUniformLocation(program, "client_tex"), 0);
   glUniform1f(glGetUniformLocation(program, "opacity"), OPACITY);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glBindTexture(GL_TEXTURE_2D, 0);
   glFlush();
   eglSwapBuffers(display, surface);
}

static void
update_band(struct image *image, unsigned frame)
{
   const int band_count = image->height / BAND_HEIGHT;
   const int y0 = (int)(frame % (unsigned)band_count) * BAND_HEIGHT;
   const unsigned generation = frame + 1;

   for (int y = y0; y < y0 + BAND_HEIGHT; ++y) {
      for (int x = 0; x < image->width; ++x)
         pattern_pixel(&image->bgra[((size_t)y * image->width + x) * 4],
                       x, y, generation);
   }

   glBindTexture(GL_TEXTURE_2D, image->texture);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, image->width);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, y0);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y0, image->width, BAND_HEIGHT,
                   GL_BGRA_EXT, GL_UNSIGNED_BYTE, image->bgra);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
   glBindTexture(GL_TEXTURE_2D, 0);
}

static bool
near_byte(uint8_t actual, uint8_t expected)
{
   const int delta = (int)actual - (int)expected;
   return delta >= -TOLERANCE && delta <= TOLERANCE;
}

static uint8_t
composite_byte(uint8_t client, uint8_t background)
{
   const float value = client * OPACITY + background * (1.0f - OPACITY);
   return (uint8_t)(value + 0.5f);
}

static bool
check_pixel(const struct image *image, int output_x, int output_y,
            const char *phase)
{
   uint8_t got[4] = {0};
   glReadPixels(output_x, output_y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);

   const int source_x = (int)((int64_t)output_x * image->width / OUTPUT_WIDTH);
   const int source_y = (int)((int64_t)output_y * image->height / OUTPUT_HEIGHT);
   const uint8_t *bgra =
      &image->bgra[((size_t)source_y * image->width + source_x) * 4];
   const uint8_t expected[4] = {
      composite_byte(bgra[2], 32),
      composite_byte(bgra[1], 64),
      composite_byte(bgra[0], 96),
      255,
   };

   for (unsigned c = 0; c < 4; ++c) {
      if (!near_byte(got[c], expected[c])) {
         fprintf(stderr,
                 "GL46 %s pixel (%d,%d) got %02x%02x%02x%02x, expected "
                 "%02x%02x%02x%02x (+/-%d)\n",
                 phase, output_x, output_y, got[0], got[1], got[2], got[3],
                 expected[0], expected[1], expected[2], expected[3],
                 TOLERANCE);
         return false;
      }
   }

   return glGetError() == GL_NO_ERROR;
}

static bool
validate(const struct image *image, const char *phase, bool noop)
{
   if (noop)
      return glGetError() == GL_NO_ERROR;

   const int points[][2] = {
      {0, 0},
      {OUTPUT_WIDTH / 3, OUTPUT_HEIGHT / 3},
      {OUTPUT_WIDTH / 2, OUTPUT_HEIGHT / 2},
      {OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1},
   };

   for (unsigned i = 0; i < sizeof(points) / sizeof(points[0]); ++i) {
      if (!check_pixel(image, points[i][0], points[i][1], phase))
         return false;
   }

   fprintf(stderr, "GL46 %s: exact BGRA/0.985 composite (+/-%d)\n",
           phase, TOLERANCE);
   return true;
}

static bool
sleep_frame(void)
{
   const struct timespec delay = {
      .tv_sec = 0,
      .tv_nsec = 1000000000L / UPDATE_HZ,
   };
   return nanosleep(&delay, NULL) == 0;
}

static const char *
mode_name(enum mode mode)
{
   switch (mode) {
   case MODE_STATIC: return "static";
   case MODE_UPDATES: return "updates";
   case MODE_RESIZE: return "resize";
   }
   return "unknown";
}

static bool
parse_mode(int argc, char **argv, enum mode *mode)
{
   if (argc != 2)
      return false;
   if (strcmp(argv[1], "--static") == 0)
      *mode = MODE_STATIC;
   else if (strcmp(argv[1], "--updates") == 0)
      *mode = MODE_UPDATES;
   else if (strcmp(argv[1], "--resize") == 0)
      *mode = MODE_RESIZE;
   else
      return false;
   return true;
}

int
main(int argc, char **argv)
{
   enum mode mode;
   if (!parse_mode(argc, argv, &mode)) {
      fprintf(stderr, "usage: %s --static|--updates|--resize\n", argv[0]);
      return 2;
   }

   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "GL46 EGL initialization failed\n");
      return EXIT_FAILURE;
   }

   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) ||
       count != 1)
      return EXIT_FAILURE;

   const EGLint surface_attrs[] = {
      EGL_WIDTH, OUTPUT_WIDTH,
      EGL_HEIGHT, OUTPUT_HEIGHT,
      EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
   if (!extensions || !strstr(extensions, "GL_EXT_texture_format_BGRA8888")) {
      fprintf(stderr, "GL46 requires GL_EXT_texture_format_BGRA8888\n");
      return 77;
   }

   GLuint program = make_program();
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glDisable(GL_DITHER);
   if (!program)
      return EXIT_FAILURE;

   struct image image = {
      .width = OUTPUT_WIDTH,
      .height = OUTPUT_HEIGHT,
   };
   const bool solid = mode == MODE_RESIZE;
   if (!allocate_pixels(&image, 0, solid) || !create_texture(&image))
      return EXIT_FAILURE;

   draw_frame(display, surface, program, &image);

   if (mode == MODE_UPDATES) {
      for (unsigned frame = 0; frame < UPDATE_FRAMES; ++frame) {
         update_band(&image, frame);
         if (glGetError() != GL_NO_ERROR) {
            fprintf(stderr, "GL46 row-band update %u failed\n", frame);
            return EXIT_FAILURE;
         }
         draw_frame(display, surface, program, &image);
         if (!noop && !sleep_frame())
            return EXIT_FAILURE;
      }
   } else if (mode == MODE_RESIZE) {
      static const int sizes[2][2] = {
         {1760, 920},
         {OUTPUT_WIDTH, OUTPUT_HEIGHT},
      };

      for (unsigned transition = 0; transition < 2; ++transition) {
         /* Match Hyprland: the preceding draw may still be in flight when the
          * old client texture loses its final application reference. */
         destroy_image(&image);
         image.width = sizes[transition][0];
         image.height = sizes[transition][1];
         if (!allocate_pixels(&image, transition + 1, true) ||
             !create_texture(&image))
            return EXIT_FAILURE;
         draw_frame(display, surface, program, &image);
      }
   }

   if (!validate(&image, "immediate", noop))
      return EXIT_FAILURE;
   if (!noop)
      sleep(3);
   if (!validate(&image, "retained +3s", noop))
      return EXIT_FAILURE;

   destroy_image(&image);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);

   printf("GL46 Hypr wl_shm BGRA texture %s %s\n", mode_name(mode),
          noop ? "submitted (pixels skipped)" : "exact immediately/3s");
   return EXIT_SUCCESS;
}
