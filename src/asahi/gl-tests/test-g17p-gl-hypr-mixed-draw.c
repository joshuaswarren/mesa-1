/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * F48: one long-lived GLES context alternating the three visible paths in the
 * broken Hyprland/Omarchy scene: Hyprland's rounded border, an R8 glyph mask,
 * and a little-endian WL_SHM_FORMAT_ARGB8888 (BGRA byte-order) client texture.
 *
 * Each pass restores the ordinary GL state it owns after a different pass has
 * dirtied viewport, scissor, blend equation/factors, colour mask, cull, depth,
 * stencil, topology, texture unit/filter/swizzle, and unpack state. Readback
 * happens only after the full sequence so it cannot accidentally serialize the
 * draws under test. --api-barrier adds glMemoryBarrier(GL_ALL_BARRIER_BITS)
 * immediately before each draw; --tight adds no application barrier.
 */

#define _GNU_SOURCE

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80e1
#endif

#define WIDTH 512
#define HEIGHT 320
#define CYCLES 64

#define CLIENT_X 32
#define CLIENT_Y 32
#define CLIENT_W 192
#define CLIENT_H 256
#define GLYPH_X 288
#define GLYPH_Y 32
#define GLYPH_W 192
#define GLYPH_H 256

static const uint8_t background[4] = {10, 13, 18, 255};

struct vertex {
   float x, y, u, v, mu, mv;
};

static const struct vertex full_vertices[4] = {
   {-1, -1, 0, 0, 0, 0},
   {-1,  1, 0, 1, 0, 1},
   { 1, -1, 1, 0, 1, 0},
   { 1,  1, 1, 1, 1, 1},
};

struct programs {
   GLuint border;
   GLuint client;
   GLuint glyph;
   GLuint dirty;
};

struct objects {
   GLuint vao;
   GLuint vbo;
   GLuint ibo;
   GLuint client_texture;
   GLuint glyph_texture;
};

struct run_state {
   bool api_barrier;
   unsigned draws;
};

static char *
read_file(const char *path)
{
   FILE *file = fopen(path, "rb");
   if (!file)
      return NULL;
   if (fseek(file, 0, SEEK_END)) {
      fclose(file);
      return NULL;
   }
   long length = ftell(file);
   if (length < 1 || fseek(file, 0, SEEK_SET)) {
      fclose(file);
      return NULL;
   }
   char *source = malloc((size_t)length + 1);
   if (!source || fread(source, 1, (size_t)length, file) != (size_t)length) {
      free(source);
      fclose(file);
      return NULL;
   }
   source[length] = '\0';
   fclose(file);
   return source;
}

static GLuint
compile_shader(GLenum stage, const char *source, const char *label)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "F48 %s compile failed: %s\n", label, log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint
make_program(const char *vs_source, const char *fs_source, const char *label,
             bool bind_hypr_names)
{
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source, label);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source, label);
   if (!vs || !fs) {
      glDeleteShader(vs);
      glDeleteShader(fs);
      return 0;
   }

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   if (bind_hypr_names) {
      glBindAttribLocation(program, 0, "pos");
      glBindAttribLocation(program, 1, "texcoord");
      glBindAttribLocation(program, 2, "texcoordMatte");
   }
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);

   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096] = {0};
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "F48 %s link failed: %s\n", label, log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static bool
gl_ok(const char *where)
{
   bool ok = true;
   for (GLenum error; (error = glGetError()) != GL_NO_ERROR;) {
      fprintf(stderr, "F48 %s GL error %#x\n", where, error);
      ok = false;
   }
   return ok;
}

static void
before_draw(struct run_state *run)
{
   if (run->api_barrier)
      glMemoryBarrier(GL_ALL_BARRIER_BITS);
   run->draws++;
}

static void
set_border_uniforms(GLuint program)
{
   static const GLfloat identity[9] = {
      1, 0, 0, 0, 1, 0, 0, 0, 1,
   };
   static const GLfloat orange_oklab[8] = {
      0.765240491f, 0.080701147f, 0.155514242f, 1.0f,
      0.765240491f, 0.080701147f, 0.155514242f, 1.0f,
   };

   GLint loc = glGetUniformLocation(program, "proj");
   if (loc >= 0)
      glUniformMatrix3fv(loc, 1, GL_FALSE, identity);

   const struct {
      const char *name;
      float x, y;
   } vec2s[] = {
      {"fullSizeUntransformed", WIDTH, HEIGHT},
      {"topLeft", 0, 0},
      {"fullSize", WIDTH, HEIGHT},
   };
   for (unsigned i = 0; i < sizeof(vec2s) / sizeof(vec2s[0]); ++i) {
      loc = glGetUniformLocation(program, vec2s[i].name);
      if (loc >= 0)
         glUniform2f(loc, vec2s[i].x, vec2s[i].y);
   }

   const struct {
      const char *name;
      float value;
   } scalars[] = {
      {"radiusOuter", 12}, {"thick", 2}, {"radius", 12},
      {"roundingPower", 2}, {"angle", 0}, {"angle2", 0},
      {"gradientLerp", 0}, {"alpha", 1},
   };
   for (unsigned i = 0; i < sizeof(scalars) / sizeof(scalars[0]); ++i) {
      loc = glGetUniformLocation(program, scalars[i].name);
      if (loc >= 0)
         glUniform1f(loc, scalars[i].value);
   }

   loc = glGetUniformLocation(program, "gradientLength");
   if (loc >= 0)
      glUniform1i(loc, 2);
   loc = glGetUniformLocation(program, "gradient2Length");
   if (loc >= 0)
      glUniform1i(loc, 0);
   loc = glGetUniformLocation(program, "gradient[0]");
   if (loc >= 0)
      glUniform4fv(loc, 2, orange_oklab);
}

static void
set_visible_state(GLint x, GLint y, GLsizei width, GLsizei height)
{
   glViewport(x, y, width, height);
   glEnable(GL_SCISSOR_TEST);
   glScissor(x, y, width, height);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDisable(GL_DEPTH_TEST);
   glDepthMask(GL_FALSE);
   glDisable(GL_STENCIL_TEST);
   glStencilMask(0xff);
   glDisable(GL_CULL_FACE);
   glFrontFace(GL_CCW);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
}

static void
clear_panel(GLint x, GLint y, GLsizei width, GLsizei height)
{
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDepthMask(GL_FALSE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   glEnable(GL_SCISSOR_TEST);
   glScissor(x, y, width, height);
   glClearColor(background[0] / 255.0f, background[1] / 255.0f,
                background[2] / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
}

static void
draw_border(const struct programs *programs, const struct objects *objects,
            struct run_state *run)
{
   set_visible_state(0, 0, WIDTH, HEIGHT);
   glDisable(GL_SCISSOR_TEST);
   glUseProgram(programs->border);
   set_border_uniforms(programs->border);
   glBindVertexArray(objects->vao);
   before_draw(run);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void
fill_client_pixels(uint8_t pixels[16], unsigned generation)
{
   /* Four asymmetric BGRA memory-order texels. The generation changes all
    * channels differently, so stale upload data and an R/B swap are distinct. */
   static const uint8_t initial_rgba[4][4] = {
      {11, 43, 101, 255}, {73, 149, 29, 255},
      {211, 37, 83, 255}, {19, 227, 167, 255},
   };
   for (unsigned i = 0; i < 4; ++i) {
      const uint8_t r = (uint8_t)(initial_rgba[i][0] + generation * 3u);
      const uint8_t g = (uint8_t)(initial_rgba[i][1] + generation * 5u);
      const uint8_t b = (uint8_t)(initial_rgba[i][2] + generation * 7u);
      pixels[i * 4 + 0] = b;
      pixels[i * 4 + 1] = g;
      pixels[i * 4 + 2] = r;
      pixels[i * 4 + 3] = 255;
   }
}

static void
upload_client_texture(GLuint texture, unsigned generation)
{
   uint8_t staging[4 * 4 * 4];
   memset(staging, 0xa5, sizeof(staging));
   uint8_t compact[16];
   fill_client_pixels(compact, generation);

   /* Put the 2x2 image inside a 4-pixel-row staging image to exercise the
    * same row-length/skip state used by wl_shm subimage uploads. */
   for (unsigned y = 0; y < 2; ++y)
      memcpy(&staging[((y + 1) * 4 + 1) * 4], &compact[y * 2 * 4], 2 * 4);

   glActiveTexture(GL_TEXTURE3);
   glBindTexture(GL_TEXTURE_2D, texture);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 4);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 1);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, GL_BGRA_EXT,
                   GL_UNSIGNED_BYTE, staging);

   /* Deliberately leave hostile unpack state for the glyph uploader to reset. */
   glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 13);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 2);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 3);
}

static void
draw_client(const struct programs *programs, const struct objects *objects,
            struct run_state *run, unsigned generation)
{
   upload_client_texture(objects->client_texture, generation);
   clear_panel(CLIENT_X, CLIENT_Y, CLIENT_W, CLIENT_H);
   set_visible_state(CLIENT_X, CLIENT_Y, CLIENT_W, CLIENT_H);
   glUseProgram(programs->client);
   glBindVertexArray(objects->vao);
   glActiveTexture(GL_TEXTURE3);
   glBindTexture(GL_TEXTURE_2D, objects->client_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
   glUniform1i(glGetUniformLocation(programs->client, "client_tex"), 3);
   glUniform1f(glGetUniformLocation(programs->client, "opacity"), 0.75f);
   before_draw(run);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void
upload_glyph_texture(GLuint texture, unsigned generation)
{
   static const uint8_t masks[2][4] = {
      {0, 64, 160, 255},
      {255, 176, 80, 0},
   };
   const uint8_t *mask = masks[generation & 1u];

   glActiveTexture(GL_TEXTURE6);
   glBindTexture(GL_TEXTURE_2D, texture);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
   glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 1, GL_RED,
                   GL_UNSIGNED_BYTE, mask);
}

static void
draw_glyph(const struct programs *programs, const struct objects *objects,
           struct run_state *run, unsigned generation)
{
   upload_glyph_texture(objects->glyph_texture, generation);
   clear_panel(GLYPH_X, GLYPH_Y, GLYPH_W, GLYPH_H);
   set_visible_state(GLYPH_X, GLYPH_Y, GLYPH_W, GLYPH_H);
   glUseProgram(programs->glyph);
   glBindVertexArray(objects->vao);
   glActiveTexture(GL_TEXTURE6);
   glBindTexture(GL_TEXTURE_2D, objects->glyph_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_ZERO);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
   glUniform1i(glGetUniformLocation(programs->glyph, "atlas"), 6);
   before_draw(run);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
}

static void
draw_dirty_state(const struct programs *programs, const struct objects *objects,
                 struct run_state *run, unsigned cycle)
{
   glUseProgram(programs->dirty);
   glBindVertexArray(objects->vao);
   glViewport(241, 145, 30, 30);
   glEnable(GL_SCISSOR_TEST);
   glScissor(245, 149, 22, 22);
   glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_REVERSE_SUBTRACT, GL_FUNC_SUBTRACT);
   glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR,
                       GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA);
   glBlendColor(0.2f, 0.7f, 0.4f, 0.6f);

   glEnable(GL_DEPTH_TEST);
   glDepthMask(GL_TRUE);
   glDepthFunc(GL_NEVER);
   before_draw(run);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   glDisable(GL_DEPTH_TEST);
   glEnable(GL_STENCIL_TEST);
   glStencilMask((GLuint)(1u << (cycle & 7u)));
   glStencilFunc(GL_NEVER, (GLint)(cycle & 7u), 0x7f);
   glStencilOp(GL_REPLACE, GL_INCR_WRAP, GL_DECR_WRAP);
   before_draw(run);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   glDisable(GL_STENCIL_TEST);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_FRONT);
   glFrontFace(GL_CW);
   before_draw(run);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   before_draw(run);
   glDrawArrays(GL_LINES, 0, 2);
   before_draw(run);
   glDrawArrays(GL_POINTS, 0, 1);
}

static bool
near_byte(uint8_t actual, uint8_t expected, unsigned tolerance)
{
   const int delta = (int)actual - (int)expected;
   return delta >= -(int)tolerance && delta <= (int)tolerance;
}

static bool
check_pixel(const char *name, GLint x, GLint y, const uint8_t expected[4],
            unsigned tolerance)
{
   uint8_t got[4] = {0};
   glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
   bool ok = true;
   for (unsigned c = 0; c < 4; ++c)
      ok &= near_byte(got[c], expected[c], tolerance);
   if (!ok) {
      fprintf(stderr,
              "F48 %-16s (%d,%d) got %u,%u,%u,%u expected "
              "%u,%u,%u,%u (+/-%u)\n",
              name, x, y, got[0], got[1], got[2], got[3], expected[0],
              expected[1], expected[2], expected[3], tolerance);
   }
   return ok;
}

static uint8_t
composite_byte(uint8_t source, uint8_t destination, uint8_t alpha)
{
   const unsigned value = (unsigned)source * alpha +
                          (unsigned)destination * (255u - alpha);
   return (uint8_t)((value + 127u) / 255u);
}

static bool
validate_pixels(unsigned final_generation)
{
   bool ok = true;
   static const struct {
      const char *name;
      GLint x, y;
      uint8_t expected[4];
   } border_probes[] = {
      {"border-top", WIDTH / 2, 1, {255, 148, 0, 255}},
      {"border-bottom", WIDTH / 2, HEIGHT - 2, {255, 148, 0, 255}},
      {"border-left", 1, HEIGHT / 2, {255, 148, 0, 255}},
      {"border-right", WIDTH - 2, HEIGHT / 2, {255, 148, 0, 255}},
      {"rounded-corner", 0, 0, {10, 13, 18, 255}},
      {"state-dirty-gap", WIDTH / 2, HEIGHT / 2, {10, 13, 18, 255}},
   };
   for (unsigned i = 0; i < sizeof(border_probes) / sizeof(border_probes[0]);
        ++i)
      ok &= check_pixel(border_probes[i].name, border_probes[i].x,
                        border_probes[i].y, border_probes[i].expected, 1);

   uint8_t compact[16];
   fill_client_pixels(compact, final_generation);
   const GLint client_points[4][2] = {
      {CLIENT_X + CLIENT_W / 4, CLIENT_Y + CLIENT_H / 4},
      {CLIENT_X + 3 * CLIENT_W / 4, CLIENT_Y + CLIENT_H / 4},
      {CLIENT_X + CLIENT_W / 4, CLIENT_Y + 3 * CLIENT_H / 4},
      {CLIENT_X + 3 * CLIENT_W / 4, CLIENT_Y + 3 * CLIENT_H / 4},
   };
   for (unsigned i = 0; i < 4; ++i) {
      const uint8_t *bgra = &compact[i * 4];
      const uint8_t expected[4] = {
         composite_byte(bgra[2], background[0], 191),
         composite_byte(bgra[1], background[1], 191),
         composite_byte(bgra[0], background[2], 191),
         255,
      };
      char name[24];
      snprintf(name, sizeof(name), "BGRA-texel-%u", i);
      ok &= check_pixel(name, client_points[i][0], client_points[i][1],
                        expected, 2);
   }

   static const uint8_t masks[2][4] = {
      {0, 64, 160, 255},
      {255, 176, 80, 0},
   };
   const uint8_t *mask = masks[final_generation & 1u];
   for (unsigned i = 0; i < 4; ++i) {
      const uint8_t a = mask[i];
      const uint8_t expected[4] = {
         composite_byte(51, background[0], a),
         composite_byte(204, background[1], a),
         composite_byte(102, background[2], a),
         255,
      };
      char name[24];
      snprintf(name, sizeof(name), "glyph-mask-%u", i);
      ok &= check_pixel(name,
                        GLYPH_X + (GLint)((2u * i + 1u) * GLYPH_W / 8u),
                        GLYPH_Y + GLYPH_H / 2, expected, 2);
   }
   return gl_ok("readback") && ok;
}

int
main(int argc, char **argv)
{
   if (argc != 4 ||
       (strcmp(argv[3], "--tight") && strcmp(argv[3], "--api-barrier"))) {
      fprintf(stderr,
              "usage: %s tex300.vert border-rounded.frag "
              "--tight|--api-barrier\n",
              argv[0]);
      return 2;
   }

   struct run_state run = {
      .api_barrier = strcmp(argv[3], "--api-barrier") == 0,
   };
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;

   char *border_vs_source = read_file(argv[1]);
   char *border_fs_source = read_file(argv[2]);
   if (!border_vs_source || !border_fs_source) {
      fprintf(stderr, "F48 could not read exact Hyprland shaders\n");
      return 2;
   }

   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");
   EGLDisplay display = get_platform_display
      ? get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
                             EGL_DEFAULT_DISPLAY, NULL)
      : eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "F48 EGL initialization failed: %#x\n", eglGetError());
      return 1;
   }

   const EGLint context_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1,
      EGL_NONE,
   };
   EGLContext context = eglCreateContext(display, EGL_NO_CONFIG_KHR,
                                         EGL_NO_CONTEXT, context_attrs);
   if (context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
      fprintf(stderr, "F48 GLES 3.1 context failed: %#x\n", eglGetError());
      return 1;
   }

   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *version = (const char *)glGetString(GL_VERSION);
   const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
   printf("F48 renderer: %s\n", renderer ? renderer : "(none)");
   printf("F48 version: %s\n", version ? version : "(none)");
   printf("F48 mode: %s; ZINK_DEBUG=%s\n",
          run.api_barrier ? "api-barrier" : "tight",
          getenv("ZINK_DEBUG") ? getenv("ZINK_DEBUG") : "(unset)");
   if (!extensions || !strstr(extensions, "GL_EXT_texture_format_BGRA8888")) {
      fprintf(stderr, "F48 requires GL_EXT_texture_format_BGRA8888\n");
      return 77;
   }

   static const char texture_vs[] =
      "#version 300 es\n"
      "layout(location=0) in vec2 pos; layout(location=1) in vec2 uv;\n"
      "out vec2 v_uv; void main(){gl_Position=vec4(pos,0,1);v_uv=uv;}\n";
   static const char client_fs[] =
      "#version 300 es\nprecision highp float;\n"
      "uniform sampler2D client_tex; uniform float opacity;\n"
      "in vec2 v_uv; out vec4 color;\n"
      "void main(){vec4 c=texture(client_tex,v_uv);"
      "color=vec4(c.rgb*opacity,c.a*opacity);}\n";
   static const char glyph_fs[] =
      "#version 300 es\nprecision highp float;\n"
      "uniform sampler2D atlas; in vec2 v_uv; out vec4 color;\n"
      "void main(){float a=texture(atlas,v_uv).r;"
      "color=vec4(vec3(.2,.8,.4)*a,a);}\n";
   static const char dirty_vs[] =
      "#version 300 es\nlayout(location=0) in vec2 pos;\n"
      "void main(){gl_Position=vec4(pos,.25,1);gl_PointSize=2.;}\n";
   static const char dirty_fs[] =
      "#version 300 es\nprecision highp float;out vec4 color;\n"
      "void main(){color=vec4(.7,.2,.5,.5);}\n";

   struct programs programs = {
      .border = make_program(border_vs_source, border_fs_source,
                             "Hyprland border", true),
      .client = make_program(texture_vs, client_fs, "BGRA client", false),
      .glyph = make_program(texture_vs, glyph_fs, "R8 glyph", false),
      .dirty = make_program(dirty_vs, dirty_fs, "dirty state", false),
   };
   free(border_vs_source);
   free(border_fs_source);
   if (!programs.border || !programs.client || !programs.glyph ||
       !programs.dirty)
      return 1;

   struct objects objects = {0};
   glGenVertexArrays(1, &objects.vao);
   glBindVertexArray(objects.vao);
   glGenBuffers(1, &objects.vbo);
   glBindBuffer(GL_ARRAY_BUFFER, objects.vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(full_vertices), full_vertices,
                GL_DYNAMIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)offsetof(struct vertex, x));
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)offsetof(struct vertex, u));
   glEnableVertexAttribArray(2);
   glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(struct vertex),
                         (void *)offsetof(struct vertex, mu));
   static const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
   glGenBuffers(1, &objects.ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, objects.ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                GL_STATIC_DRAW);

   glGenTextures(1, &objects.client_texture);
   glBindTexture(GL_TEXTURE_2D, objects.client_texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA_EXT, 2, 2, 0, GL_BGRA_EXT,
                GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glGenTextures(1, &objects.glyph_texture);
   glBindTexture(GL_TEXTURE_2D, objects.glyph_texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, 4, 1);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   GLuint colour = 0, depth_stencil = 0, fbo = 0;
   glGenTextures(1, &colour);
   glBindTexture(GL_TEXTURE_2D, colour);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
   glGenRenderbuffers(1, &depth_stencil);
   glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          colour, 0);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, depth_stencil);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "F48 framebuffer incomplete\n");
      return 1;
   }

   glDisable(GL_DITHER);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDepthMask(GL_TRUE);
   glClearColor(background[0] / 255.0f, background[1] / 255.0f,
                background[2] / 255.0f, 1.0f);
   glClearDepthf(1.0f);
   glClearStencil(0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

   for (unsigned cycle = 0; cycle < CYCLES; ++cycle) {
      /* Rotate the three visible orders so every pairwise transition occurs
       * in both directions in the same context and command stream. */
      switch (cycle % 3u) {
      case 0:
         draw_border(&programs, &objects, &run);
         draw_client(&programs, &objects, &run, cycle);
         draw_glyph(&programs, &objects, &run, cycle);
         break;
      case 1:
         draw_glyph(&programs, &objects, &run, cycle);
         draw_border(&programs, &objects, &run);
         draw_client(&programs, &objects, &run, cycle);
         break;
      default:
         draw_client(&programs, &objects, &run, cycle);
         draw_glyph(&programs, &objects, &run, cycle);
         draw_border(&programs, &objects, &run);
         break;
      }

      draw_dirty_state(&programs, &objects, &run, cycle);

      /* Orphan/refill after the draw; the next cycle must use new storage. */
      glBindBuffer(GL_ARRAY_BUFFER, objects.vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(full_vertices), NULL,
                   GL_DYNAMIC_DRAW);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(full_vertices), full_vertices);
   }

   /* End in a canonical scene, still without a readback or finish between
    * passes. This is the exact retained image validated below. */
   const unsigned final_generation = CYCLES - 1u;
   draw_border(&programs, &objects, &run);
   draw_client(&programs, &objects, &run, final_generation);
   draw_glyph(&programs, &objects, &run, final_generation);
   glFinish();

   printf("F48 transitions: cycles=%u visible=%u dirty=%u total-draws=%u\n",
          CYCLES, CYCLES * 3u + 3u, CYCLES * 5u, run.draws);
   bool ok = gl_ok("mixed sequence");
   if (!noop) {
      ok &= validate_pixels(final_generation);
      if (ok)
         printf("F48 %s exact: rounded border + R8 glyph masks + BGRA "
                "client after mixed state sequence\n",
                run.api_barrier ? "api-barrier" : "tight");
   } else if (ok) {
      printf("F48 %s no-op G17P submission validated; pixels skipped\n",
             run.api_barrier ? "api-barrier" : "tight");
   }

   glDeleteFramebuffers(1, &fbo);
   glDeleteRenderbuffers(1, &depth_stencil);
   glDeleteTextures(1, &colour);
   glDeleteTextures(1, &objects.glyph_texture);
   glDeleteTextures(1, &objects.client_texture);
   glDeleteBuffers(1, &objects.ibo);
   glDeleteBuffers(1, &objects.vbo);
   glDeleteVertexArrays(1, &objects.vao);
   glDeleteProgram(programs.dirty);
   glDeleteProgram(programs.glyph);
   glDeleteProgram(programs.client);
   glDeleteProgram(programs.border);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglTerminate(display);
   return ok ? 0 : 1;
}
