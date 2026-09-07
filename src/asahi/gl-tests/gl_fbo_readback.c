/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A discriminating GL fill/readback oracle for Apple A18 Pro / G17P.
 *
 * gl_fullscreen_fill.c established that a full-surface clear and a
 * full-surface draw into an FBO come back wrong through glReadPixels on G17P
 * while the same test passes on llvmpipe. That result alone cannot say which
 * half is broken: an image that was never rendered and an image that was
 * rendered but copied back wrongly both read as zeros.
 *
 * Every case below shares one rendered image and varies exactly one thing, so
 * a single run separates the possibilities instead of confirming the symptom:
 *
 *   - render target kind: FBO texture attachment, FBO renderbuffer
 *     attachment, or a real window surface backed by a gbm_surface. This
 *     decides whether the physical panel is affected or only the readback,
 *     and whether the defect follows the texture-attachment path.
 *
 *   - readback size, against unchanged image contents: one full-surface
 *     glReadPixels, the same surface read as N horizontal strips, and a
 *     handful of tiny (1x1 and 16x16) reads at fixed points. The image is
 *     identical in all three, so a tiny read that is correct while the
 *     full-surface read is not proves the rendering landed and indicts the
 *     size of a single transfer. The converse indicts rendering.
 *
 *   - repetition, against unchanged image contents: the full-surface read is
 *     repeated immediately and again after a delay. A bad count that improves
 *     on re-read is a visibility race (completion signalled before the GPU's
 *     stores are observable); a bad count that is bit-identical every time is
 *     not a race and no amount of waiting will fix it.
 *
 * Nothing here disables asynchrony, sleeps to make a case pass, or skips an
 * operation: the delayed re-read is reported separately and is never allowed
 * to turn a failure into a pass.
 *
 * Build (against an installed GL stack):
 *   cc -O2 -o gl_fbo_readback gl_fbo_readback.c -lEGL -lGLESv2 -lgbm
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <fcntl.h>
#include <gbm.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* 0.2, 0.6, 0.9 and 1.0 in UNORM8. Exact under any correct conversion. */
#define DRAW_R 51
#define DRAW_G 153
#define DRAW_B 230
#define CLEAR_R 255
#define CLEAR_G 0
#define CLEAR_B 0

static int failures;
static int width = 2408, height = 1506;

static const char *vs_src = "attribute vec2 pos;\n"
                            "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *fs_src = "precision highp float;\n"
                            "void main() { gl_FragColor = vec4(0.2, 0.6, 0.9, 1.0); }\n";

static bool
check_gl(const char *what)
{
   GLenum e;
   bool ok = true;
   while ((e = glGetError()) != GL_NO_ERROR) {
      printf("  GL ERROR 0x%04x during %s\n", e, what);
      ok = false;
   }
   return ok;
}

static GLuint
compile(GLenum stage, const char *src)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof log, NULL, log);
      printf("shader compile failed: %s\n", log);
      exit(2);
   }
   return s;
}

/* Count pixels in `px` (w*h RGBA8) that are not exactly (r,g,b,255). */
static long
count_bad(const unsigned char *px, long w, long h, unsigned char r,
          unsigned char g, unsigned char b, long *first_bad)
{
   long bad = 0;
   *first_bad = -1;
   for (long i = 0; i < w * h; i++) {
      if (px[i * 4 + 0] != r || px[i * 4 + 1] != g || px[i * 4 + 2] != b ||
          px[i * 4 + 3] != 255) {
         if (*first_bad < 0)
            *first_bad = i;
         bad++;
      }
   }
   return bad;
}

static void
report(const char *case_name, const char *phase, long total, long bad,
       long first_bad, const unsigned char *px)
{
   printf("  %-28s %-22s total=%-9ld bad=%-9ld", case_name, phase, total, bad);
   if (first_bad >= 0)
      printf(" first_bad=(%ld,%ld) got=%02x%02x%02x%02x", first_bad % width,
             first_bad / width, px[first_bad * 4 + 0], px[first_bad * 4 + 1],
             px[first_bad * 4 + 2], px[first_bad * 4 + 3]);
   printf("\n");
   if (bad)
      failures++;
}

static void
sleep_ms(long ms)
{
   struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
   nanosleep(&ts, NULL);
}

/*
 * Read the same unchanged surface three ways and report each independently.
 * `expect_*` is the colour every pixel must have.
 */
static void
probe_readbacks(const char *case_name, unsigned char expect_r,
                unsigned char expect_g, unsigned char expect_b)
{
   const long total = (long)width * height;
   unsigned char *px = malloc((size_t)total * 4);
   unsigned char *px2 = malloc((size_t)total * 4);
   if (!px || !px2) {
      printf("  out of memory\n");
      exit(2);
   }

   glPixelStorei(GL_PACK_ALIGNMENT, 1);

   /* 1. One full-surface read. */
   memset(px, 0xa5, (size_t)total * 4);
   glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px);
   check_gl("full readback");
   long first_bad;
   long bad_full = count_bad(px, width, height, expect_r, expect_g, expect_b,
                             &first_bad);
   report(case_name, "read:full", total, bad_full, first_bad, px);

   /* 2. Immediately repeat it. Contents cannot have changed, so a different
    * answer is a visibility problem and an identical answer is not. */
   memset(px2, 0xa5, (size_t)total * 4);
   glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px2);
   check_gl("repeat readback");
   long bad_repeat =
      count_bad(px2, width, height, expect_r, expect_g, expect_b, &first_bad);
   printf("  %-28s %-22s total=%-9ld bad=%-9ld identical_to_first=%s\n",
          case_name, "read:full-again", total, bad_repeat,
          memcmp(px, px2, (size_t)total * 4) == 0 ? "yes" : "NO");

   /* 3. And once more after a delay, still without touching the surface. */
   const char *delay_env = getenv("NEO_READBACK_DELAY_MS");
   long delay_ms = delay_env ? atol(delay_env) : 2000;
   sleep_ms(delay_ms);
   memset(px2, 0xa5, (size_t)total * 4);
   glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px2);
   check_gl("delayed readback");
   long bad_delayed =
      count_bad(px2, width, height, expect_r, expect_g, expect_b, &first_bad);
   printf("  %-28s read:full-after-%-8ldtotal=%-9ld bad=%-9ld "
          "identical_to_first=%s\n",
          case_name, delay_ms, total, bad_delayed,
          memcmp(px, px2, (size_t)total * 4) == 0 ? "yes" : "NO");

   /* 4. The same surface as horizontal strips. Each transfer is 1/16 the
    * size; the pixels are the same pixels. */
   const int strips = 16;
   memset(px2, 0xa5, (size_t)total * 4);
   for (int s = 0; s < strips; s++) {
      int y0 = (int)((long)height * s / strips);
      int y1 = (int)((long)height * (s + 1) / strips);
      if (y1 <= y0)
         continue;
      glReadPixels(0, y0, width, y1 - y0, GL_RGBA, GL_UNSIGNED_BYTE,
                   px2 + (size_t)y0 * width * 4);
   }
   check_gl("strip readback");
   long bad_strips =
      count_bad(px2, width, height, expect_r, expect_g, expect_b, &first_bad);
   report(case_name, "read:16-strips", total, bad_strips, first_bad, px2);

   /* 5. Tiny reads at fixed points. Four bytes each: too small for any
    * plausible per-transfer size effect to hide the answer. */
   const struct {
      const char *name;
      int x, y;
   } points[] = {
      {"px(0,0)", 0, 0},
      {"px(mid)", width / 2, height / 2},
      {"px(last)", width - 1, height - 1},
      {"px(0,last)", 0, height - 1},
      {"px(last,0)", width - 1, 0},
   };
   for (unsigned i = 0; i < sizeof points / sizeof points[0]; i++) {
      unsigned char one[4] = {0xa5, 0xa5, 0xa5, 0xa5};
      glReadPixels(points[i].x, points[i].y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                   one);
      bool ok = one[0] == expect_r && one[1] == expect_g &&
                one[2] == expect_b && one[3] == 255;
      printf("  %-28s %-22s at (%d,%d) got=%02x%02x%02x%02x %s\n", case_name,
             points[i].name, points[i].x, points[i].y, one[0], one[1], one[2],
             one[3], ok ? "ok" : "BAD");
      if (!ok)
         failures++;
   }
   check_gl("point readback");

   /* 6. A 16x16 patch at the origin: still 1 KiB, but enough to show whether
    * a small rectangle is exact while the whole surface is not. */
   {
      unsigned char patch[16 * 16 * 4];
      memset(patch, 0xa5, sizeof patch);
      int pw = width < 16 ? width : 16;
      int ph = height < 16 ? height : 16;
      glReadPixels(0, 0, pw, ph, GL_RGBA, GL_UNSIGNED_BYTE, patch);
      long fb;
      long bad = count_bad(patch, pw, ph, expect_r, expect_g, expect_b, &fb);
      printf("  %-28s %-22s total=%-9d bad=%-9ld\n", case_name,
             "read:16x16-origin", pw * ph, bad);
      if (bad)
         failures++;
   }
   check_gl("patch readback");

   free(px);
   free(px2);
}

static void
draw_full_surface(void)
{
   static GLuint prog, vbo;
   if (!prog) {
      prog = glCreateProgram();
      glAttachShader(prog, compile(GL_VERTEX_SHADER, vs_src));
      glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fs_src));
      glBindAttribLocation(prog, 0, "pos");
      glLinkProgram(prog);
      GLint linked = 0;
      glGetProgramiv(prog, GL_LINK_STATUS, &linked);
      if (!linked) {
         char log[4096];
         glGetProgramInfoLog(prog, sizeof log, NULL, log);
         printf("link failed: %s\n", log);
         exit(2);
      }
      static const GLfloat verts[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
      glGenBuffers(1, &vbo);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
   }
   glUseProgram(prog);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   GLint loc = glGetAttribLocation(prog, "pos");
   glEnableVertexAttribArray(loc);
   glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* Run clear-then-probe and draw-then-probe against whatever framebuffer is
 * currently bound. */
static void
run_case(const char *case_name)
{
   printf("case %s (%dx%d)\n", case_name, width, height);
   glViewport(0, 0, width, height);

   glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();
   check_gl("clear");
   {
      char name[64];
      snprintf(name, sizeof name, "%s/clear", case_name);
      probe_readbacks(name, CLEAR_R, CLEAR_G, CLEAR_B);
   }

   draw_full_surface();
   glFinish();
   check_gl("draw");
   {
      char name[64];
      snprintf(name, sizeof name, "%s/draw", case_name);
      probe_readbacks(name, DRAW_R, DRAW_G, DRAW_B);
   }
}


/* ------------------------------------------------------------------ *
 * Draw-shape sweep.
 *
 * The cases above prove that ONE full-surface triangle is exact.  That is
 * not the same claim as "a full-surface quad is exact": a quad reaches the
 * hardware as two primitives, through an index list or a strip/fan topology,
 * and every one of those is a separate encoder path.  The G17P desktop shows
 * a background quad rendered as a single triangle, so the question a test has
 * to answer is precisely which shape loses its second primitive.
 *
 * Every case below paints the identical full-surface quad, differing only in
 * how that quad is expressed.  The clear underneath is red, so a missing
 * primitive shows up as a red half rather than as "some pixels are wrong",
 * and the two probe points name which half is missing without a screenshot:
 *
 *   lower-left  (W/8, H/8)   is covered only by the quad's first triangle
 *   upper-right (7W/8, 7H/8) is covered only by the quad's second triangle
 *
 * under the TL,BL,BR,TR corner order Cogl uses for a rectangle, which is the
 * order the GNOME background is drawn with.
 *
 * The shape list is not a guess at what might break.  Cogl flushes a journal
 * batch of exactly one rectangle as
 *
 *     _cogl_framebuffer_draw_attributes(fb, pipeline,
 *                                       COGL_VERTICES_MODE_TRIANGLE_FAN,
 *                                       state->current_vertex, 4, ...)
 *
 * (mutter 50.4, cogl/cogl/cogl-journal.c) and a batch of two or more as
 * indexed COGL_VERTICES_MODE_TRIANGLES over the shared rectangle index buffer,
 * whose indices are UNSIGNED_BYTE up to 64 rectangles and UNSIGNED_SHORT above
 * that (cogl/cogl/cogl-indices.c).  Those three shapes -- a bare four-vertex
 * fan at a non-zero `first`, a U8 indexed triangle list and a U16 one -- are
 * therefore what a GNOME desktop actually issues, and all three are below.
 * ------------------------------------------------------------------ */

/* Quad corners in Cogl's rectangle order: TL, BL, BR, TR. */
static const GLfloat quad_corners[8] = {
   -1.0f, 1.0f,   /* 0 TL */
   -1.0f, -1.0f,  /* 1 BL */
   1.0f,  -1.0f,  /* 2 BR */
   1.0f,  1.0f,   /* 3 TR */
};

/* The same quad as an unindexed triangle list: (TL,BL,BR) then (TL,BR,TR). */
static const GLfloat quad_list6[12] = {
   -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
   -1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 1.0f,
};

/* Strip order TL, BL, TR, BR: (TL,BL,TR) then (BL,TR,BR). */
static const GLfloat quad_strip4[8] = {
   -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
};

/* Cogl draws a lone rectangle as glDrawArrays(GL_TRIANGLE_FAN, first, 4) with
 * `first` a multiple of four into a shared journal vertex buffer, so the fan
 * and strip cases are also run at a non-zero `first`. */
static const GLfloat quad_corners_at4[16] = {
   -0.1f, -0.1f, -0.2f, -0.2f, -0.3f, -0.3f, -0.4f, -0.4f,
   -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,
};
static const GLfloat quad_strip4_at4[16] = {
   -0.1f, -0.1f, -0.2f, -0.2f, -0.3f, -0.3f, -0.4f, -0.4f,
   -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f,
};

/* quad_list6 preceded by two vertices that must never be drawn, for the
 * glDrawArrays(first=2) case: `first` is a separate hardware field. */
static const GLfloat quad_list6_at2[16] = {
   -0.1f, -0.1f, -0.2f, -0.2f,
   -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f, -1.0f,
   -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f, 1.0f,
};

/* One oversized triangle covering the surface: the case already known exact. */
static const GLfloat big_triangle[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};

static const GLubyte idx_list6_u8[6] = {0, 1, 2, 0, 2, 3};
static const GLushort idx_list6_u16[6] = {0, 1, 2, 0, 2, 3};
static const GLuint idx_list6_u32[6] = {0, 1, 2, 0, 2, 3};
static const GLushort idx_strip4_u16[4] = {0, 1, 3, 2};
static const GLushort idx_fan4_u16[4] = {0, 1, 2, 3};
/* Two leading indices that must be skipped, for the firstIndex case. */
static const GLushort idx_list6_u16_at2[8] = {3, 3, 0, 1, 2, 0, 2, 3};

struct shape_case {
   const char *name;
   GLenum mode;
   const GLfloat *verts; /* NDC xy pairs */
   unsigned nverts;      /* vertices in the array */
   unsigned first;       /* glDrawArrays first, or unused when indexed */
   unsigned count;       /* vertices or indices to draw */
   GLenum index_type;    /* 0 for glDrawArrays */
   const void *indices;
   unsigned nindices;     /* indices in the array */
   unsigned first_index;  /* offset into the index array, in elements */
   bool client_arrays;    /* pass pointers instead of buffer objects */
};

static GLuint shape_prog;

static GLuint
build_program(const char *fs)
{
   GLuint p = glCreateProgram();
   glAttachShader(p, compile(GL_VERTEX_SHADER, vs_src));
   glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
   glBindAttribLocation(p, 0, "pos");
   glLinkProgram(p);
   GLint linked = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof log, NULL, log);
      printf("link failed: %s\n", log);
      exit(2);
   }
   return p;
}

static unsigned
index_type_size(GLenum type)
{
   switch (type) {
   case GL_UNSIGNED_BYTE:
      return 1;
   case GL_UNSIGNED_SHORT:
      return 2;
   default:
      return 4;
   }
}

/*
 * Read the whole surface once and describe what is wrong geometrically, not
 * just how many pixels are wrong.  `alt_*` is the colour the surface was
 * cleared to, so an untouched pixel is reported as "clear" rather than as an
 * unexplained value.
 */
static long
check_surface(const char *case_name, unsigned char r, unsigned char g,
              unsigned char b, unsigned char ar, unsigned char ag,
              unsigned char ab)
{
   const long total = (long)width * height;
   unsigned char *px = malloc((size_t)total * 4);
   if (!px) {
      printf("  out of memory\n");
      exit(2);
   }
   memset(px, 0xa5, (size_t)total * 4);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, px);
   check_gl("shape readback");

   long bad = 0, clear_left = 0;
   long minx = width, maxx = -1, miny = height, maxy = -1;
   long first_bad = -1;
   for (long y = 0; y < height; y++) {
      for (long x = 0; x < width; x++) {
         const unsigned char *p = px + ((y * width) + x) * 4;
         if (p[0] == r && p[1] == g && p[2] == b && p[3] == 255)
            continue;
         bad++;
         if (p[0] == ar && p[1] == ag && p[2] == ab)
            clear_left++;
         if (first_bad < 0)
            first_bad = y * width + x;
         if (x < minx)
            minx = x;
         if (x > maxx)
            maxx = x;
         if (y < miny)
            miny = y;
         if (y > maxy)
            maxy = y;
      }
   }

   /* The two points that name which triangle of the quad is missing. */
   const struct {
      const char *name;
      long x, y;
   } probes[] = {
      {"lower-left", width / 8, height / 8},
      {"upper-right", (long)width * 7 / 8, (long)height * 7 / 8},
      {"centre", width / 2, height / 2},
   };

   printf("  %-26s bad=%-9ld still-clear=%-9ld", case_name, bad, clear_left);
   if (bad)
      printf(" bbox=(%ld,%ld)-(%ld,%ld) first=(%ld,%ld) got=%02x%02x%02x%02x",
             minx, miny, maxx, maxy, first_bad % width, first_bad / width,
             px[first_bad * 4 + 0], px[first_bad * 4 + 1],
             px[first_bad * 4 + 2], px[first_bad * 4 + 3]);
   printf("\n");

   printf("  %-26s", "");
   for (unsigned i = 0; i < sizeof probes / sizeof probes[0]; i++) {
      const unsigned char *p = px + ((probes[i].y * width) + probes[i].x) * 4;
      printf(" %s=%02x%02x%02x%s", probes[i].name, p[0], p[1], p[2],
             (p[0] == r && p[1] == g && p[2] == b) ? "(ok)" : "(BAD)");
   }
   printf("\n");

   /* A coarse map, printed top row first so it reads like the screen. */
   if (bad) {
      const int mh = 9, mw = 25;
      for (int my = mh - 1; my >= 0; my--) {
         long y = (long)height * (2 * my + 1) / (2 * mh);
         printf("  %-26s |", my == mh - 1 ? "map(o=drawn c=clear)" : "");
         for (int mx = 0; mx < mw; mx++) {
            long x = (long)width * (2 * mx + 1) / (2 * mw);
            const unsigned char *p = px + ((y * width) + x) * 4;
            char ch = '?';
            if (p[0] == r && p[1] == g && p[2] == b)
               ch = 'o';
            else if (p[0] == ar && p[1] == ag && p[2] == ab)
               ch = 'c';
            putchar(ch);
         }
         printf("|\n");
      }
   }

   free(px);
   if (bad)
      failures++;
   return bad;
}

static void
run_shape_sweep(const char *surface_name)
{
   static const struct shape_case cases[] = {
      {"tri-single-3", GL_TRIANGLES, big_triangle, 3, 0, 3, 0, NULL, 0, 0,
       false},
      {"tri-list-6", GL_TRIANGLES, quad_list6, 6, 0, 6, 0, NULL, 0, 0, false},
      {"tri-list-6-first2", GL_TRIANGLES, quad_list6_at2, 8, 2, 6, 0, NULL, 0,
       0, false},
      {"tri-strip-4", GL_TRIANGLE_STRIP, quad_strip4, 4, 0, 4, 0, NULL, 0, 0,
       false},
      {"tri-fan-4", GL_TRIANGLE_FAN, quad_corners, 4, 0, 4, 0, NULL, 0, 0,
       false},
      {"tri-fan-4-first4", GL_TRIANGLE_FAN, quad_corners_at4, 8, 4, 4, 0, NULL,
       0, 0, false},
      {"tri-strip-4-first4", GL_TRIANGLE_STRIP, quad_strip4_at4, 8, 4, 4, 0,
       NULL, 0, 0, false},
      {"tri-fan-4-client", GL_TRIANGLE_FAN, quad_corners, 4, 0, 4, 0, NULL, 0,
       0, true},
      {"idx-u8-tri-list-6", GL_TRIANGLES, quad_corners, 4, 0, 6,
       GL_UNSIGNED_BYTE, idx_list6_u8, 6, 0, false},
      {"idx-u16-tri-list-6", GL_TRIANGLES, quad_corners, 4, 0, 6,
       GL_UNSIGNED_SHORT, idx_list6_u16, 6, 0, false},
      {"idx-u32-tri-list-6", GL_TRIANGLES, quad_corners, 4, 0, 6,
       GL_UNSIGNED_INT, idx_list6_u32, 6, 0, false},
      {"idx-u16-tri-list-6-at2", GL_TRIANGLES, quad_corners, 4, 0, 6,
       GL_UNSIGNED_SHORT, idx_list6_u16_at2, 8, 2, false},
      {"idx-u16-tri-strip-4", GL_TRIANGLE_STRIP, quad_corners, 4, 0, 4,
       GL_UNSIGNED_SHORT, idx_strip4_u16, 4, 0, false},
      {"idx-u16-tri-fan-4", GL_TRIANGLE_FAN, quad_corners, 4, 0, 4,
       GL_UNSIGNED_SHORT, idx_fan4_u16, 4, 0, false},
      {"idx-u16-tri-list-6-client", GL_TRIANGLES, quad_corners, 4, 0, 6,
       GL_UNSIGNED_SHORT, idx_list6_u16, 6, 0, true},
      {"tri-list-6-client", GL_TRIANGLES, quad_list6, 6, 0, 6, 0, NULL, 0, 0,
       true},
   };

   const char *exts = (const char *)glGetString(GL_EXTENSIONS);
   const bool have_u32 = exts && strstr(exts, "GL_OES_element_index_uint");
   const char *broken[32];
   int nbroken = 0;

   printf("case %s draw-shapes (%dx%d)\n", surface_name, width, height);
   if (!shape_prog)
      shape_prog = build_program(fs_src);
   glUseProgram(shape_prog);
   glViewport(0, 0, width, height);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);
   const GLint loc = glGetAttribLocation(shape_prog, "pos");

   for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
      const struct shape_case *c = &cases[i];
      if (c->index_type == GL_UNSIGNED_INT && !have_u32) {
         printf("  %-26s skipped (no GL_OES_element_index_uint)\n", c->name);
         continue;
      }

      glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      GLuint vbo = 0, ibo = 0;
      glEnableVertexAttribArray(loc);
      if (c->client_arrays) {
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, c->verts);
      } else {
         glGenBuffers(1, &vbo);
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)c->nverts * 2 * sizeof(GLfloat),
                      c->verts, GL_STATIC_DRAW);
         glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
      }

      if (c->index_type == 0) {
         glDrawArrays(c->mode, (GLint)c->first, (GLsizei)c->count);
      } else {
         const unsigned esz = index_type_size(c->index_type);
         const size_t byte_off = (size_t)c->first_index * esz;
         if (c->client_arrays) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glDrawElements(c->mode, (GLsizei)c->count, c->index_type,
                           (const char *)c->indices + byte_off);
         } else {
            glGenBuffers(1, &ibo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         (GLsizeiptr)c->nindices * esz, c->indices,
                         GL_STATIC_DRAW);
            glDrawElements(c->mode, (GLsizei)c->count, c->index_type,
                           (const void *)byte_off);
         }
      }
      glFinish();
      check_gl(c->name);

      char full[96];
      snprintf(full, sizeof full, "%s/%s", surface_name, c->name);
      if (check_surface(full, DRAW_R, DRAW_G, DRAW_B, CLEAR_R, CLEAR_G,
                        CLEAR_B)) {
         if (nbroken < (int)(sizeof broken / sizeof broken[0]))
            broken[nbroken++] = c->name;
      }

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      if (vbo)
         glDeleteBuffers(1, &vbo);
      if (ibo)
         glDeleteBuffers(1, &ibo);
   }

   printf("  %-26s ", "SHAPES BROKEN:");
   if (!nbroken) {
      printf("none\n");
   } else {
      for (int i = 0; i < nbroken; i++)
         printf("%s%s", i ? ", " : "", broken[i]);
      printf("\n");
   }
}

/* ------------------------------------------------------------------ *
 * Fragment-output precision sweep.
 *
 * gl_FragColor = vec4(0.2, 0.6, 0.9, 1.0) reads back with blue 229 on G17P
 * where 230 is required.  0.9 is the only channel of that shader whose UNORM8
 * image is a tie (0.9f * 255.0f is exactly 229.5f in binary32), so on its own
 * it cannot say whether the tie was rounded the wrong way or whether the whole
 * value was computed at half precision - fp16(0.9) * 255 is 229.47, which
 * lands on 229 under any rounding.
 *
 * Each value below is scored against four models, all computed here rather
 * than hardcoded:
 *
 *   f32-rne    round-to-nearest-even of f32(v) * 255.0f  <- the correct answer
 *   f32-trunc  truncation of the same product
 *   f16-rne    the same, after v has passed through binary16
 *   f16-trunc  both defects at once
 *
 * The values are chosen so that no single model explains all of them unless it
 * is the real one: 0.5 and 0.39509804 separate rounding from truncation while
 * binary16 changes nothing, and 0.7 and 0.3 separate binary16 from binary32
 * while rounding and truncation agree.
 *
 * Each value is drawn twice, once from a literal in the shader and once from a
 * highp uniform, because a literal that is folded to half precision and a
 * fragment pipeline that runs at half precision are different bugs with
 * different fixes.
 * ------------------------------------------------------------------ */

static uint16_t
f32_to_f16_bits(float f)
{
   uint32_t x;
   memcpy(&x, &f, sizeof x);
   const uint32_t sign = (x >> 16) & 0x8000u;
   const int32_t e = (int32_t)((x >> 23) & 0xffu);
   const uint32_t m = x & 0x7fffffu;

   if (e == 255)
      return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));

   int32_t he = e - 127 + 15;
   if (he >= 31)
      return (uint16_t)(sign | 0x7c00u);

   if (he <= 0) {
      if (he < -10)
         return (uint16_t)sign;
      const uint32_t mant = m | 0x800000u;
      const unsigned shift = (unsigned)(14 - he);
      const uint32_t round_bit = 1u << (shift - 1);
      const uint32_t sticky = mant & (round_bit - 1);
      uint32_t out = mant >> shift;
      if ((mant & round_bit) && (sticky || (out & 1)))
         out++;
      /* A carry out of the mantissa lands in the exponent field, which is
       * exactly the encoding of the smallest normal. */
      return (uint16_t)(sign | out);
   }

   const uint32_t round_bit = 1u << 12;
   const uint32_t sticky = m & (round_bit - 1);
   uint32_t out = m >> 13;
   if ((m & round_bit) && (sticky || (out & 1))) {
      out++;
      if (out == 0x400u) {
         out = 0;
         he++;
         if (he >= 31)
            return (uint16_t)(sign | 0x7c00u);
      }
   }
   return (uint16_t)(sign | ((uint32_t)he << 10) | out);
}

static float
f16_bits_to_f32(uint16_t h)
{
   const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
   const uint32_t exp = (h >> 10) & 0x1fu;
   const uint32_t man = h & 0x3ffu;
   uint32_t out;
   if (exp == 0) {
      if (man == 0) {
         out = sign;
      } else {
         int e = -1;
         uint32_t m = man;
         do {
            e++;
            m <<= 1;
         } while (!(m & 0x400u));
         out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
      }
   } else if (exp == 31) {
      out = sign | 0x7f800000u | (man << 13);
   } else {
      out = sign | ((exp + 127 - 15) << 23) | (man << 13);
   }
   float f;
   memcpy(&f, &out, sizeof f);
   return f;
}

/* Round half to even, matching what a correct float->UNORM8 must do. */
static unsigned
unorm8_rne(float scaled)
{
   float fl = floorf(scaled);
   float diff = scaled - fl;
   unsigned i = (unsigned)fl;
   if (diff > 0.5f || (diff == 0.5f && (i & 1)))
      i++;
   return i > 255 ? 255 : i;
}

static unsigned
unorm8_trunc(float scaled)
{
   float fl = floorf(scaled);
   unsigned i = (unsigned)fl;
   return i > 255 ? 255 : i;
}

struct prec_model {
   unsigned f32_rne, f32_trunc, f16_rne, f16_trunc;
};

static struct prec_model
prec_models(const char *literal)
{
   const float v = (float)atof(literal);
   const float vh = f16_bits_to_f32(f32_to_f16_bits(v));
   struct prec_model m;
   m.f32_rne = unorm8_rne(v * 255.0f);
   m.f32_trunc = unorm8_trunc(v * 255.0f);
   m.f16_rne = unorm8_rne(vh * 255.0f);
   m.f16_trunc = unorm8_trunc(vh * 255.0f);
   return m;
}

static void
run_precision_sweep(const char *surface_name)
{
   /* Every value is written to all three colour channels, so a channel-local
    * defect is visible as disagreement between them. */
   static const char *const values[] = {
      "0.9", "0.7", "0.3", "0.5", "0.39509804", "0.60294118", "0.33333334",
   };

   printf("case %s frag-precision (%dx%d)\n", surface_name, width, height);
   glViewport(0, 0, width, height);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);

   GLuint vbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof big_triangle, big_triangle,
                GL_STATIC_DRAW);

   for (unsigned i = 0; i < sizeof values / sizeof values[0]; i++) {
      const char *v = values[i];
      const struct prec_model m = prec_models(v);

      for (int uniform = 0; uniform < 2; uniform++) {
         char fs[512];
         if (uniform) {
            snprintf(fs, sizeof fs,
                     "precision highp float;\n"
                     "uniform highp vec4 ucolour;\n"
                     "void main() { gl_FragColor = ucolour; }\n");
         } else {
            snprintf(fs, sizeof fs,
                     "precision highp float;\n"
                     "void main() { gl_FragColor = vec4(%s, %s, %s, 1.0); }\n",
                     v, v, v);
         }

         GLuint prog = build_program(fs);
         glUseProgram(prog);
         if (uniform) {
            const float f = (float)atof(v);
            GLint ul = glGetUniformLocation(prog, "ucolour");
            glUniform4f(ul, f, f, f, 1.0f);
         }

         glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);
         glBindBuffer(GL_ARRAY_BUFFER, vbo);
         GLint loc = glGetAttribLocation(prog, "pos");
         glEnableVertexAttribArray(loc);
         glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glFinish();
         check_gl("precision draw");

         unsigned char centre[4] = {0xa5, 0xa5, 0xa5, 0xa5};
         unsigned char corner[4] = {0xa5, 0xa5, 0xa5, 0xa5};
         glPixelStorei(GL_PACK_ALIGNMENT, 1);
         glReadPixels(width / 2, height / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                      centre);
         glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, corner);
         check_gl("precision readback");

         const bool uniform_channels =
            centre[0] == centre[1] && centre[1] == centre[2];
         const unsigned got = centre[2];
         char verdict[128];
         verdict[0] = '\0';
         if (got == m.f32_rne)
            strcat(verdict, "f32-rne ");
         if (got == m.f32_trunc)
            strcat(verdict, "f32-trunc ");
         if (got == m.f16_rne)
            strcat(verdict, "f16-rne ");
         if (got == m.f16_trunc)
            strcat(verdict, "f16-trunc ");
         if (!verdict[0])
            strcat(verdict, "NO-MODEL");

         printf("  %-14s %-8s got=%02x%02x%02x corner=%02x%02x%02x "
                "f32-rne=%3u f32-trunc=%3u f16-rne=%3u f16-trunc=%3u  "
                "matches: %s%s\n",
                v, uniform ? "uniform" : "literal", centre[0], centre[1],
                centre[2], corner[0], corner[1], corner[2], m.f32_rne,
                m.f32_trunc, m.f16_rne, m.f16_trunc, verdict,
                uniform_channels ? "" : " [CHANNELS DIFFER]");

         /* Only the correct model counts as a pass. */
         if (got != m.f32_rne || !uniform_channels ||
             centre[0] != corner[0] || centre[1] != corner[1] ||
             centre[2] != corner[2])
            failures++;

         glDeleteProgram(prog);
      }
   }

   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glDeleteBuffers(1, &vbo);
}


/* ------------------------------------------------------------------ *
 * UNORM8 ramp: every one of the 256 representable values, twice.
 *
 * The draw cases above all report one colour, so a wrong byte in them cannot
 * say where the byte was lost.  These two sub-cases separate the two halves of
 * that path and cover the whole value range instead of one colour:
 *
 *   upload   glTexImage2D a ramp into an RGBA8 texture and glReadPixels it
 *            straight back.  No rendering at all, so any value that changes
 *            was changed by the upload or the readback.
 *
 *   render   draw the same ramp with a shader whose output for column x is
 *            exactly x/255.  A correct float->UNORM8 conversion returns every
 *            x; truncation loses roughly half of them, and the report says
 *            which model the returned bytes agree with.
 *
 * Both run in their own 256x4 framebuffer so the whole range is covered
 * whatever size the rest of the run uses.
 * ------------------------------------------------------------------ */

static const char *ramp_fs_src =
   "precision highp float;\n"
   "void main() {\n"
   "   float v = floor(gl_FragCoord.x) / 255.0;\n"
   "   gl_FragColor = vec4(v, v, v, 1.0);\n"
   "}\n";

static void
run_unorm_ramp(void)
{
   const int rw = 256, rh = 4;
   unsigned char *src = malloc((size_t)rw * rh * 4);
   unsigned char *got = malloc((size_t)rw * rh * 4);
   if (!src || !got) {
      printf("  out of memory\n");
      exit(2);
   }
   for (int y = 0; y < rh; y++) {
      for (int x = 0; x < rw; x++) {
         unsigned char *p = src + ((y * rw) + x) * 4;
         p[0] = (unsigned char)x;
         p[1] = (unsigned char)(255 - x);
         p[2] = (unsigned char)((x * 7) & 0xff);
         p[3] = 255;
      }
   }

   GLuint tex, fbo;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rw, rh, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, src);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          tex, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      printf("case unorm-ramp: incomplete FBO, skipped\n");
      goto out;
   }

   printf("case unorm-ramp (256x4)\n");

   /* 1. Upload and read straight back. */
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   memset(got, 0xa5, (size_t)rw * rh * 4);
   glFinish();
   glReadPixels(0, 0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, got);
   check_gl("ramp upload readback");
   {
      long bad = 0;
      int first = -1;
      for (long i = 0; i < (long)rw * rh * 4; i++) {
         if (got[i] != src[i]) {
            if (first < 0)
               first = (int)i;
            bad++;
         }
      }
      printf("  %-26s bytes_wrong=%-6ld", "upload-readback", bad);
      if (first >= 0)
         printf(" first at px(%d,%d) ch%d: sent=%u got=%u", (first / 4) % rw,
                (first / 4) / rw, first % 4, src[first], got[first]);
      printf("\n");
      if (bad)
         failures++;
   }

   /* 2. Render the same ramp and score the result against both models. */
   {
      GLuint prog = build_program(ramp_fs_src);
      glUseProgram(prog);
      glViewport(0, 0, rw, rh);
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_BLEND);
      glDisable(GL_SCISSOR_TEST);
      glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      GLuint vbo;
      glGenBuffers(1, &vbo);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof big_triangle, big_triangle,
                   GL_STATIC_DRAW);
      GLint loc = glGetAttribLocation(prog, "pos");
      glEnableVertexAttribArray(loc);
      glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
      check_gl("ramp draw");

      memset(got, 0xa5, (size_t)rw * rh * 4);
      glReadPixels(0, 0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, got);
      check_gl("ramp draw readback");

      long n_rne = 0, n_trunc = 0, n_other = 0;
      int first_wrong = -1;
      for (int x = 0; x < rw; x++) {
         const float v = (float)x / 255.0f;
         const unsigned rne = unorm8_rne(v * 255.0f);
         const unsigned tr = unorm8_trunc(v * 255.0f);
         const unsigned g = got[x * 4 + 0];
         if (g == rne)
            n_rne++;
         else if (g == tr)
            n_trunc++;
         else
            n_other++;
         if (g != rne && first_wrong < 0)
            first_wrong = x;
      }
      printf("  %-26s columns=%d  match_f32_rne=%ld  match_f32_trunc_only=%ld"
             "  match_neither=%ld",
             "render-ramp", rw, n_rne, n_trunc, n_other);
      if (first_wrong >= 0) {
         const float v = (float)first_wrong / 255.0f;
         printf("  first_wrong x=%d expected=%u got=%u", first_wrong,
                unorm8_rne(v * 255.0f), got[first_wrong * 4]);
      }
      printf("\n");
      if (n_rne != rw)
         failures++;

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glDeleteBuffers(1, &vbo);
      glDeleteProgram(prog);
   }

out:
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &tex);
   free(src);
   free(got);
}

int
main(int argc, char **argv)
{
   if (argc > 2) {
      width = atoi(argv[1]);
      height = atoi(argv[2]);
   }
   if (width <= 0 || height <= 0) {
      printf("bad size\n");
      return 2;
   }

   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");

   /* NEO_SURFACELESS is the software control: with no DRM device at all a
    * software rasterizer runs exactly these cases, which is what proves the
    * oracle itself is sound. It has no gbm_device, so the window-surface case
    * reports itself skipped rather than silently disappearing. */
   struct gbm_device *gbm = NULL;
   EGLDisplay dpy = EGL_NO_DISPLAY;
   if (getenv("NEO_SURFACELESS")) {
      if (!get_platform_display) {
         printf("no eglGetPlatformDisplayEXT\n");
         return 2;
      }
      dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
                                 EGL_DEFAULT_DISPLAY, NULL);
   } else {
      const char *node = getenv("NEO_RENDER_NODE");
      int fd = open(node ? node : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
      if (fd < 0) {
         perror("open render node");
         return 2;
      }
      gbm = gbm_create_device(fd);
      if (!gbm) {
         printf("gbm_create_device failed\n");
         return 2;
      }
      dpy = get_platform_display
               ? get_platform_display(EGL_PLATFORM_GBM_KHR, gbm, NULL)
               : eglGetDisplay((EGLNativeDisplayType)gbm);
   }
   EGLint major, minor;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &major, &minor)) {
      printf("no EGL display\n");
      return 2;
   }

   /* A window-surface case needs a real config; the FBO cases do not care. */
   const EGLint cfg_attrs[] = {EGL_SURFACE_TYPE,
                               EGL_WINDOW_BIT,
                               EGL_RENDERABLE_TYPE,
                               EGL_OPENGL_ES2_BIT,
                               EGL_RED_SIZE,
                               8,
                               EGL_GREEN_SIZE,
                               8,
                               EGL_BLUE_SIZE,
                               8,
                               EGL_ALPHA_SIZE,
                               8,
                               EGL_NONE};
   EGLConfig cfg = (EGLConfig)0;
   EGLint n = 0;
   bool have_cfg = eglChooseConfig(dpy, cfg_attrs, &cfg, 1, &n) && n >= 1;

   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 2, EGL_NONE};
   EGLContext ctx = eglCreateContext(dpy, have_cfg ? cfg : (EGLConfig)0,
                                     EGL_NO_CONTEXT, ctx_attrs);
   if (ctx == EGL_NO_CONTEXT) {
      printf("no EGL context: 0x%x\n", eglGetError());
      return 2;
   }
   if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("eglMakeCurrent(surfaceless) failed: 0x%x\n", eglGetError());
      return 2;
   }
   printf("GL_RENDERER=%s\nGL_VERSION=%s\n", glGetString(GL_RENDERER),
          glGetString(GL_VERSION));
   const char *exts = (const char *)glGetString(GL_EXTENSIONS);
   bool have_rgba8 = exts && strstr(exts, "GL_OES_rgb8_rgba8");
   printf("GL_OES_rgb8_rgba8=%s\n", have_rgba8 ? "yes" : "no");

   /* Case 0: no rendering at all. Upload a known constant with
    * glTexImage2D, attach the texture to an FBO and read it straight back.
    * This exercises the upload and readback transfers and nothing else, so it
    * separates "the image was never rendered" from "the image cannot be moved
    * between a tiled texture and linear memory". */
   {
      const long total = (long)width * height;
      unsigned char *src = malloc((size_t)total * 4);
      if (!src) {
         printf("out of memory\n");
         return 2;
      }
      for (long i = 0; i < total; i++) {
         src[i * 4 + 0] = CLEAR_R;
         src[i * 4 + 1] = CLEAR_G;
         src[i * 4 + 2] = CLEAR_B;
         src[i * 4 + 3] = 255;
      }
      GLuint tex, fbo;
      glGenTextures(1, &tex);
      glBindTexture(GL_TEXTURE_2D, tex);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, src);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tex, 0);
      glFinish();
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
         printf("case upload-only (%dx%d)\n", width, height);
         check_gl("upload");
         probe_readbacks("upload-only/texdata", CLEAR_R, CLEAR_G, CLEAR_B);
      } else {
         printf("case upload-only: incomplete FBO, skipped\n");
      }
      glDeleteFramebuffers(1, &fbo);
      glDeleteTextures(1, &tex);
      free(src);
   }

   /* Case 0b: the whole UNORM8 value range, with and without rendering. */
   run_unorm_ramp();

   /* Case 1: FBO with a texture colour attachment. */
   {
      GLuint tex, fbo;
      glGenTextures(1, &tex);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tex, 0);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
         if (!getenv("NEO_SKIP_READBACK_CASES"))
            run_case("fbo-texture");
         run_shape_sweep("fbo-texture");
         run_precision_sweep("fbo-texture");
      } else {
         printf("case fbo-texture: incomplete FBO, skipped\n");
      }
      glDeleteFramebuffers(1, &fbo);
      glDeleteTextures(1, &tex);
   }

   /* Case 2: FBO with a renderbuffer colour attachment. Same rendering, a
    * storage object that is never a texture. */
   {
      GLuint rb, fbo;
      glGenRenderbuffers(1, &rb);
      glBindRenderbuffer(GL_RENDERBUFFER, rb);
      glRenderbufferStorage(GL_RENDERBUFFER,
                            have_rgba8 ? GL_RGBA8_OES : GL_RGBA4, width,
                            height);
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_RENDERBUFFER, rb);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
         if (have_rgba8) {
            run_case("fbo-renderbuffer");
         } else {
            /* RGBA4 cannot represent the test colours exactly; running it
             * would report bogus mismatches. Say so instead. */
            printf("case fbo-renderbuffer: no GL_OES_rgb8_rgba8, skipped "
                   "(RGBA4 cannot hold the exact test colours)\n");
         }
      } else {
         printf("case fbo-renderbuffer: incomplete FBO, skipped\n");
      }
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fbo);
      glDeleteRenderbuffers(1, &rb);
   }

   /* Case 3: a real window surface on a gbm_surface. This is the buffer a
    * compositor would scan out, so it decides whether the panel is affected
    * or only the readback path. */
   if (getenv("NEO_SKIP_WINDOW_SURFACE")) {
      printf("case window-surface: NEO_SKIP_WINDOW_SURFACE set, skipped\n");
   } else if (have_cfg && gbm) {
      EGLint native_visual = 0;
      eglGetConfigAttrib(dpy, cfg, EGL_NATIVE_VISUAL_ID, &native_visual);
      struct gbm_surface *gs = gbm_surface_create(
         gbm, width, height, (uint32_t)native_visual,
         GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT);
      if (!gs)
         gs = gbm_surface_create(gbm, width, height, (uint32_t)native_visual,
                                 GBM_BO_USE_RENDERING);
      if (!gs) {
         printf("case window-surface: gbm_surface_create failed, skipped\n");
      } else {
         EGLSurface surf = eglCreateWindowSurface(
            dpy, cfg, (EGLNativeWindowType)gs, NULL);
         if (surf == EGL_NO_SURFACE) {
            printf("case window-surface: eglCreateWindowSurface failed 0x%x, "
                   "skipped\n",
                   eglGetError());
         } else if (!eglMakeCurrent(dpy, surf, surf, ctx)) {
            printf("case window-surface: eglMakeCurrent failed 0x%x, "
                   "skipped\n",
                   eglGetError());
         } else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (!getenv("NEO_SKIP_READBACK_CASES"))
               run_case("window-surface");
            run_shape_sweep("window-surface");
            run_precision_sweep("window-surface");
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
            eglDestroySurface(dpy, surf);
         }
         gbm_surface_destroy(gs);
      }
   } else {
      printf("case window-surface: %s, skipped\n",
             gbm ? "no window-capable EGLConfig" : "surfaceless run has no gbm_device");
   }

   printf("%s (%d failing checks)\n", failures ? "GL-FBO-READBACK FAIL"
                                               : "GL-FBO-READBACK PASS",
          failures);
   return failures ? 1 : 0;
}
