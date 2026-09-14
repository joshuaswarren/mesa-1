/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Two conformance failures point at the default framebuffer rather than at any
 * shader.
 *
 * KHR-GL30.framebuffer_blit.scissor_blit blits from the default framebuffer
 * into a colour texture and reads back an alpha of zero where it expects one.
 * A configuration with no alpha channel must still read alpha as one, so this
 * checks what the default framebuffer actually returns -- through glReadPixels
 * and through a blit into an RGBA texture, which are separate paths.
 *
 * KHR-GL30.api.coverage fails with GL_INVALID_OPERATION from glReadBuffer, so
 * this also walks the buffers a default framebuffer may name and reports which
 * the driver refuses.
 *
 * Both answers came back clean, on hardware and on llvmpipe alike: alpha reads
 * one on a configuration without it, a blit into an RGBA texture keeps it with
 * or without a scissor and with or without a depth attachment, and only the
 * stereo buffers refuse a glReadBuffer. So neither failure is in these paths.
 *
 * ASAHI_EGL_CONFIG_ID selects a configuration by id, the way the conformance
 * runner does, because scissor_blit fails exactly on the configurations whose
 * default framebuffer carries depth (2, 3, 4) and passes on the one that does
 * not (1).
 *
 * The failing blit is, in the driver's own terms:
 *
 *   b10g10r10a2_unorm -> r8g8b8a8_unorm  mask=0xf  scissor=1[0,0,32,32]
 *   src[0,64 64x-64]  dst[0,0 64x64]  filter=0
 *
 * -- vertically flipped, because its source is the window and its destination
 * a texture. This oracle issues a blit with **byte-identical** parameters,
 * from a real window of the same configuration, and it comes out right. So the
 * failure is not in the parameters; it depends on what the context did before.
 *
 * Each knob below was added to chase that, and each one passes on hardware:
 * ASAHI_EGL_CONFIG_ID picks the conformance runner's configuration by id,
 * ASAHI_EGL_WINDOW takes a real window rather than a pbuffer, ASAHI_EGL_SWAPS
 * presents first so the context blits out of a freshly acquired buffer, and
 * ASAHI_DEPTH_BLIT_FIRST issues the flipped scissored depth blit that the
 * conformance case performs immediately before its colour one.
 *
 * The blit was innocent. Read straight out of the driver, the window's own
 * memory is already zero by the time the blit samples it, while the window's
 * depth still holds what the case cleared into it. Something earlier wrote
 * zeros over the colour, and the last two checks below say what.
 *
 * The case asks for a depth clear it cannot get: the window's depth is already
 * loaded, so the driver cannot fast clear it and u_blitter draws a full screen
 * quad with colour writes masked off instead. That draw loses the colour it
 * was drawn over -- but only on a ten-bit target. An RGBA8 attachment of the
 * same shape survives it, and blending the destination back over itself, which
 * loads the same tile, survives on both. So neither the tilebuffer load nor
 * the store is at fault: it is a colour mask of exactly zero.
 *
 * Formats whose tilebuffer writes cannot be masked -- RGB10A2 among them --
 * left their colour mask to nir_lower_blend, which deletes the store outright
 * when the mask is zero. On G17P a covered fragment that stores nothing loses
 * the tile it was drawn over, which agx_nir_lower_tilebuffer already knew and
 * handled for every other format. Writing nothing needs no masked store, so
 * the mask now goes to the tilebuffer lowering whatever the format.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 64

static const char *
error_name(GLenum error)
{
   switch (error) {
   case GL_NO_ERROR:
      return "none";
   case GL_INVALID_ENUM:
      return "INVALID_ENUM";
   case GL_INVALID_VALUE:
      return "INVALID_VALUE";
   case GL_INVALID_OPERATION:
      return "INVALID_OPERATION";
   default:
      return "other";
   }
}

/* What the default framebuffer says about itself, which is what decides
 * whether an alpha of one is required of it. */
static void
describe_default(void)
{
   GLint red = 0, green = 0, blue = 0, alpha = 0, depth = 0, stencil = 0;
   GLint samples = 0, doubled = 0;

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT,
                                         GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE,
                                         &red);
   glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT,
                                         GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE,
                                         &green);
   glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT,
                                         GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE,
                                         &blue);
   glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT,
                                         GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE,
                                         &alpha);
   glGetIntegerv(GL_DEPTH_BITS, &depth);
   glGetIntegerv(GL_STENCIL_BITS, &stencil);
   glGetIntegerv(GL_SAMPLES, &samples);
   glGetIntegerv(GL_DOUBLEBUFFER, &doubled);
   while (glGetError() != GL_NO_ERROR)
      ;

   printf("default framebuffer: r%d g%d b%d a%d depth%d stencil%d samples%d "
          "double%d\n",
          red, green, blue, alpha, depth, stencil, samples, doubled);
}

/* GL_INVALID_OPERATION here is the api.coverage failure. Report every buffer
 * rather than stopping at the first, so one run names them all. */
static void
walk_read_buffers(void)
{
   static const struct {
      const char *name;
      GLenum buffer;
   } buffers[] = {
      {"NONE", GL_NONE},         {"FRONT", GL_FRONT},
      {"BACK", GL_BACK},         {"LEFT", GL_LEFT},
      {"RIGHT", GL_RIGHT},       {"FRONT_LEFT", GL_FRONT_LEFT},
      {"FRONT_RIGHT", GL_FRONT_RIGHT}, {"BACK_LEFT", GL_BACK_LEFT},
      {"BACK_RIGHT", GL_BACK_RIGHT},   {"FRONT_AND_BACK", GL_FRONT_AND_BACK},
   };

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   while (glGetError() != GL_NO_ERROR)
      ;

   for (unsigned i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
      glReadBuffer(buffers[i].buffer);
      printf("glReadBuffer(%-15s) -> %s\n", buffers[i].name,
             error_name(glGetError()));
   }

   glReadBuffer(GL_BACK);
   while (glGetError() != GL_NO_ERROR)
      ;
}

/* Clear the default framebuffer to an opaque colour, then read it two ways.
 * Both must agree, and both must report alpha one. */
static bool
check_alpha(void)
{
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDisable(GL_SCISSOR_TEST);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   float direct[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
   glReadBuffer(GL_BACK);
   glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_RGBA, GL_FLOAT, direct);

   /* The same contents through a blit into a texture that does have alpha,
    * which is the path the conformance case takes. */
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   /* The conformance case's destination also carries a depth texture, and a
    * colour blit into it is what loses the alpha. */
   GLuint depth;
   glGenTextures(1, &depth);
   glBindTexture(GL_TEXTURE_2D, depth);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, SIZE, SIZE, 0,
                GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                          depth, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "default framebuffer: no RGBA8 destination\n");
      return false;
   }

   /* A value that cannot be confused with either the source or a zeroed
    * destination. */
   glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
   glClear(GL_COLOR_BUFFER_BIT);

   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
   glReadBuffer(GL_BACK);

   bool ok = true;
   for (unsigned scissored = 0; scissored < 2; ++scissored) {
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      glDisable(GL_SCISSOR_TEST);
      glClear(GL_COLOR_BUFFER_BIT);

      if (scissored) {
         glEnable(GL_SCISSOR_TEST);
         glScissor(0, 0, SIZE / 2, SIZE / 2);
      } else {
         glDisable(GL_SCISSOR_TEST);
      }

      /* The conformance case blits depth out of the same window into the same
       * framebuffer immediately before the colour blit, flipped and scissored
       * exactly as this one is. Reproduce that order: the colour blit is only
       * wrong when it follows it. */
      if (getenv("ASAHI_DEPTH_BLIT_FIRST") != NULL) {
         glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
         glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
         glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                           GL_DEPTH_BUFFER_BIT, GL_NEAREST);
      }

      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
      glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glDisable(GL_SCISSOR_TEST);

      float inside[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
      glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_RGBA, GL_FLOAT, inside);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      glReadBuffer(GL_BACK);

      if (glGetError() != GL_NO_ERROR) {
         fprintf(stderr, "default framebuffer: blit raised a GL error\n");
         return false;
      }

      const bool match = inside[3] == direct[3];
      ok &= match;
      printf("blit%-11s alpha %.3f, default framebuffer reads %.3f  %s\n",
             scissored ? " (scissored)" : "", inside[3], direct[3],
             match ? "" : "MISMATCH");
   }

   printf("default framebuffer reads RGBA = [%.3f %.3f %.3f %.3f]\n", direct[0],
          direct[1], direct[2], direct[3]);

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   glDeleteTextures(1, &depth);
   return ok;
}

/* Read one pixel of the bound read framebuffer's colour. */
static void
read_colour(GLenum buffer, float rgba[4])
{
   rgba[0] = rgba[1] = rgba[2] = rgba[3] = -1.0f;
   glReadBuffer(buffer);
   glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_RGBA, GL_FLOAT, rgba);
}

/* Read one pixel of the default framebuffer's colour. */
static void
read_default_colour(float rgba[4])
{
   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   read_colour(GL_BACK, rgba);
}

static bool
same_colour(const float a[4], const float b[4])
{
   for (unsigned c = 0; c < 4; ++c) {
      if (fabsf(a[c] - b[c]) > 1.0f / 255.0f)
         return false;
   }
   return true;
}

/*
 * The conformance case sets the window's colour, and then -- before it blits
 * that colour anywhere -- clears the window's depth and reads it back. Nothing
 * between the two touches the colour, so the colour must survive both.
 *
 * This is the one step the checks above never reproduced, and it is the step
 * the passing configuration skips: scissor_blit fails on configurations 2, 3
 * and 4, whose default framebuffer carries depth, and passes on 1, which has
 * none and so never clears it. Read the colour back after each step, so a
 * single run says which one loses it.
 */
static bool
check_depth_clear_keeps_colour(bool interleave)
{
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDisable(GL_SCISSOR_TEST);
   glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   float cleared[4];
   read_default_colour(cleared);

   /* The conformance case does not go straight from the window's colour to its
    * depth: it clears a texture of its own in between, so the window's batch is
    * flushed and rebuilt rather than extended. */
   if (interleave) {
      GLuint scratch_texture, scratch_fbo;
      glGenTextures(1, &scratch_texture);
      glBindTexture(GL_TEXTURE_2D, scratch_texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
      glGenFramebuffers(1, &scratch_fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, scratch_fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, scratch_texture, 0);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      float unused[4] = {0};
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_RGBA, GL_FLOAT, unused);

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glReadBuffer(GL_BACK);
      glDeleteFramebuffers(1, &scratch_fbo);
      glDeleteTextures(1, &scratch_texture);
   }

   /* The conformance case's own depth value, cleared into the window. */
   glClearDepth(0.25);
   glClear(GL_DEPTH_BUFFER_BIT);

   float after_depth_clear[4];
   read_default_colour(after_depth_clear);

   float depth = -1.0f;
   glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT,
                &depth);

   float after_depth_read[4];
   read_default_colour(after_depth_read);

   /* And finally through the blit, which is the path that reads zeros. */
   GLuint texture, fbo;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
   glClear(GL_COLOR_BUFFER_BIT);

   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
   glReadBuffer(GL_BACK);
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, SIZE / 2, SIZE / 2);
   glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE, GL_COLOR_BUFFER_BIT,
                     GL_NEAREST);
   glDisable(GL_SCISSOR_TEST);

   float blitted[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
   glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
   glReadBuffer(GL_COLOR_ATTACHMENT0);
   glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_RGBA, GL_FLOAT, blitted);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   glReadBuffer(GL_BACK);

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "default framebuffer: depth clear raised a GL error\n");
      return false;
   }

   const struct {
      const char *what;
      const float *colour;
   } steps[] = {
      {"cleared", cleared},
      {"after clearing depth", after_depth_clear},
      {"after reading depth", after_depth_read},
      {"through a blit", blitted},
   };

   bool ok = true;
   for (unsigned i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
      const bool match = same_colour(steps[i].colour, cleared);
      ok &= match;
      printf("%-13s colour %-20s [%.3f %.3f %.3f %.3f] %s\n",
             interleave ? "interleaved" : "direct", steps[i].what,
             steps[i].colour[0], steps[i].colour[1], steps[i].colour[2],
             steps[i].colour[3], match ? "" : "LOST");
   }
   printf("%-13s depth reads %.6f\n", interleave ? "interleaved" : "direct",
          depth);

   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &texture);
   return ok;
}

static GLuint
compile(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint ok = 0;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024] = {0};
      glGetShaderInfoLog(shader, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "default framebuffer: shader failed to compile: %s\n",
              log);
      return 0;
   }
   return shader;
}

static const char *depth_vs_source =
   "#version 330 core\n"
   "void main() {\n"
   "  vec2 p = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0,\n"
   "                (gl_VertexID & 2) == 0 ? -1.0 : 1.0);\n"
   "  gl_Position = vec4(p, 0.5, 1.0);\n"
   "}\n";

static const char *depth_fs_source =
   "#version 330 core\n"
   "out vec4 frag;\n"
   "void main() {\n"
   "  frag = vec4(1.0, 0.0, 0.0, 1.0);\n"
   "}\n";

/*
 * A depth-only draw must not disturb the colour it shares a tile with.
 *
 * This is what the conformance case really does. It asks for a depth clear,
 * but the window's depth is already loaded by then, so the driver cannot fast
 * clear it and u_blitter draws a full screen quad with colour writes masked
 * off instead. The batch reloads the window's colour, draws depth over it, and
 * stores the colour back at end of tile -- and the colour comes back zero.
 *
 * A fast clear of the same depth leaves the colour alone, so the reload and
 * the store are both fine on their own: it is the masked draw between them
 * that loses the colour.
 */
/* Two ways to ask a draw to leave the colour it finds alone. Both reach the
 * same place -- a tilebuffer load of the destination -- on a format whose
 * tilebuffer writes cannot be masked. */
enum preserve_mode {
   /* Colour writes masked off entirely, depth written: the shape u_blitter
    * reaches when it cannot fast clear the depth. */
   PRESERVE_MASKED_DEPTH,
   /* Blend the destination back over itself. */
   PRESERVE_BLEND_DST,
};

static bool
check_draw_keeps_colour(const char *what, GLenum colour_format,
                        enum preserve_mode mode)
{
   GLuint vs = compile(GL_VERTEX_SHADER, depth_vs_source);
   GLuint fs = compile(GL_FRAGMENT_SHADER, depth_fs_source);
   if (!vs || !fs)
      return false;

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[1024] = {0};
      glGetProgramInfoLog(program, sizeof(log) - 1, NULL, log);
      fprintf(stderr, "default framebuffer: link failed: %s\n", log);
      return false;
   }

   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   /* The window is the conformance case's own target; a texture of the same
    * shape says whether the loss belongs to the window or to the draw. */
   GLuint texture = 0, depth_texture = 0, fbo = 0;
   if (colour_format) {
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexImage2D(GL_TEXTURE_2D, 0, colour_format, SIZE, SIZE, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
      glGenTextures(1, &depth_texture);
      glBindTexture(GL_TEXTURE_2D, depth_texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, SIZE, SIZE, 0,
                   GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                             GL_TEXTURE_2D, depth_texture, 0);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
         printf("%-28s unsupported\n", what);
         return true;
      }
   } else {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
   }

   glDisable(GL_SCISSOR_TEST);
   glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   /* Read it back so the clear is submitted on its own. The draw below then
    * begins a batch that must reload the colour rather than clear it, which is
    * the shape the conformance case reaches. */
   float cleared[4];
   read_colour(colour_format ? GL_COLOR_ATTACHMENT0 : GL_BACK, cleared);

   glUseProgram(program);
   if (mode == PRESERVE_MASKED_DEPTH) {
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_ALWAYS);
      glDepthMask(GL_TRUE);
      glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
   } else {
      glEnable(GL_BLEND);
      glBlendFunc(GL_ZERO, GL_ONE);
   }

   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_BLEND);
   glUseProgram(0);

   float after[4];
   read_colour(colour_format ? GL_COLOR_ATTACHMENT0 : GL_BACK, after);

   float depth = -1.0f;
   glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT,
                &depth);

   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "default framebuffer: depth draw raised a GL error\n");
      return false;
   }

   const bool ok = same_colour(after, cleared);
   printf("%-28s [%.3f %.3f %.3f %.3f] -> [%.3f %.3f %.3f %.3f] "
          "depth %.6f %s\n",
          what, cleared[0], cleared[1], cleared[2], cleared[3], after[0],
          after[1], after[2], after[3], depth, ok ? "" : "LOST");

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glReadBuffer(GL_BACK);
   if (fbo)
      glDeleteFramebuffers(1, &fbo);
   if (texture)
      glDeleteTextures(1, &texture);
   if (depth_texture)
      glDeleteTextures(1, &depth_texture);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   return ok;
}

/* The failing conformance blit copies the window -- b10g10r10a2_unorm, as the
 * driver's resource log names it -- into an r8g8b8a8_unorm texture under a
 * scissor. Build that pair directly, so no window is needed. */
static bool
check_rgb10_a2_blit(void)
{
   GLuint source_texture, source_fbo;
   glGenTextures(1, &source_texture);
   glBindTexture(GL_TEXTURE_2D, source_texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2, SIZE, SIZE, 0, GL_RGBA,
                GL_UNSIGNED_INT_2_10_10_10_REV, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glGenFramebuffers(1, &source_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, source_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          source_texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "rgb10a2 blit: no RGB10_A2 render target\n");
      return true;
   }
   glDisable(GL_SCISSOR_TEST);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   GLuint destination_texture, destination_fbo;
   glGenTextures(1, &destination_texture);
   glBindTexture(GL_TEXTURE_2D, destination_texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glGenFramebuffers(1, &destination_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, destination_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          destination_texture, 0);

   bool ok = true;
   for (unsigned combination = 0; combination < 4; ++combination) {
      const bool scissored = combination & 1;
      const bool flipped = combination & 2;

      glBindFramebuffer(GL_FRAMEBUFFER, destination_fbo);
      glDisable(GL_SCISSOR_TEST);
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      glClear(GL_COLOR_BUFFER_BIT);

      if (scissored) {
         glEnable(GL_SCISSOR_TEST);
         glScissor(0, 0, SIZE / 2, SIZE / 2);
      }

      glBindFramebuffer(GL_READ_FRAMEBUFFER, source_fbo);
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destination_fbo);
      /* The conformance blit that fails is vertically flipped -- the driver
       * sees src[0,64 64x-64] -- because its source is the window and its
       * destination a texture. Flip here too. */
      if (flipped) {
         glBlitFramebuffer(0, 0, SIZE, SIZE, 0, SIZE, SIZE, 0,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
      } else {
         glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
      }
      glDisable(GL_SCISSOR_TEST);

      float inside[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
      glBindFramebuffer(GL_READ_FRAMEBUFFER, destination_fbo);
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      glReadPixels(SIZE / 4, SIZE / 4, 1, 1, GL_RGBA, GL_FLOAT, inside);

      if (glGetError() != GL_NO_ERROR) {
         fprintf(stderr, "rgb10a2 blit: raised a GL error\n");
         return false;
      }

      const bool match = inside[3] == 1.0f;
      ok &= match;
      printf("rgb10a2 -> rgba8 %-9s%-9s alpha %.3f  %s\n",
             scissored ? "scissored" : "", flipped ? "flipped" : "", inside[3],
             match ? "" : "MISMATCH, wanted 1.000");
   }

   glDeleteFramebuffers(1, &source_fbo);
   glDeleteFramebuffers(1, &destination_fbo);
   glDeleteTextures(1, &source_texture);
   glDeleteTextures(1, &destination_texture);
   return ok;
}

int
main(void)
{
   /* The conformance failure needs the window itself as the blit source, not a
    * texture of the same format, so ASAHI_EGL_WINDOW asks for a real one. */
   const bool want_window = getenv("ASAHI_EGL_WINDOW") != NULL;
   Display *xdisplay = NULL;
   if (want_window) {
      xdisplay = XOpenDisplay(NULL);
      if (!xdisplay) {
         fprintf(stderr, "no X display for a window surface\n");
         return 1;
      }
   }

   EGLDisplay display = xdisplay
                           ? eglGetDisplay((EGLNativeDisplayType)xdisplay)
                           : eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;

   /* No alpha, and depth: that is the shape of every configuration the
    * conformance run fails on. The default framebuffer must still read alpha
    * as one, and clearing its depth must not cost it its colour. */
   const EGLint *config_attributes = (const EGLint[]){
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 0, EGL_DEPTH_SIZE, 16, EGL_NONE,
   };
   const EGLint window_config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_DEPTH_SIZE, 16, EGL_NONE,
   };
   if (want_window)
      config_attributes = window_config_attributes;
   /* The conformance runner names an EGL config by id, and the failure is
    * config dependent, so allow the same selection here. */
   EGLConfig config;
   EGLint count;
   const char *wanted = getenv("ASAHI_EGL_CONFIG_ID");
   if (wanted) {
      EGLConfig all[256];
      if (!eglGetConfigs(display, all, 256, &count))
         return 1;
      const EGLint id = (EGLint)strtol(wanted, NULL, 10);
      bool found = false;
      for (EGLint i = 0; i < count; ++i) {
         EGLint this_id = 0;
         eglGetConfigAttrib(display, all[i], EGL_CONFIG_ID, &this_id);
         if (this_id != id)
            continue;
         config = all[i];
         found = true;
         break;
      }
      if (!found) {
         fprintf(stderr, "no EGL config with id %d\n", id);
         return 1;
      }
   } else if (!eglChooseConfig(display, config_attributes, &config, 1,
                               &count) ||
              count != 1) {
      return 1;
   }

   {
      static const struct {
         const char *name;
         EGLint attribute;
      } attributes[] = {
         {"id", EGL_CONFIG_ID},      {"r", EGL_RED_SIZE},
         {"g", EGL_GREEN_SIZE},      {"b", EGL_BLUE_SIZE},
         {"a", EGL_ALPHA_SIZE},      {"depth", EGL_DEPTH_SIZE},
         {"stencil", EGL_STENCIL_SIZE}, {"samples", EGL_SAMPLES},
         {"surface", EGL_SURFACE_TYPE},
      };
      printf("egl config:");
      for (unsigned i = 0; i < sizeof(attributes) / sizeof(attributes[0]);
           ++i) {
         EGLint value = 0;
         eglGetConfigAttrib(display, config, attributes[i].attribute, &value);
         printf(" %s%d", attributes[i].name, value);
      }
      printf("\n");
   }
   EGLSurface surface;
   if (want_window) {
      EGLint visual = 0;
      eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &visual);
      const Window root = DefaultRootWindow(xdisplay);
      const Window window =
         XCreateSimpleWindow(xdisplay, root, 0, 0, SIZE, SIZE, 0, 0, 0);
      XMapWindow(xdisplay, window);
      XSync(xdisplay, False);
      surface = eglCreateWindowSurface(display, config,
                                       (EGLNativeWindowType)window, NULL);
   } else {
      const EGLint surface_attributes[] = {EGL_WIDTH, SIZE, EGL_HEIGHT, SIZE,
                                           EGL_NONE};
      surface = eglCreatePbufferSurface(display, config, surface_attributes);
   }
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   if (getenv("ASAHI_GL_NOOP_SHIM") != NULL)
      return 0;

   /* Which driver answered. A GL client that misses the campaign's loader
    * environment lands on llvmpipe, where every one of these checks passes, so
    * say so rather than let a control be read as a hardware result. */
   printf("renderer %s (%s)\n", glGetString(GL_RENDERER), glGetString(GL_VENDOR));

   glViewport(0, 0, SIZE, SIZE);

   /* A window that has been presented hands the next frame a different buffer,
    * so a blit that samples the default framebuffer afterwards is reading a
    * buffer the context has only just acquired. ASAHI_EGL_SWAPS asks for that
    * shape, which a pbuffer never reaches. */
   const char *swaps = getenv("ASAHI_EGL_SWAPS");
   for (int i = 0, n = swaps ? atoi(swaps) : 0; i < n; ++i) {
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      eglSwapBuffers(display, surface);
   }

   describe_default();
   walk_read_buffers();
   bool ok = check_alpha();
   ok &= check_rgb10_a2_blit();
   ok &= check_depth_clear_keeps_colour(false);
   ok &= check_depth_clear_keeps_colour(true);
   ok &= check_draw_keeps_colour("masked depth draw: window", 0,
                                 PRESERVE_MASKED_DEPTH);
   ok &= check_draw_keeps_colour("masked depth draw: rgba8", GL_RGBA8,
                                 PRESERVE_MASKED_DEPTH);
   ok &= check_draw_keeps_colour("masked depth draw: rgb10_a2", GL_RGB10_A2,
                                 PRESERVE_MASKED_DEPTH);
   ok &= check_draw_keeps_colour("blend destination: window", 0,
                                 PRESERVE_BLEND_DST);
   ok &= check_draw_keeps_colour("blend destination: rgba8", GL_RGBA8,
                                 PRESERVE_BLEND_DST);
   ok &= check_draw_keeps_colour("blend destination: rgb10_a2", GL_RGB10_A2,
                                 PRESERVE_BLEND_DST);

   printf("default framebuffer: %s\n", ok ? "match" : "MISMATCH");
   return ok ? 0 : 1;
}
