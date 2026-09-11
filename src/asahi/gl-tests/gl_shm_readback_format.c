/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Does the stack accept the exact (format, type) pairs a wlr-screencopy SHM
 * capture passes to glReadPixels?
 *
 * Hyprland 0.56.1 captures an output into an SHM buffer by allocating a fresh
 * FBO in the client's DRM format, clearing it to {0,0,0,0}, drawing the monitor
 * mirror texture into it, and reading it back --
 * src/render/gl/GLFramebuffer.cpp:136 CGLFramebuffer::readPixels().  Two things
 * about that function make a rejected read indistinguishable from a working one:
 *
 *   - it rewrites the read format before the call.  With no swizzle in the
 *     format table and a table glFormat of GL_RGBA it substitutes GL_BGRA_EXT
 *     (GLFramebuffer.cpp:225-227), so the pair actually issued is not the pair
 *     the format table describes; and
 *
 *   - it never calls glGetError() and returns true unconditionally
 *     (GLFramebuffer.cpp:232-244).  A rejected glReadPixels writes nothing,
 *     the compositor still reports the frame ready, and the client keeps the
 *     zero-filled memfd it allocated.  In an alpha-carrying capture format that
 *     is a fully transparent image, which every viewer flattens onto white.
 *
 * So a blank capture with no error anywhere, correct pixels on the panel, and no
 * GPU fault is exactly what a rejected (format, type) pair looks like, and it is
 * decided entirely in Mesa's state tracker -- before any GPU work.  That is why
 * this probe is meaningful under the no-op G17P drm-shim, where nothing executes.
 *
 * The probe walks both capture formats Hyprland can negotiate.  It is the 10-bit
 * one that matters: getPreferredReadFormat() returns the output's DRM format
 * (Monitor.cpp:2482) and the session flips XRGB2101010/ARGB2101010 to
 * XBGR2101010 (ScreenshareSession.cpp:110-111), so a 10-bit output produces a
 * 10-bit capture unless misc:screencopy_force_8b is set.
 *
 * Build (against an installed GL stack):
 *   cc -O2 -o gl_shm_readback_format gl_shm_readback_format.c -lEGL -lGLESv2
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80E1
#endif

static const char *fmtname(GLenum e) {
   switch (e) {
   case GL_RGBA: return "GL_RGBA";
   case GL_RGB: return "GL_RGB";
   case GL_BGRA_EXT: return "GL_BGRA_EXT";
   case GL_RGBA_INTEGER: return "GL_RGBA_INTEGER";
   default: return "?";
   }
}
static const char *typname(GLenum e) {
   switch (e) {
   case GL_UNSIGNED_BYTE: return "GL_UNSIGNED_BYTE";
   case GL_UNSIGNED_INT_2_10_10_10_REV: return "GL_UNSIGNED_INT_2_10_10_10_REV";
   case GL_UNSIGNED_SHORT_5_6_5: return "GL_UNSIGNED_SHORT_5_6_5";
   case GL_HALF_FLOAT: return "GL_HALF_FLOAT";
   default: return "?";
   }
}

static int probe(const char *label, GLenum internal, int w, int h,
                 GLenum pairs[][2], int npairs)
{
   GLuint tex, fb;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, internal, w, h);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   if (st != GL_FRAMEBUFFER_COMPLETE) {
      printf("%-22s FBO INCOMPLETE (0x%x)\n", label, st);
      return 1;
   }
   while (glGetError() != GL_NO_ERROR) {}

   GLint irf = 0, irt = 0;
   glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &irf);
   glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &irt);
   printf("%-22s %dx%d  IMPLEMENTATION_COLOR_READ_FORMAT=%s(0x%x) TYPE=%s(0x%x)\n",
          label, w, h, fmtname((GLenum)irf), irf, typname((GLenum)irt), irt);

   int bad = 0;
   static unsigned char buf[4096 * 8];
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   for (int i = 0; i < npairs; ++i) {
      while (glGetError() != GL_NO_ERROR) {}
      /* one row only: enough to trigger validation, small enough to be safe */
      glReadPixels(0, 0, w < 512 ? w : 512, 1, pairs[i][0], pairs[i][1], buf);
      GLenum e = glGetError();
      printf("      glReadPixels(%-12s, %-32s) -> %s\n",
             fmtname(pairs[i][0]), typname(pairs[i][1]),
             e == GL_NO_ERROR ? "OK" :
             e == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION  *** WRITES NOTHING ***" :
             e == GL_INVALID_ENUM ? "GL_INVALID_ENUM  *** WRITES NOTHING ***" : "OTHER ERROR");
      if (e != GL_NO_ERROR) bad = 1;
   }
   glPixelStorei(GL_PACK_ALIGNMENT, 4);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fb);
   glDeleteTextures(1, &tex);
   return bad;
}

int main(void)
{
   EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (dpy == EGL_NO_DISPLAY) { printf("no EGL display\n"); return 2; }
   if (!eglInitialize(dpy, NULL, NULL)) { printf("eglInitialize failed\n"); return 2; }
   eglBindAPI(EGL_OPENGL_ES_API);
   EGLint cfgattr[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
   EGLConfig cfg; EGLint n = 0;
   if (!eglChooseConfig(dpy, cfgattr, &cfg, 1, &n) || n < 1) { printf("no config\n"); return 2; }
   EGLint ctxattr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxattr);
   if (ctx == EGL_NO_CONTEXT) { printf("no context\n"); return 2; }
   if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) { printf("makeCurrent failed\n"); return 2; }

   printf("GL_RENDERER: %s\n", (const char *)glGetString(GL_RENDERER));
   printf("GL_VERSION : %s\n", (const char *)glGetString(GL_VERSION));
   const char *ext = (const char *)glGetString(GL_EXTENSIONS);
   printf("EXT_read_format_bgra: %s\n",
          (ext && strstr(ext, "GL_EXT_read_format_bgra")) ? "yes" : "NO");
   printf("EXT_texture_format_BGRA8888: %s\n",
          (ext && strstr(ext, "GL_EXT_texture_format_BGRA8888")) ? "yes" : "NO");

   /* Hyprland ARGB8888/XRGB8888 shm capture: FBO is RGBA8, and readPixels()
    * rewrites GL_RGBA -> GL_BGRA_EXT when the format carries no swizzle. */
   GLenum rgba8_pairs[][2] = {
      { GL_RGBA,      GL_UNSIGNED_BYTE },                  /* spec-mandated */
      { GL_BGRA_EXT,  GL_UNSIGNED_BYTE },                  /* what Hyprland sends */
   };
   /* Hyprland ABGR2101010/XBGR2101010 shm capture: FBO is RGB10_A2. */
   GLenum rgb10_pairs[][2] = {
      { GL_RGBA,      GL_UNSIGNED_BYTE },
      { GL_RGBA,      GL_UNSIGNED_INT_2_10_10_10_REV },    /* what Hyprland sends */
      { GL_BGRA_EXT,  GL_UNSIGNED_INT_2_10_10_10_REV },
   };

   int bad = 0;
   bad |= probe("RGBA8 (ARGB8888)",  GL_RGBA8,   2408, 1506, rgba8_pairs, 2);
   bad |= probe("RGB10_A2 (A2B10..)", GL_RGB10_A2, 2408, 1506, rgb10_pairs, 3);

   printf("%s\n", bad ? "READFMT: at least one pair Hyprland sends is REJECTED"
                      : "READFMT: every pair Hyprland sends is accepted");
   return bad ? 1 : 0;
}
