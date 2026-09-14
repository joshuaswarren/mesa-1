/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_depth_config
 * reports a depth failure, but its trace shows depth arriving correctly and
 * the stencil half reading zero. This separates the two aspects and the two
 * sample counts, so a failure says which combination loses the stencil.
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 16

static bool
run(unsigned samples)
{
   GLuint src_rb, src_fbo, dst_rb, dst_fbo;

   glGenRenderbuffers(1, &src_rb);
   glBindRenderbuffer(GL_RENDERBUFFER, src_rb);
   if (samples > 1) {
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples,
                                       GL_DEPTH24_STENCIL8, SIZE, SIZE);
   } else {
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SIZE, SIZE);
   }
   glGenFramebuffers(1, &src_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, src_fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, src_rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "stencil blit: %u samples unsupported as a source\n",
              samples);
      return true;
   }

   glGenRenderbuffers(1, &dst_rb);
   glBindRenderbuffer(GL_RENDERBUFFER, dst_rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SIZE, SIZE);
   glGenFramebuffers(1, &dst_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, dst_rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "stencil blit: the destination is incomplete\n");
      return false;
   }

   /* Distinct values on both sides, so a lost blit and a stale destination
    * look different. */
   glBindFramebuffer(GL_FRAMEBUFFER, src_fbo);
   glClearDepth(0.25);
   glClearStencil(1);
   glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

   glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
   glClearDepth(1.0);
   glClearStencil(7);
   glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

   glBindFramebuffer(GL_READ_FRAMEBUFFER, src_fbo);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_fbo);
   glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                     GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
   glFinish();
   if (glGetError() != GL_NO_ERROR) {
      fprintf(stderr, "stencil blit: %u samples raised a GL error\n", samples);
      return false;
   }

   glBindFramebuffer(GL_READ_FRAMEBUFFER, dst_fbo);
   GLfloat depth = 0.0f;
   GLubyte stencil = 0;
   glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT,
                &depth);
   glReadPixels(SIZE / 2, SIZE / 2, 1, 1, GL_STENCIL_INDEX,
                GL_UNSIGNED_BYTE, &stencil);

   const bool depth_ok = depth > 0.20f && depth < 0.30f;
   const bool stencil_ok = stencil == 1;
   fprintf(stderr,
           "stencil blit: %u samples -> depth %.3f %s, stencil %u %s\n",
           samples, depth, depth_ok ? "ok" : "WRONG (expected 0.25)",
           stencil, stencil_ok ? "ok" : "WRONG (expected 1)");

   glDeleteFramebuffers(1, &dst_fbo);
   glDeleteRenderbuffers(1, &dst_rb);
   glDeleteFramebuffers(1, &src_fbo);
   glDeleteRenderbuffers(1, &src_rb);
   return depth_ok && stencil_ok;
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
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attributes, &config, 1, &count) ||
       count != 1)
      return 1;
   const EGLint surface_attributes[] = {EGL_WIDTH, 64, EGL_HEIGHT, 64,
                                        EGL_NONE};
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return 1;

   if (getenv("ASAHI_GL_NOOP_SHIM") != NULL)
      return 0;

   bool ok = true;
   static const unsigned counts[] = {1u, 2u, 4u};
   for (unsigned i = 0; i < 3; ++i)
      ok &= run(counts[i]);

   fprintf(stderr, "stencil blit: %s\n", ok ? "match" : "MISMATCH");
   return ok ? 0 : 1;
}
