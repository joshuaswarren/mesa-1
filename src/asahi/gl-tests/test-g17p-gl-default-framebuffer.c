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
 */

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

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

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;

   /* No alpha is requested, which is the configuration the conformance run
    * fails on: the default framebuffer must still read alpha as one. */
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 0, EGL_NONE,
   };
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
   const EGLint surface_attributes[] = {EGL_WIDTH, SIZE, EGL_HEIGHT, SIZE,
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

   glViewport(0, 0, SIZE, SIZE);
   describe_default();
   walk_read_buffers();
   const bool ok = check_alpha();

   printf("default framebuffer: %s\n", ok ? "match" : "MISMATCH");
   return ok ? 0 : 1;
}
