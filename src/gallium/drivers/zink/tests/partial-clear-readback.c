/* SPDX-License-Identifier: MIT
 * Copyright 2026 Aurora Silicon
 * Standalone GLX regression: a partial read spanning several deferred clears.
 * cc -O2 partial-clear-readback.c -o partial-clear-readback \
 *    $(pkg-config --cflags --libs gl x11)
 * Run on a visible X11/Xwayland display with zink selected. No input required.
 */
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SIZE 128
#define READ_SIZE 64
static unsigned char pixels[READ_SIZE * READ_SIZE * 4];

static unsigned char
expected(int frame, int tile, int channel)
{
   return 17 + (frame * 37 + tile * 29 + channel * 53) % 220;
}

int
main(int argc, char **argv)
{
   int frames = argc > 1 ? atoi(argv[1]) : 100;
   if (frames <= 0)
      return 2;
   Display *dpy = XOpenDisplay(NULL);
   if (!dpy)
      return 77;
   int attrs[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
                  GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_RED_SIZE, 8,
                  GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
                  GLX_DOUBLEBUFFER, True, None};
   int count;
   GLXFBConfig *configs = glXChooseFBConfig(dpy, DefaultScreen(dpy), attrs, &count);
   if (!configs || !count)
      return 77;
   XVisualInfo *visual = glXGetVisualFromFBConfig(dpy, configs[0]);
   if (!visual)
      return 77;
   XSetWindowAttributes wa = {
      .colormap = XCreateColormap(dpy, RootWindow(dpy, visual->screen),
                                 visual->visual, AllocNone),
      .event_mask = StructureNotifyMask,
   };
   Window win = XCreateWindow(dpy, RootWindow(dpy, visual->screen), 0, 0,
                              SIZE, SIZE, 0, visual->depth, InputOutput,
                              visual->visual, CWColormap | CWEventMask, &wa);
   XStoreName(dpy, win, "zink partial-clear readback regression");
   XMapWindow(dpy, win);
   XEvent event;
   do { XNextEvent(dpy, &event); } while (event.type != MapNotify);
   GLXContext ctx = glXCreateNewContext(dpy, configs[0], GLX_RGBA_TYPE, NULL, True);
   if (!ctx || !glXMakeCurrent(dpy, win, ctx))
      return 77;
   printf("renderer=%s frames=%d\n", glGetString(GL_RENDERER), frames);
   glViewport(0, 0, SIZE, SIZE);
   glEnable(GL_SCISSOR_TEST);
   unsigned failed = 0;
   for (int frame = 0; frame < frames; ++frame) {
      /* No full clear: that would independently force all pending clears out.
       * Every sampled pixel is covered by a scissored clear in this frame.
       */
      for (int tile = 0; tile < 16; ++tile) {
         glScissor((tile % 4) * 32, (tile / 4) * 32, 32, 32);
         glClearColor(expected(frame, tile, 0) / 255.0f,
                      expected(frame, tile, 1) / 255.0f,
                      expected(frame, tile, 2) / 255.0f, 1);
         glClear(GL_COLOR_BUFFER_BIT);
      }
      glReadPixels(0, 0, READ_SIZE, READ_SIZE, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      unsigned bad = 0;
      for (int y = 0; y < READ_SIZE; ++y) {
         for (int x = 0; x < READ_SIZE; ++x) {
            int tile = (y / 32) * 4 + x / 32;
            for (int c = 0; c < 3; ++c)
               bad += abs(pixels[(y * READ_SIZE + x) * 4 + c] -
                          expected(frame, tile, c)) > 1;
         }
      }
      GLenum error = glGetError();
      failed += bad != 0 || error != GL_NO_ERROR;
      printf("frame=%d bad_rgb_channels=%u error=%#x\n", frame, bad, error);
      glXSwapBuffers(dpy, win);
      usleep(10000);
   }
   printf("SUMMARY frames=%d failed=%u\n", frames, failed);
   glXMakeCurrent(dpy, None, NULL);
   glXDestroyContext(dpy, ctx);
   XDestroyWindow(dpy, win);
   XFreeColormap(dpy, wa.colormap);
   XFree(visual);
   XFree(configs);
   XCloseDisplay(dpy);
   return failed != 0;
}
