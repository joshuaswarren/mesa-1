/* SPDX-License-Identifier: MIT
 * Copyright 2026 Aurora Silicon
 * A resolved attachment may be discarded; a subsequent single-to-multisample
 * blit must preserve its destination for the next partial draw. No readback
 * separates those operations.
 * cc -O2 -Wall -Wextra blit-after-invalidate.c -o blit-after-invalidate \
 *    $(pkg-config --cflags --libs gl x11)
 */
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 128

static GLuint
shader(GLenum type, const char *source)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &source, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(2);
   }
   return s;
}

static GLuint
program(const char *fragment)
{
   const char *vertex =
      "#version 330\n"
      "void main() { vec2 p[3] = vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));"
      "gl_Position=vec4(p[gl_VertexID],0,1); }\n";
   GLuint vs = shader(GL_VERTEX_SHADER, vertex);
   GLuint fs = shader(GL_FRAGMENT_SHADER, fragment);
   GLuint p = glCreateProgram();
   glAttachShader(p, vs);
   glAttachShader(p, fs);
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   glDeleteShader(vs);
   glDeleteShader(fs);
   if (!ok)
      exit(2);
   return p;
}

static GLuint
texture_framebuffer(GLuint *texture)
{
   GLuint fb;
   glGenTextures(1, texture);
   glBindTexture(GL_TEXTURE_2D, *texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         GL_TEXTURE_2D, *texture, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      exit(77);
   return fb;
}

int
main(int argc, char **argv)
{
   int frames = argc > 1 ? atoi(argv[1]) : 1;
   if (frames < 1)
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
   XStoreName(dpy, win, "zink blit-after-invalidate regression");
   XMapWindow(dpy, win);
   XEvent event;
   do { XNextEvent(dpy, &event); } while (event.type != MapNotify);
   GLXContext ctx = glXCreateNewContext(dpy, configs[0], GLX_RGBA_TYPE, NULL, True);
   if (!ctx || !glXMakeCurrent(dpy, win, ctx))
      return 77;
   printf("renderer=%s version=%s frames=%d\n",
          glGetString(GL_RENDERER), glGetString(GL_VERSION), frames);

   GLuint solid = program("#version 330\nout vec4 c; void main(){c=vec4(1,0,0,1);}");
   GLuint sample = program("#version 330\nuniform sampler2D s; out vec4 c;"
                           "void main(){c=texelFetch(s,ivec2(gl_FragCoord.xy),0);}");
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   GLuint resolved_tex, output_tex, check_tex;
   GLuint resolved = texture_framebuffer(&resolved_tex);
   GLuint output = texture_framebuffer(&output_tex);
   GLuint check = texture_framebuffer(&check_tex);
   GLuint bait, rb;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, SIZE, SIZE);
   glGenFramebuffers(1, &bait);
   glBindFramebuffer(GL_FRAMEBUFFER, bait);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 77;
   glViewport(0, 0, SIZE, SIZE);
   glBindFramebuffer(GL_FRAMEBUFFER, output);
   glClearColor(0, 1, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish(); /* Initialize the preserved region once, before the test. */

   unsigned failed = 0;
   for (int frame = 0; frame < frames; ++frame) {
      glBindFramebuffer(GL_FRAMEBUFFER, bait);
      glDisable(GL_SCISSOR_TEST);
      glUseProgram(solid);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, bait);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolved);
      glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glBindFramebuffer(GL_FRAMEBUFFER, bait);
      GLenum attachment = GL_COLOR_ATTACHMENT0;
      glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &attachment);

      /* Populate the multisample attachment using an internal blitter pass.
       * The following partial draw needs its preserved green background.
       */
      glBindFramebuffer(GL_READ_FRAMEBUFFER, output);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, bait);
      glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glBindFramebuffer(GL_FRAMEBUFFER, bait);
      glEnable(GL_SCISSOR_TEST);
      glScissor(16, 16, 64, 64);
      glUseProgram(solid);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glDisable(GL_SCISSOR_TEST);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, bait);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolved);
      glBlitFramebuffer(0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glBindFramebuffer(GL_FRAMEBUFFER, bait);
      glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &attachment);
      glBindFramebuffer(GL_FRAMEBUFFER, check);
      glUseProgram(sample);
      glBindTexture(GL_TEXTURE_2D, resolved_tex);
      glUniform1i(glGetUniformLocation(sample, "s"), 0);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      unsigned char pixels[SIZE * SIZE * 4];
      glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      unsigned bad = 0;
      for (int y = 0; y < SIZE; ++y) {
         for (int x = 0; x < SIZE; ++x) {
            bool red = x >= 16 && x < 80 && y >= 16 && y < 80;
            const unsigned char *p = pixels + (y * SIZE + x) * 4;
            bad += abs((int)p[0] - (red ? 255 : 0)) > 1;
            bad += abs((int)p[1] - (red ? 0 : 255)) > 1;
            bad += p[2] > 1;
         }
      }
      GLenum error = glGetError();
      failed += bad != 0 || error != GL_NO_ERROR;
      printf("frame=%d bad_rgb_channels=%u error=%#x\n", frame, bad, error);
   }
   printf("SUMMARY frames=%d failed=%u\n", frames, failed);
   glDeleteFramebuffers(1, &bait);
   glDeleteRenderbuffers(1, &rb);
   GLuint fbs[] = {resolved, output, check};
   GLuint tex[] = {resolved_tex, output_tex, check_tex};
   glDeleteFramebuffers(3, fbs);
   glDeleteTextures(3, tex);
   glDeleteProgram(solid);
   glDeleteProgram(sample);
   glDeleteVertexArrays(1, &vao);
   glXMakeCurrent(dpy, None, NULL);
   glXDestroyContext(dpy, ctx);
   XDestroyWindow(dpy, win);
   XFreeColormap(dpy, wa.colormap);
   XFree(visual);
   XFree(configs);
   XCloseDisplay(dpy);
   return failed ? 1 : 0;
}
