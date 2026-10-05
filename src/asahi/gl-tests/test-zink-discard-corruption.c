/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * EGL setup follows Mesa's test-g17p-gl-depth-texture.c (MIT).
 * Reproduces the t-1130 intermittent offscreen discard corruption.
 * Default draw/query sequence matches discard-depth-notequal.c.
 * T1192_RUNS changes draws per arm (default 9), T1192_TIMER=0 disables
 * elapsed queries, T1192_CHECK_EACH=1 validates each draw instead of only
 * the final draw, and T1192_ITERATIONS changes fragment loop length.
 * T1192_SHADER=solid/discard removes texture and loop work; T1192_DEPTH=0
 * disables depth tests. T1192_OCCLUSION=1 reports an independent visibility
 * query. T1192_RECHECK retries bad readbacks immediately and after 10 ms;
 * T1192_READ_DELAY_US delays the initial readback. Delays are diagnostics.
 * argv[1] is the output prefix; bad frames include an 8-bit PGM mask.
 * Run with Zink -> Honeykrisp; no display or Steam changes.
 */
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W 512
#define H 512

static GLuint
shader(GLenum stage, const char *src)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(1);
   }
   return s;
}

static double
now_ms(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int
main(int argc, char **argv)
{
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor, n;
   if (!eglInitialize(d, &major, &minor) || !eglBindAPI(EGL_OPENGL_ES_API))
      return 1;
   EGLint ca[] = {EGL_SURFACE_TYPE,
                  EGL_PBUFFER_BIT,
                  EGL_RENDERABLE_TYPE,
                  EGL_OPENGL_ES3_BIT,
                  EGL_RED_SIZE,
                  8,
                  EGL_GREEN_SIZE,
                  8,
                  EGL_BLUE_SIZE,
                  8,
                  EGL_ALPHA_SIZE,
                  8,
                  EGL_DEPTH_SIZE,
                  24,
                  EGL_NONE};
   EGLConfig cfg;
   if (!eglChooseConfig(d, ca, &cfg, 1, &n) || n != 1)
      return 1;
   EGLint pa[] = {EGL_WIDTH, W, EGL_HEIGHT, H, EGL_NONE};
   EGLint ctxa[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   EGLSurface surf = eglCreatePbufferSurface(d, cfg, pa);
   EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ctxa);
   if (!eglMakeCurrent(d, surf, surf, ctx))
      return 1;
   printf("renderer=%s version=%s\n", glGetString(GL_RENDERER),
          glGetString(GL_VERSION));
   if (!strstr((const char *)glGetString(GL_RENDERER), "zink"))
      return 2;
   const char *vs =
      "#version 300 es\n"
      "const vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));\n"
      "out vec2 uv; void main(){uv=p[gl_VertexID]*0.5+0.5;"
      "gl_Position=vec4(p[gl_VertexID],-1,1);}\n";
   const char *fs =
      "#version 300 es\nprecision highp float;\n"
      "in vec2 uv; uniform sampler2D tex; uniform int iterations;\n"
      "out vec4 colour; void main(){ bool c=gl_FragCoord.x<256.0;\n"
      "if(c) discard; if(!c){ float v=texture(tex,uv).r;\n"
      "for(int i=0;i<iterations;i++) v=sin(v+float(i)*0.01)+0.3;\n"
      "colour=vec4(0,1,clamp(v*0.1+0.5,0.0,1.0),1); }}\n";
   const char *variant = getenv("T1192_SHADER");
   bool full = variant && (!strcmp(variant, "solid") ||
                           !strcmp(variant, "texture-loop-nodiscard"));
   if (variant && (!strcmp(variant, "solid") || !strcmp(variant, "discard")))
      fs =
         full
            ? "#version 300 es\nprecision highp float; out vec4 colour; void main(){colour=vec4(0,1,0.5,1);}\n"
            : "#version 300 es\nprecision highp float; out vec4 colour; void main(){bool c=gl_FragCoord.x<256.0; if(c) discard; if(!c) colour=vec4(0,1,0.5,1);}\n";
   char reduced_fs[2048];
   if (variant && !strcmp(variant, "texture-loop-nodiscard")) {
      snprintf(
         reduced_fs, sizeof(reduced_fs),
         "#version 300 es\nprecision highp float; in vec2 uv; uniform sampler2D tex; "
         "uniform int iterations;\n"
         "out vec4 colour; void main(){float v=texture(tex,uv).r; "
         "for(int i=0;i<iterations;i++) v=sin(v+float(i)*0.01)+0.3; "
         "colour=vec4(0,1,clamp(v*0.1+0.5,0.0,1.0),1);}\n");
      fs = reduced_fs;
   }
   if (variant && !strcmp(variant, "varying-loop")) {
      snprintf(
         reduced_fs, sizeof(reduced_fs),
         "#version 300 es\nprecision highp float; in vec2 uv; uniform int iterations;\n"
         "out vec4 colour; void main(){bool c=gl_FragCoord.x<256.0; if(c) discard;"
         "if(!c){float v=uv.x; for(int i=0;i<iterations;i++) "
         "v=sin(v+float(i)*0.01)+0.3; "
         "colour=vec4(0,1,clamp(v*0.1+0.5,0.0,1.0),1);}}\n");
      fs = reduced_fs;
   }
   if (variant && (!strcmp(variant, "explicit-lod-loop") ||
                   !strcmp(variant, "constant-tex-loop"))) {
      const char *sample = !strcmp(variant, "explicit-lod-loop")
                              ? "textureLod(tex,uv,0.0).r"
                              : "texture(tex,vec2(0.5)).r";
      snprintf(
         reduced_fs, sizeof(reduced_fs),
         "#version 300 es\nprecision highp float; in vec2 uv; uniform sampler2D tex; "
         "uniform int iterations;\n"
         "out vec4 colour; void main(){bool c=gl_FragCoord.x<256.0; if(c) discard;"
         "if(!c){float v=%s; for(int i=0;i<iterations;i++) "
         "v=sin(v+float(i)*0.01)+0.3; "
         "colour=vec4(0,1,clamp(v*0.1+0.5,0.0,1.0),1);}}\n",
         sample);
      fs = reduced_fs;
   }
   if (variant &&
       (!strcmp(variant, "constant-tex") || !strcmp(variant, "explicit-lod") ||
        !strcmp(variant, "varying"))) {
      const char *expression =
         !strcmp(variant, "constant-tex")   ? "texture(tex,vec2(0.5)).r"
         : !strcmp(variant, "explicit-lod") ? "textureLod(tex,uv,0.0).r"
                                            : "uv.x";
      snprintf(
         reduced_fs, sizeof(reduced_fs),
         "#version 300 es\nprecision highp float; in vec2 uv; uniform sampler2D tex;\n"
         "out vec4 colour; void main(){bool c=gl_FragCoord.x<256.0; if(c) discard;"
         "if(!c) colour=vec4(0,1,%s,1);}\n",
         expression);
      fs = reduced_fs;
   }
   bool depth = !getenv("T1192_DEPTH") || atoi(getenv("T1192_DEPTH"));
   GLuint p = glCreateProgram();
   glAttachShader(p, shader(GL_VERTEX_SHADER, vs));
   glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
   glLinkProgram(p);
   GLint linked;
   glGetProgramiv(p, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      return 1;
   }
   glUseProgram(p);
   const char *value = getenv("T1192_ITERATIONS");
   glUniform1i(glGetUniformLocation(p, "iterations"),
               value ? atoi(value) : 1024);
   GLuint tex, vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   uint8_t white[] = {64, 64, 64, 255};
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                white);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   if (depth)
      glEnable(GL_DEPTH_TEST);
   glViewport(0, 0, W, H);
   PFNGLGETQUERYOBJECTUI64VEXTPROC query_result =
      (PFNGLGETQUERYOBJECTUI64VEXTPROC)eglGetProcAddress(
         "glGetQueryObjectui64vEXT");
   bool timer = query_result && strstr((const char *)glGetString(GL_EXTENSIONS),
                                       "GL_EXT_disjoint_timer_query");
   if (getenv("T1192_TIMER") && !atoi(getenv("T1192_TIMER")))
      timer = false;
   unsigned runs = getenv("T1192_RUNS") ? atoi(getenv("T1192_RUNS")) : 9;
   if (!runs || runs > 10000)
      return 2;
   bool check_each =
      getenv("T1192_CHECK_EACH") && atoi(getenv("T1192_CHECK_EACH"));
   unsigned failures = 0, checked = 0;
   bool occlusion =
      getenv("T1192_OCCLUSION") && atoi(getenv("T1192_OCCLUSION"));
   GLuint visible_query = 0;
   if (occlusion)
      glGenQueries(1, &visible_query);
   GLuint query = 0;
   if (timer)
      glGenQueries(1, &query);
   uint8_t *pixels = malloc(W * H * 4);
   if (!pixels)
      return 1;
   for (unsigned mode = 0; mode < 2; ++mode) {
      glDepthFunc(mode ? GL_ALWAYS : GL_NOTEQUAL);
      glDepthMask(GL_TRUE);
      glClearDepthf(0.0f);
      glClear(GL_DEPTH_BUFFER_BIT);
      glDepthMask(GL_FALSE);
      for (unsigned run = 0; run < runs; ++run) {
         glClearColor(1, 0, 0, 1);
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         glFinish();
         double start = now_ms();
         if (timer)
            glBeginQuery(GL_TIME_ELAPSED_EXT, query);
         if (occlusion)
            glBeginQuery(GL_ANY_SAMPLES_PASSED, visible_query);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         if (occlusion)
            glEndQuery(GL_ANY_SAMPLES_PASSED);
         if (timer)
            glEndQuery(GL_TIME_ELAPSED_EXT);
         glFinish();
         printf("depth=%s run=%u cpu_submit_finish_ms=%.6f\n",
                mode ? "always" : "notequal", run, now_ms() - start);
         if (timer) {
            GLuint64 ns;
            query_result(query, GL_QUERY_RESULT, &ns);
            GLint disjoint;
            glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
            printf("depth=%s run=%u gpu_elapsed_ms=%.6f disjoint=%d\n",
                   mode ? "always" : "notequal", run, ns / 1e6, disjoint);
            if (disjoint)
               return 1;
         }
         if (occlusion) {
            GLuint visible = 0;
            glGetQueryObjectuiv(visible_query, GL_QUERY_RESULT, &visible);
            printf("depth=%s run=%u visible=%u expected_visible=%u\n",
                   mode ? "always" : "notequal", run, visible,
                   (unsigned)(mode || !depth));
            failures += !!visible != (mode || !depth);
         }
         if (!check_each && run + 1 != runs)
            continue;
         if (getenv("T1192_READ_DELAY_US")) {
            unsigned us = atoi(getenv("T1192_READ_DELAY_US"));
            struct timespec delay = {us / 1000000, (us % 1000000) * 1000};
            nanosleep(&delay, NULL);
         }
         glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
         unsigned bad = 0;
         uint64_t hash = UINT64_C(14695981039346656037);
         uint8_t mask[W * H];
         for (unsigned i = 0; i < W * H; i++) {
            uint8_t *v = pixels + 4 * i;
            bool survivor = (mode || !depth) && (full || i % W >= W / 2);
            bool wrong = v[0] != (survivor ? 0 : 255) ||
                         v[1] != (survivor ? 255 : 0) || v[3] != 255 ||
                         (!survivor && v[2] != 0);
            bad += wrong;
            mask[i] = wrong ? 255 : 0;
            for (unsigned j = 0; j < 4; j++)
               hash = (hash ^ v[j]) * UINT64_C(1099511628211);
         }
         GLenum error = glGetError();
         checked++;
         failures += bad || error != GL_NO_ERROR;
         printf("depth=%s run=%u bad_pixels=%u fnv1a64=%016llx gl_error=0x%x\n",
                mode ? "always" : "notequal", run, bad,
                (unsigned long long)hash, error);
         if (argc > 1 && (bad || run + 1 == runs)) {
            char path[1024];
            snprintf(path, sizeof(path), "%s-%s-%04u.rgba", argv[1],
                     mode ? "always" : "notequal", run);
            FILE *f = fopen(path, "wb");
            if (!f || fwrite(pixels, 4, W * H, f) != W * H || fclose(f))
               return 1;
            if (bad) {
               snprintf(path, sizeof(path), "%s-%s-%04u-bad.pgm", argv[1],
                        mode ? "always" : "notequal", run);
               f = fopen(path, "wb");
               if (!f || fprintf(f, "P5\n%d %d\n255\n", W, H) < 0 ||
                   fwrite(mask, 1, W * H, f) != W * H || fclose(f))
                  return 1;
            }
         }
         if (bad && getenv("T1192_RECHECK")) {
            for (unsigned retry = 0; retry < 2; ++retry) {
               if (retry) {
                  struct timespec delay = {0, 10000000};
                  nanosleep(&delay, NULL);
               }
               glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
               unsigned again = 0;
               for (unsigned i = 0; i < W * H; i++) {
                  uint8_t *v = pixels + 4 * i;
                  bool survivor = (mode || !depth) && (full || i % W >= W / 2);
                  again += v[0] != (survivor ? 0 : 255) ||
                           v[1] != (survivor ? 255 : 0) || v[3] != 255 ||
                           (!survivor && v[2] != 0);
               }
               printf("depth=%s run=%u retry=%u bad_pixels=%u\n",
                      mode ? "always" : "notequal", run, retry, again);
               if (argc > 1) {
                  char path[1024];
                  snprintf(path, sizeof(path), "%s-%s-%04u-retry%u.rgba",
                           argv[1], mode ? "always" : "notequal", run, retry);
                  FILE *f = fopen(path, "wb");
                  if (!f || fwrite(pixels, 4, W * H, f) != W * H || fclose(f))
                     return 1;
               }
            }
         }
      }
   }
   free(pixels);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(d, ctx);
   eglDestroySurface(d, surf);
   eglTerminate(d);
   printf("checked=%u failed=%u\n", checked, failures);
   return failures ? 1 : 0;
}
