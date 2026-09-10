/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * F52: host-threaded Qt Quick GL 2.1 multi-window oracle.
 *
 * Reuse F51's captured/reconstructed materials, cache-growth sequence,
 * geometry and exact pixel validator. This file adds the one missing factor:
 * three unshared EGL contexts permanently owned by three render threads.
 */

#define _XOPEN_SOURCE 700
#define main f51_embedded_main
#include "test-g17p-gl-qt-multiwindow-glyph-batch.c"
#undef main

#include <pthread.h>

/* asahi_noop.c keeps one process-global mapping vector and mutates it without
 * locking. Real DRM ioctls are safe from multiple client threads, so serialize
 * only entry into the no-op shim; all GL work and driver command recording
 * remains overlapped. */
#ifdef F52_SERIALIZE_NOOP_IOCTL
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/ioctl.h>

typedef int (*ioctl_function)(int, unsigned long, ...);
static ioctl_function next_ioctl;
static pthread_once_t ioctl_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t noop_ioctl_mutex = PTHREAD_MUTEX_INITIALIZER;

static void
resolve_ioctl(void)
{
   *(void **)(&next_ioctl) = dlsym(RTLD_NEXT, "ioctl");
}

int
ioctl(int fd, unsigned long request, ...)
{
   va_list args;
   va_start(args, request);
   void *argument = va_arg(args, void *);
   va_end(args);
   pthread_once(&ioctl_once, resolve_ioctl);
   if (!next_ioctl) {
      errno = ENOSYS;
      return -1;
   }
   pthread_mutex_lock(&noop_ioctl_mutex);
   int result = next_ioctl(fd, request, argument);
   pthread_mutex_unlock(&noop_ioctl_mutex);
   return result;
}
#endif

enum f52_workload {
   THREADED_STATIC,
   THREADED_GROWTH,
};

enum f52_schedule {
   OVERLAPPED,
   SERIALIZED,
};

struct threaded_run {
   EGLDisplay display;
   struct scene scenes[MAX_WINDOWS];
   pthread_barrier_t phase;
   pthread_mutex_t start_mutex;
   pthread_cond_t start_cond;
   bool start;
   bool abort;
   bool result[MAX_WINDOWS];
   enum f52_workload workload;
   enum f52_schedule schedule;
};

struct worker_arg {
   struct threaded_run *run;
   unsigned id;
};

static bool
prepare_frame(struct scene *scene, unsigned generation, bool growth)
{
   release_retired(scene, generation);
   update_native_texture(scene, generation);
   if (!replace_or_update_sdf(scene, 0, generation, growth) ||
       !replace_or_update_sdf(scene, 1, generation, growth))
      return false;
   update_image_texture(scene, generation);

   struct vertex vertices[VERTICES];
   uint16_t indices[INDICES];
   fill_geometry(vertices, indices);
   glBindBuffer(GL_ARRAY_BUFFER, scene->vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), NULL, GL_STREAM_DRAW);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), NULL, GL_STREAM_DRAW);
   glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(indices), indices);
   return gl_ok("threaded resource phase");
}

static bool
draw_frame(struct scene *scene, bool finish)
{
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_BLEND);
   glClearColor(background[0], background[1], background[2], background[3]);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_SCISSOR_TEST);
   glEnable(GL_BLEND);
   draw_batch(scene, 0, scene->native_program, scene->native_texture, true, -1);
   draw_batch(scene, 1, scene->sdf_program, scene->sdf[0].texture, false, 0);
   draw_batch(scene, 2, scene->image_program, scene->image_texture, false, -1);
   draw_batch(scene, 3, scene->sdf_program, scene->sdf[1].texture, false, 1);
   if (finish)
      glFinish();
   else
      glFlush();
   return gl_ok("threaded draw phase");
}

static void
wait_phase(struct threaded_run *run)
{
   int status = pthread_barrier_wait(&run->phase);
   if (status != 0 && status != PTHREAD_BARRIER_SERIAL_THREAD)
      abort();
}

static void *
render_worker(void *opaque)
{
   struct worker_arg *arg = opaque;
   struct threaded_run *run = arg->run;
   struct scene *scene = &run->scenes[arg->id];

   pthread_mutex_lock(&run->start_mutex);
   while (!run->start)
      pthread_cond_wait(&run->start_cond, &run->start_mutex);
   bool stop = run->abort;
   pthread_mutex_unlock(&run->start_mutex);
   if (stop)
      return NULL;

   bool ok = eglMakeCurrent(run->display, scene->surface, scene->surface,
                            scene->context) == EGL_TRUE;
   const bool growth = run->workload == THREADED_GROWTH;

   for (unsigned cycle = 0; cycle < CYCLES; ++cycle) {
      if (run->schedule == OVERLAPPED) {
         /* All QRhi-style threads enter the update phase together. The second
          * barrier guarantees every context has recorded its uploads before
          * their draw submissions are released together. */
         wait_phase(run);
         if (ok)
            ok = prepare_frame(scene, cycle, growth);
         wait_phase(run);
         if (ok)
            ok = draw_frame(scene, false);
         wait_phase(run);
      } else {
         /* Every thread remains alive with its context current, but only the
          * selected window may issue work during each deterministic turn. */
         for (unsigned turn = 0; turn < MAX_WINDOWS; ++turn) {
            wait_phase(run);
            if (arg->id == turn && ok)
               ok = prepare_frame(scene, cycle, growth) &&
                    draw_frame(scene, true);
            wait_phase(run);
         }
      }
   }

   wait_phase(run);
   if (run->schedule == OVERLAPPED) {
      if (ok)
         ok = validate_scene(run->display, scene, CYCLES - 1,
                             getenv("ASAHI_GL_NOOP_SHIM") != NULL);
      wait_phase(run);
   } else {
      for (unsigned turn = 0; turn < MAX_WINDOWS; ++turn) {
         wait_phase(run);
         if (arg->id == turn && ok)
            ok = validate_scene(run->display, scene, CYCLES - 1,
                                getenv("ASAHI_GL_NOOP_SHIM") != NULL);
         wait_phase(run);
      }
   }
   wait_phase(run);

   eglMakeCurrent(run->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                  EGL_NO_CONTEXT);
   run->result[arg->id] = ok;
   return NULL;
}

int
main(int argc, char **argv)
{
   if (argc < 5 || argc > 7) {
      fprintf(stderr, "usage: %s native.vert native.frag sdf.vert sdf.frag "
                      "[--threaded-static|--threaded-growth] "
                      "[--overlapped|--serialized]\n", argv[0]);
      return 2;
   }
   enum f52_workload workload = THREADED_GROWTH;
   if (argc > 5 && strcmp(argv[5], "--threaded-static") == 0)
      workload = THREADED_STATIC;
   else if (argc > 5 && strcmp(argv[5], "--threaded-growth") == 0)
      workload = THREADED_GROWTH;
   else if (argc > 5)
      return 2;
   enum f52_schedule schedule = OVERLAPPED;
   if (argc > 6 && strcmp(argv[6], "--overlapped") == 0)
      schedule = OVERLAPPED;
   else if (argc > 6 && strcmp(argv[6], "--serialized") == 0)
      schedule = SERIALIZED;
   else if (argc > 6)
      return 2;

   char *native_vs = read_file(argv[1]);
   char *native_fs = read_file(argv[2]);
   char *sdf_vs = read_file(argv[3]);
   char *sdf_fs = read_file(argv[4]);
   if (!native_vs || !native_fs || !sdf_vs || !sdf_fs)
      return 2;

   struct threaded_run run = {
      .display = eglGetDisplay(EGL_DEFAULT_DISPLAY),
      .workload = workload,
      .schedule = schedule,
   };
   EGLint major, minor;
   if (run.display == EGL_NO_DISPLAY ||
       !eglInitialize(run.display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_API))
      return 1;
   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(run.display, config_attrs, &config, 1, &count) ||
       count != 1)
      return 1;

   bool ok = true;
   for (unsigned i = 0; i < MAX_WINDOWS; ++i) {
      if (!init_scene(run.display, config, &run.scenes[i], i, native_vs,
                      native_fs, sdf_vs, sdf_fs)) {
         ok = false;
         break;
      }
   }
   free(native_vs);
   free(native_fs);
   free(sdf_vs);
   free(sdf_fs);
   if (!ok)
      return 1;

   const char *version = (const char *)glGetString(GL_VERSION);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("F52 renderer: %s; version: %s; windows: 3; pthreads: 3\n",
          renderer, version);
   fflush(stdout);
   eglMakeCurrent(run.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

   pthread_barrier_init(&run.phase, NULL, MAX_WINDOWS);
   pthread_mutex_init(&run.start_mutex, NULL);
   pthread_cond_init(&run.start_cond, NULL);
   pthread_t threads[MAX_WINDOWS];
   struct worker_arg args[MAX_WINDOWS];
   unsigned created = 0;
   for (; created < MAX_WINDOWS; ++created) {
      args[created] = (struct worker_arg){&run, created};
      if (pthread_create(&threads[created], NULL, render_worker,
                         &args[created]) != 0)
         break;
   }

   pthread_mutex_lock(&run.start_mutex);
   run.abort = created != MAX_WINDOWS;
   run.start = true;
   pthread_cond_broadcast(&run.start_cond);
   pthread_mutex_unlock(&run.start_mutex);
   for (unsigned i = 0; i < created; ++i)
      pthread_join(threads[i], NULL);
   ok = created == MAX_WINDOWS;
   for (unsigned i = 0; i < MAX_WINDOWS; ++i)
      ok = ok && run.result[i];

   pthread_cond_destroy(&run.start_cond);
   pthread_mutex_destroy(&run.start_mutex);
   pthread_barrier_destroy(&run.phase);
   for (unsigned i = 0; i < MAX_WINDOWS; ++i)
      destroy_scene(run.display, &run.scenes[i]);
   eglTerminate(run.display);
#ifdef F52_SERIALIZE_NOOP_IOCTL
   pthread_mutex_destroy(&noop_ioctl_mutex);
#endif

   const char *workload_name = workload == THREADED_GROWTH
      ? "threaded-growth" : "threaded-static";
   const char *schedule_name = schedule == OVERLAPPED
      ? "overlapped" : "serialized";
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   printf("F52 %s %s: %s\n", workload_name, schedule_name,
          noop ? "G17P submission validated; pixels skipped"
               : ok ? "pixels exact (+/-2)" : "FAILED");
   return ok ? 0 : 1;
}
