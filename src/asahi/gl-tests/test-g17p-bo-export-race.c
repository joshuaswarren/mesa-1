/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"

#include <c11/threads.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#define THREADS 8

struct export_race {
   struct agx_device *dev;
   struct agx_bo *bo;
   mtx_t lock;
   cnd_t ready_cond;
   cnd_t start_cond;
   unsigned ready;
   bool start;
   int fds[THREADS];
};

struct export_thread {
   struct export_race *race;
   unsigned index;
};

static int
export_thread(void *data)
{
   struct export_thread *thread = data;
   struct export_race *race = thread->race;

   mtx_lock(&race->lock);
   race->ready++;
   cnd_signal(&race->ready_cond);
   while (!race->start)
      cnd_wait(&race->start_cond, &race->lock);
   mtx_unlock(&race->lock);

   race->fds[thread->index] = agx_bo_export(race->dev, race->bo);
   return 0;
}

int
main(void)
{
   const char *node = getenv("ASAHI_GL_RENDER_NODE");
   if (!node)
      node = "/dev/dri/renderD128";

   struct agx_device dev = {
      .fd = open(node, O_RDWR | O_CLOEXEC),
   };
   if (dev.fd < 0 || !agx_open_device(NULL, &dev)) {
      fprintf(stderr, "failed to open shim AGX device\n");
      if (dev.fd >= 0)
         close(dev.fd);
      return 1;
   }

   struct agx_bo *bo = agx_bo_create(&dev, 4096, 4096, AGX_BO_SHAREABLE,
                                     "concurrent export oracle");
   if (!bo)
      return 1;

   struct export_race race = {
      .dev = &dev,
      .bo = bo,
   };
   mtx_init(&race.lock, mtx_plain);
   cnd_init(&race.ready_cond);
   cnd_init(&race.start_cond);

   thrd_t threads[THREADS];
   struct export_thread args[THREADS];
   for (unsigned i = 0; i < THREADS; ++i) {
      race.fds[i] = -1;
      args[i] = (struct export_thread){.race = &race, .index = i};
      if (thrd_create(&threads[i], export_thread, &args[i]) != thrd_success)
         return 1;
   }

   mtx_lock(&race.lock);
   while (race.ready != THREADS)
      cnd_wait(&race.ready_cond, &race.lock);
   race.start = true;
   cnd_broadcast(&race.start_cond);
   mtx_unlock(&race.lock);

   for (unsigned i = 0; i < THREADS; ++i)
      thrd_join(threads[i], NULL);

   bool ok = (bo->flags & AGX_BO_SHARED) && bo->prime_fd >= 0;
   struct stat first = {0};
   for (unsigned i = 0; i < THREADS; ++i) {
      struct stat current = {0};
      ok &= race.fds[i] >= 0 &&
            (fcntl(race.fds[i], F_GETFD, 0) & FD_CLOEXEC) &&
            fstat(race.fds[i], &current) == 0;
      if (i == 0)
         first = current;
      else
         ok &= current.st_dev == first.st_dev && current.st_ino == first.st_ino;
      if (race.fds[i] >= 0)
         close(race.fds[i]);
   }

   cnd_destroy(&race.start_cond);
   cnd_destroy(&race.ready_cond);
   mtx_destroy(&race.lock);
   agx_bo_unreference(&dev, bo);
   agx_close_device(&dev);

   if (!ok) {
      fprintf(stderr, "concurrent BO exports did not share one PRIME fd\n");
      return 1;
   }

   puts("GL07 concurrent BO exports published one shared PRIME lifetime");
   return 0;
}
