/* SPDX-License-Identifier: MIT
 * Compiles and links near-limit SSBOs. Does not allocate SSBOs or draw.
 * cc -std=gnu11 -Wall -Wextra -Iinclude ssbo-block-limit.c -ldl -o ssbo-block-limit
 * Usage: ssbo-block-limit /path/to/libEGL.so.1
 * A DRM shim may be used for a CPU-only link regression.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define LOAD_EGL(name) __typeof__(&name) name = dlsym(lib, #name)
#define LOAD_GL(type, name) type name = (type)eglGetProcAddress(#name)

int main(int argc, char **argv)
{
   if (argc != 2)
      return 2;
   void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "%s\n", dlerror());
      return 2;
   }
   LOAD_EGL(eglGetPlatformDisplay);
   LOAD_EGL(eglInitialize);
   LOAD_EGL(eglBindAPI);
   LOAD_EGL(eglCreateContext);
   LOAD_EGL(eglMakeCurrent);
   LOAD_EGL(eglGetProcAddress);
   LOAD_EGL(eglGetError);
   LOAD_EGL(eglDestroyContext);
   LOAD_EGL(eglTerminate);
   EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
   if (!eglInitialize(dpy, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API)) {
      fprintf(stderr, "egl initialize failed: %x\n", eglGetError());
      return 2;
   }
   const EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6,
                           EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      fprintf(stderr, "egl context failed: %x\n", eglGetError());
      return 2;
   }
   LOAD_GL(PFNGLGETSTRINGPROC, glGetString);
   LOAD_GL(PFNGLGETINTEGER64VPROC, glGetInteger64v);
   LOAD_GL(PFNGLCREATESHADERPROC, glCreateShader);
   LOAD_GL(PFNGLSHADERSOURCEPROC, glShaderSource);
   LOAD_GL(PFNGLCOMPILESHADERPROC, glCompileShader);
   LOAD_GL(PFNGLCREATEPROGRAMPROC, glCreateProgram);
   LOAD_GL(PFNGLATTACHSHADERPROC, glAttachShader);
   LOAD_GL(PFNGLPROGRAMPARAMETERIPROC, glProgramParameteri);
   LOAD_GL(PFNGLLINKPROGRAMPROC, glLinkProgram);
   LOAD_GL(PFNGLGETPROGRAMIVPROC, glGetProgramiv);
   LOAD_GL(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog);
   LOAD_GL(PFNGLDELETEPROGRAMPROC, glDeleteProgram);
   LOAD_GL(PFNGLDELETESHADERPROC, glDeleteShader);
   GLint64 limit = 0;
   glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &limit);
   printf("renderer=%s\nssbo_limit=%lld\n", glGetString(GL_RENDERER), (long long)limit);
   const struct { const char *name; unsigned size, alignment; } types[] = {
      {"float", 4, 4}, {"int", 4, 4}, {"uint", 4, 4}, {"double", 8, 8},
      {"ivec2", 8, 8}, {"uvec2", 8, 8}, {"vec2", 8, 8},
      {"ivec3", 12, 16}, {"uvec3", 12, 16}, {"vec3", 12, 16}, {"dvec3", 24, 32}
   };
   /* std140 rounds the block extent to at least 16 bytes; dvec3
    * raises the alignment to 32. Test a usable boundary within the reported
    * limit, including when that limit is INT_MAX rather than a multiple. */
   unsigned block_alignment = 16;
   for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); i++)
      if (types[i].alignment > block_alignment)
         block_alignment = types[i].alignment;
   const uint64_t max_size = (uint64_t)limit & ~(uint64_t)(block_alignment - 1);
   const struct { const char *name; GLenum type, block_limit; const char *preamble; } stages[] = {
      {"compute", GL_COMPUTE_SHADER, GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, "layout(local_size_x=1) in;\n"},
      {"vertex", GL_VERTEX_SHADER, GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, ""},
      {"fragment", GL_FRAGMENT_SHADER, GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, ""},
      {"geometry", GL_GEOMETRY_SHADER, GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, "layout(points) in;layout(points,max_vertices=1) out;\n"},
      {"tess-control", GL_TESS_CONTROL_SHADER, GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, "layout(vertices=1) out;\n"},
      {"tess-eval", GL_TESS_EVALUATION_SHADER, GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, "layout(triangles) in;\n"},
   };
   unsigned checked = 0, failures = 0;
   for (unsigned stage = 0; stage < sizeof(stages) / sizeof(stages[0]); stage++) {
   GLint64 stage_blocks = 0;
   glGetInteger64v(stages[stage].block_limit, &stage_blocks);
   for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
      /* Keep the original valid/unaligned boundary window and add one
       * aligned declaration beyond the actual reported limit. */
      for (unsigned range = 0; range < 2; range++) {
      const uint64_t oversized = ((uint64_t)limit + types[i].alignment - 1) &
                                 ~(uint64_t)(types[i].alignment - 1);
      const uint64_t first = range ? oversized : max_size - 2 * types[i].size;
      const uint64_t last = range ? oversized : max_size - types[i].size;
      for (uint64_t off = first; off <= last; off++) {
         char src[1024];
         snprintf(src, sizeof(src), "#version 460\n%s"
                  "layout(std140,binding=0) buffer Block {layout(offset=%llu) %s member;};\n"
                  "layout(std430,binding=1) buffer Sink {uint result;};\n"
                  "void main(){result=uint(member==%s(1));}\n", stages[stage].preamble,
                  (unsigned long long)off, types[i].name, types[i].name);
         const char *source = src;
         GLuint sh = glCreateShader(stages[stage].type);
         glShaderSource(sh, 1, &source, NULL);
         glCompileShader(sh);
         GLuint prog = glCreateProgram();
         glProgramParameteri(prog, GL_PROGRAM_SEPARABLE, GL_TRUE);
         glAttachShader(prog, sh);
         glLinkProgram(prog);
         GLint linked = 0;
         glGetProgramiv(prog, GL_LINK_STATUS, &linked);
         const int expected = !range && stage_blocks >= 2 &&
                              (off % types[i].alignment) == 0;
         checked++;
         if (!!linked != expected) {
            char log[1024] = {0};
            glGetProgramInfoLog(prog, sizeof(log), NULL, log);
            printf("FAIL stage=%s type=%s offset=%llu linked=%d expected=%d %s\n", stages[stage].name, types[i].name,
                   (unsigned long long)off, linked, expected, log);
            failures++;
         }
         glDeleteProgram(prog);
         glDeleteShader(sh);
      }
      }
   }
   }
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   printf("checked=%u failures=%u\n", checked, failures);
   return failures ? 1 : 0;
}
