/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * QuickShell/Wayland ARGB8888 client-buffer discriminator. A Vulkan
 * B8G8R8A8_UNORM linear image is exported as DRM_FORMAT_ARGB8888, imported as
 * an EGLImage, sampled before any attachment transition, and premultiplied-
 * composited over an opaque target.
 */

#define main gl06_unused_main
#include "test-g17p-egl-dmabuf-image.c"
#undef main

#include <vulkan/vulkan.h>

static void
vk_check_argb(VkResult result, const char *where)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "GL41 %s returned %d\n", where, result);
      exit(EXIT_FAILURE);
   }
}

static uint32_t
vk_memory_type_argb(const VkPhysicalDeviceMemoryProperties *properties,
                    uint32_t bits, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   fprintf(stderr, "GL41 no exportable host-visible memory type\n");
   exit(EXIT_FAILURE);
}

static GLuint
compile_argb(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint compiled = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
   if (!compiled) {
      char log[2048] = {0};
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "GL41 shader compile failed: %s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static bool
near_argb(uint8_t actual, uint8_t expected)
{
   return actual + 1 >= expected && actual <= expected + 1;
}

static bool
read_argb(const char *phase, bool final, bool noop)
{
   struct guarded_pixels pixels;
   memset(&pixels, GUARD_BYTE, sizeof(pixels));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data);
   const bool state_ok = gl_ok(phase) && guard_ok(pixels.before) &&
                         guard_ok(pixels.after);
   if (!state_ok || noop)
      return state_ok;

   static const uint8_t sampled[2][4] = {
      {0, 64, 128, 128},
      {128, 32, 0, 128},
   };
   static const uint8_t composited[2][4] = {
      {32, 128, 223, 255},
      {160, 96, 95, 255},
   };
   const uint8_t(*expected)[4] = final ? composited : sampled;

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const uint8_t *got = &pixels.data[(y * WIDTH + x) * 4];
         const uint8_t *want = expected[x >= WIDTH / 2];
         for (unsigned c = 0; c < 4; ++c) {
            if (!near_argb(got[c], want[c])) {
               fprintf(stderr,
                       "GL41 %s (%u,%u) got %02x%02x%02x%02x, expected "
                       "%02x%02x%02x%02x (+/-1)\n",
                       phase, x, y, got[0], got[1], got[2], got[3], want[0],
                       want[1], want[2], want[3]);
               return false;
            }
         }
      }
   }

   fprintf(stderr, "GL41 %s: exact ARGB8888 regions (+/-1)\n", phase);
   return true;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "GL41 ARGB8888 client producer",
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   vk_check_argb(vkCreateInstance(&instance_info, NULL, &instance),
                 "vkCreateInstance");
   uint32_t physical_count = 0;
   vk_check_argb(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
                 "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physicals = calloc(physical_count, sizeof(*physicals));
   if (!physicals)
      return EXIT_FAILURE;
   vk_check_argb(vkEnumeratePhysicalDevices(instance, &physical_count,
                                             physicals),
                 "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < physical_count; ++i) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(physicals[i], &properties);
      if (properties.deviceID == 0x8140) {
         physical = physicals[i];
         break;
      }
   }
   free(physicals);
   if (!physical) {
      fprintf(stderr, "GL41 G17P Vulkan device missing\n");
      return EXIT_FAILURE;
   }

   const char *extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   vk_check_argb(vkCreateDevice(physical, &device_info, NULL, &device),
                 "vkCreateDevice");
   const VkExternalMemoryImageCreateInfo external_image = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external_image,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_B8G8R8A8_UNORM,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
   };
   VkImage source;
   vk_check_argb(vkCreateImage(device, &image_info, NULL, &source),
                 "vkCreateImage");
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, source, &requirements);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   const VkExportMemoryAllocateInfo export_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const VkMemoryAllocateInfo allocation_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = requirements.size,
      .memoryTypeIndex = vk_memory_type_argb(
         &memory_properties, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   vk_check_argb(vkAllocateMemory(device, &allocation_info, NULL, &memory),
                 "vkAllocateMemory");
   vk_check_argb(vkBindImageMemory(device, source, memory, 0),
                 "vkBindImageMemory");
   const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
   };
   VkSubresourceLayout layout;
   vkGetImageSubresourceLayout(device, source, &subresource, &layout);
   uint8_t *map;
   vk_check_argb(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0,
                             (void **)&map),
                 "vkMapMemory");

   struct guarded_pixels upload;
   memset(&upload, GUARD_BYTE, sizeof(upload));
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         /* Raw ARGB8888 storage is BGRA byte order on little endian. */
         static const uint8_t raw[2][4] = {
            {128, 64, 0, 128},
            {0, 32, 128, 128},
         };
         memcpy(&upload.data[(y * WIDTH + x) * 4], raw[x >= WIDTH / 2], 4);
      }
      memcpy(map + layout.offset + y * layout.rowPitch,
             upload.data + y * ROW_BYTES, ROW_BYTES);
      if (memcmp(map + layout.offset + y * layout.rowPitch,
                 upload.data + y * ROW_BYTES, ROW_BYTES)) {
         fprintf(stderr, "GL41 Vulkan BGRA8 source mismatch before export\n");
         return EXIT_FAILURE;
      }
   }
   if (!guard_ok(upload.before) || !guard_ok(upload.after))
      return EXIT_FAILURE;
   fprintf(stderr, "GL41 Vulkan BGRA8 source before export: bytes exact\n");

   PFN_vkGetMemoryFdKHR get_memory_fd =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
   if (!get_memory_fd)
      return EXIT_FAILURE;
   const VkMemoryGetFdInfoKHR fd_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = memory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   int fd = -1;
   vk_check_argb(get_memory_fd(device, &fd_info, &fd), "vkGetMemoryFdKHR");

   setenv("MESA_LOADER_DRIVER_OVERRIDE", "zink", 1);
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor))
      return EXIT_FAILURE;
   const char *egl_extensions = eglQueryString(display, EGL_EXTENSIONS);
   if (!egl_extensions ||
       !has_extension(egl_extensions, "EGL_EXT_image_dma_buf_import") ||
       !has_extension(egl_extensions,
                      "EGL_EXT_image_dma_buf_import_modifiers") ||
       !has_extension(egl_extensions, "EGL_KHR_image_base"))
      return EXIT_FAILURE;
   PFNEGLCREATEIMAGEKHRPROC create_image =
      (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   PFNEGLDESTROYIMAGEKHRPROC destroy_image =
      (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
   PFNEGLQUERYDMABUFFORMATSEXTPROC query_formats =
      (PFNEGLQUERYDMABUFFORMATSEXTPROC)eglGetProcAddress(
         "eglQueryDmaBufFormatsEXT");
   PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers =
      (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress(
         "eglQueryDmaBufModifiersEXT");
   PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target =
      (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress(
         "glEGLImageTargetTexture2DOES");
   if (!create_image || !destroy_image || !query_formats || !query_modifiers ||
       !image_target ||
       !format_and_modifier_supported(display, query_formats, query_modifiers)) {
      fprintf(stderr, "GL41 ARGB8888+LINEAR import support missing\n");
      return EXIT_FAILURE;
   }

   if (!eglBindAPI(EGL_OPENGL_ES_API))
      return EXIT_FAILURE;
   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return EXIT_FAILURE;
   const EGLint surface_attrs[] = {
      EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE,
   };
   const EGLint context_attrs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE,
   };
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      return EXIT_FAILURE;

   const EGLint image_attrs[] = {
      EGL_WIDTH, WIDTH,
      EGL_HEIGHT, HEIGHT,
      EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
      EGL_DMA_BUF_PLANE0_FD_EXT, fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)layout.offset,
      EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)layout.rowPitch,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)DRM_FORMAT_MOD_LINEAR,
      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, 0,
      EGL_NONE,
   };
   EGLImageKHR image = create_image(display, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, image_attrs);
   if (image == EGL_NO_IMAGE_KHR) {
      fprintf(stderr, "GL41 ARGB8888 EGLImage import failed: %#x\n",
              eglGetError());
      return EXIT_FAILURE;
   }
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   image_target(GL_TEXTURE_2D, image);
   if (!gl_ok("ARGB8888 EGLImage bind") || !destroy_image(display, image))
      return EXIT_FAILURE;
   close(fd);

   static const char vertex_source[] =
      "attribute vec2 pos;attribute vec2 uv;varying vec2 tc;"
      "void main(){gl_Position=vec4(pos,0.0,1.0);tc=uv;}";
   static const char fragment_source[] =
      "precision highp float;varying vec2 tc;uniform sampler2D tex;"
      "void main(){gl_FragColor=texture2D(tex,tc);}";
   GLuint vertex = compile_argb(GL_VERTEX_SHADER, vertex_source);
   GLuint fragment = compile_argb(GL_FRAGMENT_SHADER, fragment_source);
   GLuint program = glCreateProgram();
   glAttachShader(program, vertex);
   glAttachShader(program, fragment);
   glBindAttribLocation(program, 0, "pos");
   glBindAttribLocation(program, 1, "uv");
   glLinkProgram(program);
   GLint linked = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!vertex || !fragment || !linked)
      return EXIT_FAILURE;
   static const GLfloat vertices[] = {
      -1, -1, 0, 0, 1, -1, 1, 0,
      -1, 1, 0, 1, 1, 1, 1, 1,
   };
   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_DITHER);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                         vertices);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                         vertices + 2);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   /* Sample first so an attachment transition cannot repair the repro. */
   glDisable(GL_BLEND);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   const bool direct_ok = read_argb("ARGB8888 direct sample", false, noop);

   GLuint imported_fbo;
   glGenFramebuffers(1, &imported_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, imported_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          texture, 0);
   const bool attachment_ok =
      glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE &&
      read_argb("ARGB8888 attachment control", false, noop);

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_BLEND);
   glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
   glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                       GL_ONE_MINUS_SRC_ALPHA);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   const bool composite_ok =
      read_argb("ARGB8888 premultiplied composite immediate", true, noop);
   sleep(3);
   const bool retained_ok =
      read_argb("ARGB8888 premultiplied composite retained", true, noop);

   if (!direct_ok || !attachment_ok || !composite_ok || !retained_ok)
      return EXIT_FAILURE;

   puts(noop ? "GL41 ARGB8888 client import submitted; pixels skipped"
             : "GL41 ARGB8888 client import/sample/composite exact "
               "immediately/3s");
   return EXIT_SUCCESS;
}
