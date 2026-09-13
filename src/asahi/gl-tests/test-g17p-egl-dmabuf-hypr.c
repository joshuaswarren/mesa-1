/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Hyprland's direct ABGR8888 dma-buf -> EGLImage -> texture sampling path.
 */

#define main gl06_unused_main
#include "test-g17p-egl-dmabuf-image.c"
#undef main
#include <vulkan/vulkan.h>

static void
vk_check(VkResult result, const char *where)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "GL08 %s returned %d\n", where, result);
      exit(1);
   }
}

static uint32_t
vk_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
               uint32_t bits, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   fprintf(stderr, "GL08 no exportable host-visible memory type\n");
   exit(1);
}

static GLuint
compile(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok = 0;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048];
      GLsizei length = 0;
      glGetShaderInfoLog(shader, sizeof(log), &length, log);
      fprintf(stderr, "GL08 shader compile failed: %.*s\n", length, log);
      return 0;
   }
   return shader;
}

static bool
abgr_linear_supported(EGLDisplay display,
                      PFNEGLQUERYDMABUFFORMATSEXTPROC query_formats,
                      PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers)
{
   EGLint count = 0;
   if (!query_formats(display, 0, NULL, &count) || count <= 0)
      return false;
   EGLint *formats = calloc((size_t)count, sizeof(*formats));
   if (!formats || !query_formats(display, count, formats, &count))
      return false;
   bool found = false;
   for (EGLint i = 0; i < count; ++i)
      found |= formats[i] == DRM_FORMAT_ABGR8888;
   free(formats);
   if (!found)
      return false;

   count = 0;
   if (!query_modifiers(display, DRM_FORMAT_ABGR8888, 0, NULL, NULL,
                        &count) || count <= 0)
      return false;
   EGLuint64KHR *modifiers = calloc((size_t)count, sizeof(*modifiers));
   EGLBoolean *external = calloc((size_t)count, sizeof(*external));
   if (!modifiers || !external ||
       !query_modifiers(display, DRM_FORMAT_ABGR8888, count, modifiers,
                        external, &count))
      return false;
   found = false;
   for (EGLint i = 0; i < count; ++i)
      found |= modifiers[i] == DRM_FORMAT_MOD_LINEAR;
   free(external);
   free(modifiers);
   return found;
}

static bool
read_exact(const char *phase, const uint8_t *expected, bool noop)
{
   struct guarded_pixels got;
   memset(&got, GUARD_BYTE, sizeof(got));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, got.data);
   if (!gl_ok(phase) || !guard_ok(got.before) || !guard_ok(got.after))
      return false;
   if (noop || memcmp(got.data, expected, DATA_BYTES) == 0) {
      fprintf(stderr, "GL08 %s: %s\n", phase,
              noop ? "state exact (no-op pixels skipped)" : "pixels exact");
      return true;
   }
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
      if (memcmp(&got.data[4 * i], &expected[4 * i], 4)) {
         fprintf(stderr, "GL08 %s first mismatch %u: got %02x%02x%02x%02x expected %02x%02x%02x%02x\n",
                 phase, i, got.data[4*i], got.data[4*i+1], got.data[4*i+2],
                 got.data[4*i+3], expected[4*i], expected[4*i+1],
                 expected[4*i+2], expected[4*i+3]);
         break;
      }
   }
   return false;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;

   /* Model the client side exactly: Honeykrisp exports a linear Vulkan RGBA8
    * image as a dma-buf, then Hyprland's Zink EGL display imports ABGR8888. */
   const VkApplicationInfo vk_app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "GL08 dma-buf producer",
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo vk_instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &vk_app,
   };
   VkInstance vk_instance;
   vk_check(vkCreateInstance(&vk_instance_info, NULL, &vk_instance),
            "vkCreateInstance");
   uint32_t vk_physical_count = 0;
   vk_check(vkEnumeratePhysicalDevices(vk_instance, &vk_physical_count, NULL),
            "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *vk_physicals =
      calloc(vk_physical_count, sizeof(*vk_physicals));
   vk_check(vkEnumeratePhysicalDevices(vk_instance, &vk_physical_count,
                                       vk_physicals),
            "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice vk_physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < vk_physical_count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(vk_physicals[i], &props);
      if (props.deviceID == 0x8140) {
         vk_physical = vk_physicals[i];
         break;
      }
   }
   free(vk_physicals);
   if (!vk_physical)
      return 1;
   const char *vk_extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
   const VkDeviceCreateInfo vk_device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = vk_extensions,
   };
   VkDevice vk_device;
   vk_check(vkCreateDevice(vk_physical, &vk_device_info, NULL, &vk_device),
            "vkCreateDevice");
   const VkExternalMemoryImageCreateInfo external_image = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const VkImageCreateInfo vk_image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external_image,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
   };
   VkImage vk_source;
   vk_check(vkCreateImage(vk_device, &vk_image_info, NULL, &vk_source),
            "vkCreateImage(export)");
   VkMemoryRequirements vk_requirements;
   vkGetImageMemoryRequirements(vk_device, vk_source, &vk_requirements);
   VkPhysicalDeviceMemoryProperties vk_memory_properties;
   vkGetPhysicalDeviceMemoryProperties(vk_physical, &vk_memory_properties);
   const VkExportMemoryAllocateInfo export_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const VkMemoryAllocateInfo vk_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = vk_requirements.size,
      .memoryTypeIndex = vk_memory_type(
         &vk_memory_properties, vk_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory vk_memory;
   vk_check(vkAllocateMemory(vk_device, &vk_allocation, NULL, &vk_memory),
            "vkAllocateMemory(export)");
   vk_check(vkBindImageMemory(vk_device, vk_source, vk_memory, 0),
            "vkBindImageMemory(export)");
   const VkImageSubresource subresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
   };
   VkSubresourceLayout vk_layout;
   vkGetImageSubresourceLayout(vk_device, vk_source, &subresource, &vk_layout);
   void *vk_map;
   vk_check(vkMapMemory(vk_device, vk_memory, 0, VK_WHOLE_SIZE, 0, &vk_map),
            "vkMapMemory(export)");
   setenv("MESA_LOADER_DRIVER_OVERRIDE", "zink", 1);

   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor))
      return 1;
   const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
   if (!extensions ||
       !has_extension(extensions, "EGL_EXT_image_dma_buf_import") ||
       !has_extension(extensions, "EGL_EXT_image_dma_buf_import_modifiers") ||
       !has_extension(extensions, "EGL_KHR_image_base"))
      return 1;
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
   PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture =
      (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress(
         "glEGLImageTargetTexture2DOES");
   if (!create_image || !destroy_image || !query_formats ||
       !query_modifiers || !image_target_texture) {
      fprintf(stderr, "GL08 required EGLImage entrypoint missing\n");
      return 1;
   }
   if (!abgr_linear_supported(display, query_formats, query_modifiers)) {
      fprintf(stderr, "GL08 ABGR8888+LINEAR not advertised for dma-buf import\n");
      return 1;
   }

   if (!eglBindAPI(EGL_OPENGL_ES_API))
      return 1;
   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8, EGL_NONE,
   };
   EGLConfig config;
   EGLint count = 0;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &count) || count != 1)
      return 1;
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
      return 1;

   struct guarded_pixels upload;
   memset(&upload, GUARD_BYTE, sizeof(upload));
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const uint8_t left[4] = {0x11, 0x22, 0x33, 0xff};
         const uint8_t right[4] = {0xaa, 0xbb, 0xcc, 0xff};
         memcpy(&upload.data[(y * WIDTH + x) * 4],
                x < WIDTH / 2 ? left : right, 4);
      }
   }
   for (unsigned y = 0; y < HEIGHT; ++y)
      memcpy((uint8_t *)vk_map + vk_layout.offset + y * vk_layout.rowPitch,
             upload.data + y * ROW_BYTES, ROW_BYTES);
   for (unsigned y = 0; y < HEIGHT; ++y) {
      if (memcmp((uint8_t *)vk_map + vk_layout.offset + y * vk_layout.rowPitch,
                 upload.data + y * ROW_BYTES, ROW_BYTES) != 0) {
         fprintf(stderr, "GL08 Vulkan linear source mismatch before export\n");
         return 1;
      }
   }
   fprintf(stderr, "GL08 Vulkan linear source before export: pixels exact\n");
   PFN_vkGetMemoryFdKHR get_memory_fd =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk_device,
                                                "vkGetMemoryFdKHR");
   if (!get_memory_fd)
      return 1;
   const VkMemoryGetFdInfoKHR fd_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = vk_memory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   int fd = -1;
   vk_check(get_memory_fd(vk_device, &fd_info, &fd), "vkGetMemoryFdKHR");
   const uint32_t stride = (uint32_t)vk_layout.rowPitch;
   const uint32_t offset = (uint32_t)vk_layout.offset;
   const uint64_t modifier = DRM_FORMAT_MOD_LINEAR;
   const EGLint attrs[] = {
      EGL_WIDTH, WIDTH,
      EGL_HEIGHT, HEIGHT,
      EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ABGR8888,
      EGL_DMA_BUF_PLANE0_FD_EXT, fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)offset,
      EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)modifier,
      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(modifier >> 32),
      EGL_NONE,
   };
   EGLImageKHR image = create_image(display, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
   if (image == EGL_NO_IMAGE_KHR) {
      fprintf(stderr, "GL08 eglCreateImageKHR ABGR8888 import failed: 0x%x\n",
              eglGetError());
      return 1;
   }
   GLuint imported_texture;
   glGenTextures(1, &imported_texture);
   glBindTexture(GL_TEXTURE_2D, imported_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   image_target_texture(GL_TEXTURE_2D, image);
   if (!gl_ok("ABGR8888 EGLImage bind") || !destroy_image(display, image))
      return 1;
   close(fd);

   static const char vertex_source[] =
      "attribute vec2 pos; attribute vec2 uv; varying vec2 tc;"
      "void main(){gl_Position=vec4(pos,0.0,1.0);tc=uv;}";
   static const char fragment_source_broadcast[] =
      "precision highp float; varying vec2 tc; uniform sampler2D tex;"
      "void main(){gl_FragColor=texture2D(tex,tc);}";
   static const char fragment_source_rt0[] =
      "#extension GL_EXT_draw_buffers : require\n"
      "precision highp float; varying vec2 tc; uniform sampler2D tex;"
      "void main(){gl_FragData[0]=texture2D(tex,tc);}";
   const char *fragment_source = getenv("F38_SINGLE_RT0")
                                    ? fragment_source_rt0
                                    : fragment_source_broadcast;
   GLuint vs = compile(GL_VERTEX_SHADER, vertex_source);
   GLuint fs = compile(GL_FRAGMENT_SHADER, fragment_source);
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "pos");
   glBindAttribLocation(program, 1, "uv");
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!vs || !fs || !linked)
      return 1;
   GLuint native_texture;
   glGenTextures(1, &native_texture);
   glBindTexture(GL_TEXTURE_2D, native_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, upload.data);
   static const GLfloat vertices[] = {
      -1, -1, 0, 0,  1, -1, 1, 0,
      -1,  1, 0, 1,  1,  1, 1, 1,
   };
   glViewport(0, 0, WIDTH, HEIGHT);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                         vertices);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                         vertices + 2);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glBindTexture(GL_TEXTURE_2D, native_texture);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   bool native_sample_ok =
      read_exact("ordinary GL texture sample", upload.data, noop);
   glBindTexture(GL_TEXTURE_2D, imported_texture);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   bool imported_sample_ok =
      read_exact("imported EGLImage texture sample", upload.data, noop);

   /* Reading the imported storage as an attachment bypasses the sampler and
    * descriptor lowering. Do it only after the original direct-sample repro
    * so an attachment transition cannot accidentally repair the first draw. */
   GLuint import_fbo;
   glGenFramebuffers(1, &import_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, import_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          imported_texture, 0);
   bool imported_attachment_ok =
      glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE &&
      read_exact("imported attachment read", upload.data, noop);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);

   glBindTexture(GL_TEXTURE_2D, imported_texture);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   sleep(3);
   bool retained_sample_ok =
      read_exact("imported EGLImage retained sample", upload.data, noop);
   if (!native_sample_ok || !imported_sample_ok || !imported_attachment_ok ||
       !retained_sample_ok)
      return 1;

   puts(noop ? "GL08 Hypr ABGR8888 EGLImage direct-sample state exact; no-op pixels skipped"
             : "GL08 Hypr ABGR8888 EGLImage direct-sample pixels exact immediately/3s");
   return 0;
}
