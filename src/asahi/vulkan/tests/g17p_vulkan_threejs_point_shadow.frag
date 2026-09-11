#version 450

/*
 * three.js's point-light shadow lookup, the fragment shape that made
 * vkCreateGraphicsPipelines return VK_ERROR_FEATURE_NOT_PRESENT under
 * Chromium/ANGLE.
 *
 * cubeToUV() mixes a 32-bit component with the half-precision texel-size term,
 * which reaches the packer as an fcmpsel with one 16-bit comparator sitting on
 * an odd half-register. agx3_alu_fcmp_select() pinned both comparators to
 * 32-bit and even registers, so the whole shader failed closed and every draw
 * that needed it was dropped: shadowmap_pointlight rendered blank.
 *
 * Reproduce with no hardware, against the G17P drm-shim:
 *
 *   glslangValidator -V g17p_vulkan_threejs_point_shadow.frag -o shadow.spv
 *   ASAHI_SHIM_CHIP=g17p ASAHI_SHIM_VALIDATE_G17P_STREAM=1 \
 *   LD_PRELOAD=<build>/src/asahi/drm-shim/libasahi_noop_drm_shim.so \
 *   VK_DRIVER_FILES=<build>/src/asahi/vulkan/asahi_devenv_icd.aarch64.json \
 *   AGX_MESA_DEBUG=shaders \
 *   <build>/src/asahi/vulkan/g17p-vulkan-graphics-probe vert.spv shadow.spv
 *
 * The encoder-level regression lives in
 * test_fcmp_select_half_comparator() (test-agx3-alu-core.c).
 */

layout(set = 0, binding = 1) uniform sampler2D shadowMap;
layout(set = 0, binding = 5) uniform ShadowParams {
   vec3 lightPos;
   float shadowBias;
   float shadowRadius;
   float shadowCameraNear;
   float shadowCameraFar;
} u;

layout(location = 4) in vec3 vWorldPosition;
layout(location = 0) out vec4 fragColor;

const float UnpackDownscale = 255.0 / 256.0;
const vec4 UnpackFactors =
   UnpackDownscale / vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0);

float unpackRGBAToDepth(const in vec4 v)
{
   return dot(v, UnpackFactors);
}

float texture2DCompare(sampler2D depths, vec2 uv, float compare)
{
   return step(compare, unpackRGBAToDepth(texture(depths, uv)));
}

vec2 cubeToUV(vec3 v, float texelSizeY)
{
   vec3 absV = abs(v);
   float scaleToCube = 1.0 / max(absV.x, max(absV.y, absV.z));
   absV *= scaleToCube;
   v *= scaleToCube * (1.0 - 2.0 * texelSizeY);

   vec2 planar = v.xy;
   float almostATexel = 1.5 * texelSizeY;
   float almostOne = 1.0 - almostATexel;

   if (absV.z >= almostOne) {
      if (v.z > 0.0)
         planar.x = 4.0 - v.x;
   } else if (absV.x >= almostOne) {
      float signX = sign(v.x);
      planar.x = v.z * signX + 2.0 * signX;
   } else if (absV.y >= almostOne) {
      float signY = sign(v.y);
      planar.x = v.x + 2.0 * signY + 2.0;
      planar.y = v.z * signY - 2.0;
   }

   return vec2(0.125, 0.25) * planar + vec2(0.375, 0.75);
}

void main()
{
   vec3 lightToPosition = vWorldPosition - u.lightPos;
   float dp = (length(lightToPosition) - u.shadowCameraNear) /
              (u.shadowCameraFar - u.shadowCameraNear);
   dp += u.shadowBias;

   vec3 bd3D = normalize(lightToPosition);
   vec2 texelSize = vec2(1.0) / (vec2(1024.0, 512.0) * vec2(4.0, 2.0));
   vec2 offset = vec2(-1.0, 1.0) * u.shadowRadius * texelSize.y;

   float shadow =
      (texture2DCompare(shadowMap, cubeToUV(bd3D + offset.xyy, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.yyy, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.xyx, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.yyx, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.xxy, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.yxy, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.xxx, texelSize.y), dp) +
       texture2DCompare(shadowMap, cubeToUV(bd3D + offset.yxx, texelSize.y), dp)) *
      (1.0 / 9.0);

   fragColor = vec4(vec3(shadow), 1.0);
}
