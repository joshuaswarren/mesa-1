#version 450

/*
 * Fragment half of the Qt 6 HiQ subpixel distance-field text material.
 *
 * This is the shader Omarchy's QuickShell surfaces actually run.  It was
 * recovered from Zink's own SPIR-V, dumped on Neo during the distance-field
 * capture in
 * neo-gpu-harness/logs/qs-full-sdf-capture-20260910-0737.tar.gz (dump07.spv):
 * five implicit-LOD samples of one sampler, four of them
 * OpImageSampleProjImplicitLod, combined by two smoothsteps whose bounds are
 * derived from the taps themselves, and emitted as a per-channel coverage
 * triple for a CONSTANT_COLOR / ONE_MINUS_SRC_COLOR blend.  Its constants are
 * 2.0, 0.67, 0.5, 3.0, 0.333 and 1.0.
 *
 * Variants:
 *   PROBE_TAPS   emit the four projective taps raw, with blending disabled, so
 *                a wrong tap cannot hide inside the coverage arithmetic.
 *   PROBE_CENTER emit the plain non-projective tap raw.  Control for PROBE_TAPS.
 *   MANUAL_PROJ  divide in the shader instead of calling textureProj, so the
 *                projective opcode can be A/B'd against a plain sample of the
 *                identical coordinate.
 *   BREAK_PROJ   negative control: drop the divide.  A correct driver must fail
 *                the reference with this defined; that is what proves the
 *                oracle is sensitive to the tap coordinates at all.
 *   READ_R       read the atlas through .r (identity swizzle) instead of .a
 *                (Zink's GL_ALPHA8 mapping, which is what Qt's atlas gets).
 */

/* Qt's material takes its bounds from a uniform block, not push constants,
 * and the G17P graphics probe's fixed layout is uniform buffer at binding 0
 * plus combined image sampler at binding 1.  Matching both means the shim
 * encode gate and the hardware oracle compile the identical module. */
layout(set = 0, binding = 0, std140) uniform Params {
   float alpha_min;
   float alpha_max;
   float color_w;
} params;

layout(set = 0, binding = 1) uniform sampler2D atlas;

layout(location = 0) in vec2 v_center;
layout(location = 1) in vec3 v_far_left;
layout(location = 2) in vec3 v_near_left;
layout(location = 3) in vec3 v_near_right;
layout(location = 4) in vec3 v_far_right;

layout(location = 0) out vec4 fragColor;

#ifdef READ_R
#define COMPONENT r
#else
#define COMPONENT a
#endif

float
tap(vec3 coord)
{
#if defined(BREAK_PROJ)
   return texture(atlas, coord.xy).COMPONENT;
#elif defined(MANUAL_PROJ)
   return texture(atlas, coord.xy / coord.z).COMPONENT;
#else
   return textureProj(atlas, coord).COMPONENT;
#endif
}

void
main()
{
#if defined(PROBE_CENTER)
   float c = texture(atlas, v_center).COMPONENT;
   fragColor = vec4(c, c, c, c);
#elif defined(PROBE_TAPS)
   fragColor = vec4(tap(v_far_left), tap(v_near_left), tap(v_near_right),
                    tap(v_far_right));
#else
   vec4 n;
   n.x = tap(v_far_left);
   n.y = tap(v_near_left);
   float c = texture(atlas, v_center).COMPONENT;
   n.z = tap(v_near_right);
   n.w = tap(v_far_right);

   vec2 d = min(abs(n.yw - n.xz) * 2.0, 0.67);
   vec2 lo = mix(vec2(params.alpha_min), vec2(0.5), d);
   vec2 hi = mix(vec2(params.alpha_max), vec2(0.5), d);

   n = smoothstep(vec4(lo.x, lo.x, lo.y, lo.y), vec4(hi.x, hi.x, hi.y, hi.y),
                  n);
   c = smoothstep(lo.x + lo.y, hi.x + hi.y, 2.0 * c);

   fragColor = vec4(0.333 * (n.x + n.y + c), 0.333 * (n.y + c + n.z),
                    0.333 * (c + n.z + n.w), c) *
               params.color_w;
#endif
}
