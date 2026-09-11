#version 450

/*
 * Vertex half of the Qt 6 HiQ subpixel distance-field text material, reduced to
 * a fullscreen triangle with exactly predictable sample coordinates.
 *
 * Qt's own vertex shader derives the four neighbour taps from the item-space
 * transform; that arithmetic is host-side and is not what the driver has to get
 * right.  What the driver has to get right is the shape it is handed: one plain
 * two-component coordinate plus four three-component projective coordinates,
 * all consumed by implicit-LOD samples of one sampler in one fragment shader.
 *
 * The offsets here are whole texels and the projector is exactly 2.0, so every
 * tap lands on a texel centre and the projective divide is exact in binary32.
 * A LINEAR sampler therefore returns the texel itself and the expected image is
 * computable on the host without reimplementing filtering.
 */

#ifndef TEX_SIZE
#define TEX_SIZE 32.0
#endif

/* Exactly representable, and not 1.0: dropping the divide moves every tap. */
#ifndef PROJECTOR
#define PROJECTOR 2.0
#endif

layout(location = 0) out vec2 v_center;
layout(location = 1) out vec3 v_far_left;
layout(location = 2) out vec3 v_near_left;
layout(location = 3) out vec3 v_near_right;
layout(location = 4) out vec3 v_far_right;

vec3
projective(vec2 uv, float texels)
{
   return vec3((uv + vec2(texels / TEX_SIZE, 0.0)) * PROJECTOR, PROJECTOR);
}

void
main()
{
   const vec2 positions[3] =
      vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
   vec2 position = positions[gl_VertexIndex];
   vec2 uv = position * 0.5 + 0.5;

   gl_Position = vec4(position, 0.0, 1.0);
   v_center = uv;
   v_far_left = projective(uv, -2.0);
   v_near_left = projective(uv, -1.0);
   v_near_right = projective(uv, 1.0);
   v_far_right = projective(uv, 2.0);
}
