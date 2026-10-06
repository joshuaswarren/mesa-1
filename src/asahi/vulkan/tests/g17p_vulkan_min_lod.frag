/* SPDX-License-Identifier: MIT */
#version 450

/* Regenerate each checked-in header using glslangValidator -V -DMODE=N
 * --vn g17p_vulkan_min_lod_OP_frag_spv -o g17p_vulkan_min_lod_OP_frag.h
 * g17p_vulkan_min_lod.frag, where N=1..6 and
 * OP=tex,txb,txl,txd,txf0,txf1. */
layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
#if MODE == 1
   out_color = texture(source, uv);
#elif MODE == 2
   out_color = texture(source, uv, -1.0);
#elif MODE == 3
   out_color = textureLod(source, uv, 0.0);
#elif MODE == 4
   out_color = textureGrad(source, uv, vec2(1.0 / 32.0, 0.0),
                                      vec2(0.0, 1.0 / 32.0));
#elif MODE == 5
   out_color = texelFetch(source, ivec2(0), 0);
#elif MODE == 6
   out_color = texelFetch(source, ivec2(0), 1);
#else
#error MODE must be between 1 and 6
#endif
}
