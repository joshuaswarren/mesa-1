#version 450

/* vkQuake's world fragment shape reduced to the part no other G17P texture
 * oracle covers: two distinct combined-image samplers, each sampled once,
 * multiplied together in one invocation. Every existing oracle takes exactly
 * one sample per invocation, so a second sampled texture that quietly reads
 * zero passes all of them while painting every BSP world surface black. */
layout(set = 0, binding = 0) uniform sampler2D tex_a;
layout(set = 0, binding = 1) uniform sampler2D tex_b;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(texture(tex_a, uv).rgb * texture(tex_b, uv).rgb, 1.0);
}
