#version 450

/* Identical to g17p_vulkan_two_samplers.frag except that the second sampler
 * lives in its own descriptor set. This is vkQuake's exact world binding
 * shape: set 0 diffuse, set 1 lightmap, both sampled and multiplied. */
layout(set = 0, binding = 0) uniform sampler2D tex_a;
layout(set = 1, binding = 0) uniform sampler2D tex_b;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(texture(tex_a, uv).rgb * texture(tex_b, uv).rgb, 1.0);
}
