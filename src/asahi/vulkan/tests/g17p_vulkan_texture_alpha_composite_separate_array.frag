#version 450

/* Match vk_meta blit's binding order and one-layer array view exactly. */
layout(set = 0, binding = 0) uniform sampler source_sampler;
layout(set = 0, binding = 1) uniform texture2DArray source;

layout(set = 0, binding = 2, std140) uniform CompositeParams {
   uint source_index;
   float opacity;
   vec2 texel_offset;
} params;

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint source_bias;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = texture(sampler2DArray(source, source_sampler),
                       vec3(uv + params.texel_offset, 0.0)) * params.opacity;
}
