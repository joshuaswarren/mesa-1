#version 450

layout(set = 0, binding = 0) uniform sampler2DArray source;

layout(set = 0, binding = 1, std140) uniform CompositeParams {
   uint source_index;
   float opacity;
   vec2 texel_offset;
} params;

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint source_bias;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = texture(source, vec3(uv + params.texel_offset, 0.0)) *
               params.opacity;
}
