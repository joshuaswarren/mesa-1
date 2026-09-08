#version 450

layout(set = 0, binding = 0) uniform sampler2D sources[2];

layout(set = 0, binding = 1, std140) uniform CompositeParams {
   uint source_index;
   float opacity;
   vec2 texel_offset;
} params;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = texture(sources[params.source_index],
                       uv + params.texel_offset) * params.opacity;
}
