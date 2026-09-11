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
   /* Mip-1 texel (1,0) is opaque green. The view exposes only mip 1, so
    * explicit LOD 0 exactly matches vk_meta's base-level-relative sample. */
   out_color = textureLod(source, vec3(1.5 / 32.0, 0.5 / 32.0, 0.0), 0.0) *
               params.opacity;
}
