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
   vec2 coord = uv + params.texel_offset;
   vec4 colour = texture(sources[params.source_index], coord);

   /* Retain the About shader's conditional second SMP without changing the
    * compositor oracle at runtime (the fixture supplies source_index == 1). */
   if (params.source_index == 0xffffffffu) {
      vec4 alternate = texture(sources[0], coord);
      colour = mix(alternate, colour, params.opacity);
   }

   out_color = colour * params.opacity;
}
