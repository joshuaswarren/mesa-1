#version 450

layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) out vec4 out_color;

void main()
{
   ivec2 coord = ivec2(gl_FragCoord.xy) & 15;
   out_color = texelFetch(source, coord, 2);
}
