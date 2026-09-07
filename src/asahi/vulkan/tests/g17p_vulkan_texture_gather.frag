#version 450

layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = textureGather(source, vec2(0.5), 0);
}
