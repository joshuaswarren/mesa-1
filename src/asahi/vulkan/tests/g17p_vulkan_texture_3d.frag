#version 450

layout(set = 0, binding = 0) uniform sampler3D source;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = textureLod(source, vec3(0.5, 0.5, 0.75), 1.0);
}
