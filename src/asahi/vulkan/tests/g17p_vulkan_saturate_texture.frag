#version 450

layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   vec4 sampled = textureLod(source, uv, 0.0);
   out_color = clamp(sampled * vec4(-1.0, 2.0, 0.5, 0.25), 0.0, 1.0);
}
