#version 450

layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   const vec2 dx = vec2(1.0 / 32.0, 0.0);
   const vec2 dy = vec2(0.0, 1.0 / 32.0);
   out_color = textureGrad(source, uv, dx, dy);
}
