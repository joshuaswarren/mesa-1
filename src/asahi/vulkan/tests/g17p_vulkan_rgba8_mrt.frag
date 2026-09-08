#version 450

layout(location = 0) out vec4 out_color0;
layout(location = 1) out vec4 out_color1;

void main()
{
   out_color0 = vec4(0.25, 0.125, 0.5, 0.25);
   out_color1 = vec4(0.125, 0.25, 0.0625, 0.5);
}
