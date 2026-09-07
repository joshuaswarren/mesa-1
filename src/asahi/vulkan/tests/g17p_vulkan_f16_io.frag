#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(location = 0) in f16vec4 color16;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(color16);
}
