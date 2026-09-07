#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(location = 0) in vec2 position;
layout(location = 0) out f16vec4 color16;

void main()
{
   gl_Position = vec4(position, 0.0, 1.0);
   color16 = f16vec4(1.0, 0.28, 0.05, 1.0);
}
