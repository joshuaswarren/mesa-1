#version 450

layout(location = 0) in vec4 packed_position;

void main()
{
   gl_Position = vec4(packed_position.xy * 4.0 - 1.0, 0.0, 1.0);
}
