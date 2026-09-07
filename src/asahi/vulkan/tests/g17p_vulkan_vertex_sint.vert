#version 450

layout(location = 0) in ivec2 integer_position;

void main()
{
   gl_Position = vec4(vec2(integer_position), 0.0, 1.0);
}
