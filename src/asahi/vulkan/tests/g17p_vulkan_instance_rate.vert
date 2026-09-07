#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 transform;

void main()
{
   gl_Position = vec4(position * transform.xy + transform.zw, 0.0, 1.0);
}
