#version 450

layout(location = 0) in vec2 position;

void main()
{
   if (gl_InstanceIndex == 5)
      gl_Position = vec4(position, 0.0, 1.0);
   else
      gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
}
