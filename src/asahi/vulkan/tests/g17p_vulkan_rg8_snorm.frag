#version 450

layout(location = 0) out vec2 out_value;

void main()
{
   out_value = vec2(-64.0 / 127.0, 17.0 / 127.0);
}
