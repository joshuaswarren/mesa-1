#version 450

layout(location = 0) out vec4 out_value;

void main()
{
   out_value = vec4(-64.0 / 127.0, 17.0 / 127.0,
                    96.0 / 127.0, -33.0 / 127.0);
}
