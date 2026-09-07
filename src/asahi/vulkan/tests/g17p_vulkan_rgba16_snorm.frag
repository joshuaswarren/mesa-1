#version 450

layout(location = 0) out vec4 out_value;

void main()
{
   out_value = vec4(-16384.0 / 32767.0, 4097.0 / 32767.0,
                    24576.0 / 32767.0, -8193.0 / 32767.0);
}
