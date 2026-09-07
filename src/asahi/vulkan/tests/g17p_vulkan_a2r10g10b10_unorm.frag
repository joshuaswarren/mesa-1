#version 450

layout(location = 0) out vec4 out_value;

void main()
{
   out_value = vec4(341.0 / 1023.0, 682.0 / 1023.0,
                    939.0 / 1023.0, 2.0 / 3.0);
}
