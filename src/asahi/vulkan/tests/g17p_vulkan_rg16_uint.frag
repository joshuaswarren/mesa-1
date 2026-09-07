#version 450

layout(location = 0) out uvec2 out_value;

void main()
{
   out_value = uvec2(0x1357u, 0xace1u);
}
