#version 450

layout(location = 0) out uvec4 out_value;

void main()
{
   out_value = uvec4(0x1357u, 0xace1u, 0x2468u, 0xfdb9u);
}
