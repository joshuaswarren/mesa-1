#version 450

layout(location = 0) out uvec4 out_value;

void main()
{
   out_value = uvec4(0x13u, 0x57u, 0xacu, 0xe1u);
}
