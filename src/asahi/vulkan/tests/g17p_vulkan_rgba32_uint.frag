#version 450

layout(location = 0) out uvec4 out_value;

void main()
{
   out_value = uvec4(0x89abcdefu, 0x01234567u, 0xdeadbeefu, 0x76543210u);
}
