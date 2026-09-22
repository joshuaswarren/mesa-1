#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attribs;

void main()
{
   payload = 0x48495400u | uint(gl_PrimitiveID & 0xFF);
}
