#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint payload;

layout(shaderRecordEXT) buffer Record { uint tag; uint mask; } record;

void main()
{
   payload = record.tag ^ 0x4D495353u;
}
