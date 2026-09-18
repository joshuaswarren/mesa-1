#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_tracing_position_fetch : require

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attribs;

void main()
{
   vec3 a = gl_HitTriangleVertexPositionsEXT[0];
   vec3 b = gl_HitTriangleVertexPositionsEXT[1];
   vec3 c = gl_HitTriangleVertexPositionsEXT[2];

   payload = floatBitsToUint(a.x + b.y + c.z);
}
