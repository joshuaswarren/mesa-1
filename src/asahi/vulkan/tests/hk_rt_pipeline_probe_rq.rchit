#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevel;

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attribs;

void main()
{
   rayQueryEXT rq;
   rayQueryInitializeEXT(rq, topLevel, gl_RayFlagsOpaqueEXT, 0xFFu,
                         gl_WorldRayOriginEXT, 0.0, vec3(0.0, 0.0, 1.0), 10.0);

   while (rayQueryProceedEXT(rq)) { }

   payload = uint(rayQueryGetIntersectionTypeEXT(rq, true));
}
