#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attribs;

void main()
{
   vec3 sum = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT +
              gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT;

   mat4x3 o2w = gl_ObjectToWorldEXT;
   mat4x3 w2o = gl_WorldToObjectEXT;
   sum += o2w[3] + w2o[3] + o2w[0] + w2o[1];

   float scalars = gl_RayTminEXT + gl_RayTmaxEXT + gl_HitTEXT + attribs.x;

   uint ints = uint(gl_PrimitiveID) ^ uint(gl_InstanceID) ^
               uint(gl_InstanceCustomIndexEXT) ^ uint(gl_GeometryIndexEXT) ^
               gl_HitKindEXT ^ gl_IncomingRayFlagsEXT;

   payload = ints ^ floatBitsToUint(sum.x + sum.y + sum.z + scalars);
}
