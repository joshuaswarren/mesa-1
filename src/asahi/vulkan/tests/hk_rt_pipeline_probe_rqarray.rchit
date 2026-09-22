#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

/*
 * The shape dEQP-VK.ray_query.builtin.rayqueryterminate.chit.* compiles: an
 * ARRAY of ray queries, indexed in a loop, with rayQueryTerminateEXT.  A
 * single query would not exercise a deref_array on a ray-query variable.
 */

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevel;

layout(location = 0) rayPayloadInEXT uint payload;
hitAttributeEXT vec2 attribs;

const int nQueries = 2;

void main()
{
   rayQueryEXT rayQueries[nQueries];
   int nProceeds[nQueries];

   for (int i = nQueries - 1; i >= 0; --i) {
      rayQueryInitializeEXT(rayQueries[i], topLevel, 0u, 0xFFu,
                            gl_WorldRayOriginEXT, 0.0001,
                            vec3(0.0, 0.0, -1.0), 9.0);
      nProceeds[i] = 0;
   }

   while (true) {
      int successful = 0;

      for (int i = 0; i < nQueries; ++i) {
         if (rayQueryProceedEXT(rayQueries[i])) {
            nProceeds[i]++;
            successful++;

            if (i != nQueries - 1)
               rayQueryTerminateEXT(rayQueries[i]);
         }
      }

      if (successful == 0)
         break;
   }

   payload = uint(nProceeds[0]) | (uint(nProceeds[1]) << 8);
}
