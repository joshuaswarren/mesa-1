#version 460
#extension GL_EXT_shader_quad_control : require
#extension GL_KHR_shader_subgroup_quad : require

layout(full_quads) in;
layout(quad_derivatives) in;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void
main()
{
   uint x = uint(gl_FragCoord.x);
   uint count = (x & 1u) + 1u;
   float scaled_uv = uv.x * 32.0;
   uint quad_sum = 0u;
   float derivative_sum = 0.0;

   for (uint i = 0; i < count; ++i) {
      /* Both sources are defined before control flow diverges. The odd lane's
       * second iteration therefore has a fully defined inactive partner. */
      quad_sum += subgroupQuadSwapHorizontal(x);
      /* Keep a real smooth input in the helper-lane protocol. The varying
       * spans exactly 0..1 over 32 pixels, so its scaled derivative is one. */
      derivative_sum += dFdx(scaled_uv);
   }

   uint expected_quad = (x ^ 1u) * count;
   bool quad_ok = quad_sum == expected_quad;
   bool derivative_ok = derivative_sum == float(count);
   color = vec4(quad_ok ? 1.0 : 0.0,
                derivative_ok ? 1.0 : 0.0,
                (x & 1u) != 0u ? 1.0 : 0.0, 1.0);
}
