#version 450
layout(location = 0) in vec4 payload;
layout(set = 0, binding = 0) uniform samplerBuffer data_buffer;
layout(location = 0, xfb_buffer = 0, xfb_offset = 16,
       xfb_stride = 32) out vec4 captured;
void
main()
{
   const vec2 positions[3] =
      vec2[3](vec2(-0.8, -0.8), vec2(0.8, -0.8), vec2(0.0, 0.8));
   gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
   captured = payload + texelFetch(data_buffer, gl_VertexIndex);
}
