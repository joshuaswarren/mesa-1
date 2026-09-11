#version 450

#ifndef UV_SCALE
#define UV_SCALE 0.5
#endif

layout(location = 0) out vec2 v_texcoord;

void main()
{
   const vec2 positions[3] = vec2[3](
      vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
   vec2 base_uv = positions[gl_VertexIndex] * 0.5 + 0.5;
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   v_texcoord = base_uv * UV_SCALE;
}
