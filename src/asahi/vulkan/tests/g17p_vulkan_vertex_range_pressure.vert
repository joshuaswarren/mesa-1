#version 450
#extension GL_ARB_shader_draw_parameters : require
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 3) in vec4 a3;
layout(location = 4) in vec4 a4;
layout(location = 5) in vec4 a5;
layout(set = 0, binding = 0) uniform sampler2D tex0;
layout(set = 1, binding = 0) uniform sampler2D tex1;
layout(set = 2, binding = 0) uniform sampler2D tex2;
layout(set = 0, binding = 1) uniform Block { vec4 rows[32]; } blk;
void main()
{
   vec4 attribs = a0 + a1 + a2 + a3 + a4 + a5;
   vec2 uv = attribs.xy * 0.5 + 0.5;
   vec4 sampled = textureLod(tex0, uv, 0.0) +
                  textureLod(tex1, uv, 0.0) +
                  textureLod(tex2, uv, 0.0);
   float base = float(gl_VertexIndex - gl_BaseVertexARB);
   vec4 pulled = blk.rows[gl_VertexIndex & 31] + blk.rows[17];

   float acc = 0.28;
   for (int i = 0; i < (gl_VertexIndex & 3); ++i)
      acc = acc * 0.5 + attribs.x;

   gl_Position = vec4(attribs.xy + base + sampled.xy + pulled.xy, acc, 1.0);
}
