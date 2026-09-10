#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 v_texcoord;

void main()
{
   gl_Position = vec4(position, 0.0, 1.0);

   /* Pair with blur2's uv / 2 so the implicit derivatives select mip 1 of
    * the 2x source used by the offscreen oracle. */
   v_texcoord = (position + vec2(1.0));
}
