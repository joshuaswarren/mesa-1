#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 v_texcoord;

void main()
{
   gl_Position = vec4(position, 0.0, 1.0);

   /* Hyprland draws the downsample into the first half of a full-sized work
    * buffer. Model that half-sized raster while preserving blur1's uv * 2. */
   v_texcoord = ((position + vec2(1.0)) * 0.5) * 0.5;
}
