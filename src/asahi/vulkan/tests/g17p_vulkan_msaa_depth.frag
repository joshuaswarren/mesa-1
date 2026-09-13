#version 450

layout(constant_id = 0) const int colour_index = 0;

layout(location = 0) out vec4 out_color;

void
main()
{
   /* Both colours convert exactly to UNORM8, and every channel of both is a
    * multiple of 4, so an N-sample box resolve of any mixture of the two is
    * an exact integer at 2x and 4x. */
   const vec4 colours[2] = vec4[2](
      vec4(128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 16.0 / 255.0),
      vec4(16.0 / 255.0, 32.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0));

   out_color = colours[colour_index];
}
