#version 450
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec4 data_in[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 16,
       xfb_stride = 32) out vec4 captured;
void
main()
{
   gl_Position = gl_in[0].gl_Position;
   captured = data_in[0];
   EmitVertex();
   EndPrimitive();
}
