#version 450
layout(location = 0) in vec4 payload;
layout(set = 0, binding = 0) uniform samplerBuffer data_buffer;
layout(location = 0) out vec4 captured;
void
main()
{
   gl_Position = vec4(0, 0, 0, 1);
   captured = payload + texelFetch(data_buffer, gl_VertexIndex);
}
