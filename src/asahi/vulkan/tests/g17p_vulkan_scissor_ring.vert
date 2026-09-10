#version 450

/*
 * One full-viewport triangle from gl_VertexIndex alone: no vertex buffer, no
 * descriptors, nothing that could clip the primitive itself. Every pixel of
 * the attachment is inside the primitive, so the only thing that decides
 * which pixels a draw writes is the scissor rectangle.
 */
void
main()
{
   vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
