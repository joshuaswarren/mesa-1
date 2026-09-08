#version 450
layout(location = 0) in vec4 payload;

/* The committed SPIR-V header adds TransformFeedback/Xfb plus Offset 16 and
 * XfbStride 32 decorations to this output.  That deliberate hole, combined
 * with the nonzero destination binding, detects zero-based or stride-confused
 * writes in the public oracle. */
layout(location = 0) out vec4 captured;

void main()
{
   const vec2 positions[3] = vec2[3](vec2(-0.8, -0.8),
                                     vec2( 0.8, -0.8),
                                     vec2( 0.0,  0.8));
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   captured = payload;
}
