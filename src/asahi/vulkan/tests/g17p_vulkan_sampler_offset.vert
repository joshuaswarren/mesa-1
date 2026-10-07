#version 450
/* The uniform buffer at set 0 binding 0 only has to occupy the set: the
 * texture word's offset is a property of the layout, not of the shaders. */
layout(location = 0) out vec2 uv;
void main()
{
   const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0),
                                     vec2( 3.0, -1.0),
                                     vec2(-1.0,  3.0));
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   uv = (positions[gl_VertexIndex] + 1.0) * 0.5;
}
