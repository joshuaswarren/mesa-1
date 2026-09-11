#version 450

/* The constant-colour fragment program with its colour moved into a push
 * constant, so four secondary command buffers replayed into one render pass
 * can each paint a distinguishable quadrant. */

layout(push_constant) uniform QuadrantColor {
   vec4 color;
} quadrant;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = quadrant.color;
}
