#version 450
/* log2 of an absolute value whose operand is read again, so the source is
 * still live at the logarithm. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   float x = texcoord.x * texcoord.y;
   uFragColor = vec4 (texture (tex, texcoord.xy).rgb * log2 (abs (x)), x);
}
