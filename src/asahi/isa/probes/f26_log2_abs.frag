#version 450
/* The live GNOME session's refusal: a 32-bit log2 whose source is an
 * absolute value. NIR folds abs() into the source modifier. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   float x = texcoord.x * texcoord.y;
   uFragColor = vec4 (texture (tex, texcoord.xy).rgb * log2 (abs (x)), 1.0);
}
