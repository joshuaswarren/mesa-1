#version 450
/* The same negated exponential with the negated value read again afterwards,
 * so the source register is still live at the exp2. Encoding "last use" over
 * a live register would be a silent miscompile rather than a refusal, which
 * is why the lifetime hint has to be modelled and not pinned. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   float x = texcoord.x * texcoord.y;
   float fog = exp2 (-x);
   uFragColor = vec4 (texture (tex, texcoord.xy).rgb * fog, x);
}
