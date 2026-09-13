#version 450
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
/* `dot(view - world, normal) > 0.0` with the negate landing on a comparator:
 * ordinary back-face and rim-light arithmetic. */
void main() {
   vec3 d = vcolor.xyz - texcoord.xyz;
   uFragColor = (-dot(d, vnormal) > texcoord.w) ? vec4(1.0) : vcolor;
}
