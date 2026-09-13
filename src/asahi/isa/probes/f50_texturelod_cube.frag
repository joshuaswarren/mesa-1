#version 450
layout (binding = 1) uniform samplerCube tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
/* Image-based lighting reads its prefiltered probe at a roughness-derived
 * level. SuperTuxKart's IBL.frag does exactly this. */
void main() {
   vec3 v = normalize(vcolor.xyz);
   vec3 n = normalize(vnormal);
   vec3 dir = 2.0 * dot(v, n) * n - v;
   uFragColor = textureLod(tex, dir, 7.0 * texcoord.x);
}
