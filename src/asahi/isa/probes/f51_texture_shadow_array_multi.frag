#version 450
layout (binding = 1) uniform sampler2DArrayShadow shadowtex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
/* More than one texture instruction, so the combined-sampler lowering cannot
 * claim it and the descriptors arrive indexed. This is the shadow-cascade
 * filter SuperTuxKart's sunlightshadow.frag runs. */
void main() {
   vec2 uv = texcoord.xy * 0.5 + 0.5;
   float layer = floor(vcolor.w * 4.0);
   float ref = texcoord.z * 0.5 + 0.5;
   float sum = 0.0;
   sum += texture(shadowtex, vec4(uv, layer, ref));
   sum += texture(shadowtex, vec4(uv + vec2(1.0 / 1024.0, 0.0), layer, ref));
   sum += texture(shadowtex, vec4(uv + vec2(0.0, 1.0 / 1024.0), layer, ref));
   sum += texture(shadowtex, vec4(uv + vec2(1.0 / 1024.0, 1.0 / 1024.0), layer, ref));
   uFragColor = vec4(sum * 0.25);
}
