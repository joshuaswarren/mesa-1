#version 450
/* vkQuake's fog term: `exp (-density * density * coord * coord)` in
 * Shaders/world_common.inc and basic_common.inc. NIR lowers exp to exp2 and
 * folds the negation into the source, so the packer sees a 32-bit exp2 whose
 * source carries the negate modifier.
 *
 * Written with exp2 directly, because with exp() the negation is absorbed
 * into the log2(e) constant multiply and never reaches the packer. Every
 * runtime value comes from texcoord.xy: common.vert writes constants into
 * every other component, and the linker folds those away. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   float d = texcoord.x;
   float z = texcoord.y;
   float fog = exp2 (-(d * d * z * z));
   uFragColor = vec4 (mix (vnormal, texture (tex, texcoord.xy).rgb,
                           clamp (fog, 0.0, 1.0)), 1.0);
}
