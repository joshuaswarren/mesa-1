#version 450
/* vkQuake world.frag's quantized-lightmap path: a 32-bit floor(). The
 * compact hand packer only covers the half form. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   vec2 uv = (floor(vec2(16384.0) * texcoord.xy) + 0.5) / vec2(16384.0);
   uFragColor = texture(tex, uv);
}
