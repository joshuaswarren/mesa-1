#version 450
/* GNOME's second refusal: a 32-bit ceil(). Only the half form was encoded. */
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   vec2 uv = ceil(vec2(64.0) * texcoord.xy) / vec2(64.0);
   uFragColor = texture(tex, uv) * vcolor;
}
