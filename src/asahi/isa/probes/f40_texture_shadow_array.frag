#version 450
layout (binding = 1) uniform sampler2DArrayShadow tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   uFragColor = vec4(texture(tex, vec4(texcoord.xy, 1.0, 0.5)));
}
