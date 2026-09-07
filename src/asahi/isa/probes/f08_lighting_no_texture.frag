#version 450
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   const vec3 L = normalize(vec3(0.4,0.7,0.6)); float d = max(dot(normalize(vnormal), L), 0.0); uFragColor = vec4(vcolor.rgb * (0.35 + 0.65*d), 1.0);
}
