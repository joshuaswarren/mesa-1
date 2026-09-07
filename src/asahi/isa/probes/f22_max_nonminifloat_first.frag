#version 450
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
void main() {
   uFragColor = vec4(max(0.0031308, vcolor.r), min(0.0031308, vcolor.g),
                     0.0, 1.0);
}
