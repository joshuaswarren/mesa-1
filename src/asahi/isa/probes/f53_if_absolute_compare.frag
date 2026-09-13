#version 450
layout (binding = 1) uniform sampler2D tex;
layout (location = 0) in vec4 texcoord;
layout (location = 1) in vec4 vcolor;
layout (location = 2) in vec3 vnormal;
layout (location = 0) out vec4 uFragColor;
/* Qt Quick's conicalgradient.frag branches on two absolute values. */
void main() {
   float t;
   if (abs(texcoord.y) == abs(texcoord.x))
      t = atan(-texcoord.y + 0.002, texcoord.x) * 0.15915494;
   else
      t = atan(-texcoord.y, texcoord.x) * 0.15915494;
   uFragColor = texture(tex, vec2(t - floor(t), 0.5)) * vcolor.w;
}
