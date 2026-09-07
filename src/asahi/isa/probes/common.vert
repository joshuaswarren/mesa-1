#version 450
layout(location = 0) out vec4 texcoord;
layout(location = 1) out vec4 vcolor;
layout(location = 2) out vec3 vnormal;
void main() {
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
   texcoord = vec4(p, 0.0, 1.0);
   vcolor = vec4(p, 0.5, 1.0);
   vnormal = vec3(p, 1.0);
}
