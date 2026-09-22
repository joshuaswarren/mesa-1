#version 450

layout(location = 0) out vec4 out_color;

const vec4 colors[8] = vec4[8](
   vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0),
   vec4(0.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0),
   vec4(1.0, 0.0, 1.0, 1.0), vec4(0.0, 1.0, 1.0, 1.0),
   vec4(0.5, 0.25, 0.75, 1.0), vec4(0.25, 0.75, 0.5, 1.0));

void main()
{
   out_color = colors[int(gl_FragCoord.x) & 7];
}
