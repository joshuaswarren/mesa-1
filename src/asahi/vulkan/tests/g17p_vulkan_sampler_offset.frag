#version 450
/* The only fragment resource: one combined image sampler at binding 1. */
layout(set = 0, binding = 1) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main()
{
   color = texture(tex, uv);
}
