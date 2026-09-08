#version 450

layout(set = 0, binding = 0) uniform sampler2DShadow source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main()
{
   /* One cleared depth value, two comparison references: neither half can
    * pass accidentally if the reference operand is ignored or mis-staged. */
   float reference = uv.x < 0.5 ? 0.125 : 0.75;
   float passed = texture(source, vec3(uv, reference));
   out_color = vec4(passed);
}
