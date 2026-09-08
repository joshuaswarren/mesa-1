#version 450

layout(set = 0, binding = 0) uniform samplerCube source;
layout(location = 0) out vec4 out_color;

layout(constant_id = 0) const float direction_x = 1.0;
layout(constant_id = 1) const float direction_y = 0.0;
layout(constant_id = 2) const float direction_z = 0.0;

void main()
{
   out_color = texture(source, vec3(direction_x, direction_y, direction_z));
}
