#version 450

/* Depth is a specialization constant so one module serves every draw. */
layout(constant_id = 0) const float draw_depth = 0.0;

layout(location = 0) in vec2 position;

void
main()
{
   gl_Position = vec4(position, draw_depth, 1.0);
}
