#version 460
#extension GL_ARB_shader_viewport_layer_array : require

layout(location = 0) in vec2 position;

void
main()
{
   gl_Position = vec4(position, 0.0, 1.0);
   gl_Layer = gl_InstanceIndex;
}
