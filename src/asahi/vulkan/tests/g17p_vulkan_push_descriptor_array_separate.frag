#version 450

layout(set = 0, binding = 0) uniform sampler source_sampler;
layout(set = 0, binding = 1) uniform texture2DArray source_texture;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
   color = texture(sampler2DArray(source_texture, source_sampler),
                   vec3(uv, 0.0));
}
