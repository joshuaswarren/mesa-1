#version 450

layout(set = 0, binding = 0) uniform sampler2DArray source_image;
layout(push_constant) uniform LayerSelection {
   int layer;
} selection;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
   color = textureLod(source_image, vec3(uv, float(selection.layer)), 0.0);
}
