#version 450

layout(constant_id = 0) const int sample_index = 0;

layout(set = 0, binding = 0) uniform sampler2DMS source;

layout(location = 0) out vec4 out_color;

void
main()
{
   out_color = texelFetch(source, ivec2(gl_FragCoord.xy), sample_index);
}
