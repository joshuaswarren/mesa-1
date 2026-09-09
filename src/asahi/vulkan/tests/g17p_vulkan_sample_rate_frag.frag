#version 450

layout(location = 0) out vec4 out_color;

void
main()
{
   const vec4 samples[4] = vec4[4](
      vec4(1.0, 0.0, 0.0, 1.0),
      vec4(0.0, 1.0, 0.0, 1.0),
      vec4(0.0, 0.0, 1.0, 1.0),
      vec4(1.0, 1.0, 1.0, 1.0));
   out_color = samples[gl_SampleID];
}
