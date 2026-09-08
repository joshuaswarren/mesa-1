#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in uint source_bias_in;
layout(location = 0) out vec2 uv;
layout(location = 1) flat out uint source_bias;

void main()
{
   gl_Position = vec4(position, 0.0, 1.0);
   uv = (position + vec2(1.0)) * 0.5;

   /* All fixture vertices carry one. Keeping this as a real integer vertex
    * attribute prevents cross-stage constant folding and forces the same
    * mixed flat+smooth coefficient protocol as real compositor quads. */
   source_bias = source_bias_in;
}
