#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   /* Premultiplied client colours. The low-alpha half was excluded from the
    * blurred-background stencil, but the client itself is still composited. */
   out_color = gl_FragCoord.x < 16.0
                  ? vec4(0.0, 0.0, 0.125, 0.125)
                  : vec4(0.5, 0.0, 0.0, 0.5);
}
