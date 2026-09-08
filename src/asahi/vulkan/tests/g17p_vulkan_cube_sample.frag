#version 450

layout(set = 0, binding = 0) uniform samplerCube source;
layout(location = 0) out vec4 out_color;

void main()
{
   float x = gl_FragCoord.x;
   vec3 direction = x < 5.0  ? vec3( 1.0,  0.0,  0.0) :
                    x < 10.0 ? vec3(-1.0,  0.0,  0.0) :
                    x < 15.0 ? vec3( 0.0,  1.0,  0.0) :
                    x < 20.0 ? vec3( 0.0, -1.0,  0.0) :
                    x < 25.0 ? vec3( 0.0,  0.0,  1.0) :
                               vec3( 0.0,  0.0, -1.0);
   out_color = texture(source, direction);
}
