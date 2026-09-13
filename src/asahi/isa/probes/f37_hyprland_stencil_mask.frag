#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   /* Hyprland's default popup blur ignore-alpha threshold is 0.2. */
   float client_alpha = gl_FragCoord.x < 16.0 ? 0.125 : 0.5;
   if (client_alpha <= 0.2)
      discard;
   out_color = vec4(1.0);
}
