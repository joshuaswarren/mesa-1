#version 450

/* Reduced from Omarchy QuickShell's rounded-rectangle fragment path.  The
 * live scaled value forces the G17P backend shape which used to refuse as:
 *
 *   fmul [d32:reg s0=32:reg.abs s1=16:imm64]
 *
 * Immediate 64 is AGX's minifloat encoding of 2.0. */
layout(location = 0) in vec4 texcoord;
layout(location = 0) out vec4 colour;

void main()
{
   float value = texcoord.x;
   float scaled = abs(value) * 2.0;
   float clamped = min(scaled, 0.67);
   colour = vec4(clamped, scaled, value, 1.0);
}
