#version 450

/*
 * A constant 16/255 in every channel, blended ONE/ONE. The attachment then
 * reads back as 16 times the number of scissored draws that covered the
 * pixel, so the image is a literal coverage count: 0 is a hole, 16 is right,
 * 32 is a doubly-blended seam. No sampling, no interpolation, no rounding
 * that could blur those apart.
 */
layout(location = 0) out vec4 color;

void
main()
{
   color = vec4(16.0 / 255.0);
}
