#version 450

layout(location = 0) out vec4 out_color;

void
main()
{
   /* 128/64/32/16 exactly after UNORM8 conversion. Every one of those is
    * divisible by 4, so a box resolve of any coverage pattern at 2x or 4x is
    * an exact integer and the oracle needs no rounding model. */
   out_color = vec4(128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 16.0 / 255.0);
}
