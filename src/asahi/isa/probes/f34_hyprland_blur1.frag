#version 450

#ifndef TEX_BINDING
#define TEX_BINDING 0
#endif

layout(set = 0, binding = TEX_BINDING) uniform sampler2D tex;
layout(location = 0) in vec2 v_texcoord;
layout(location = 0) out vec4 fragColor;

const float Pr = 0.299;
const float Pg = 0.587;
const float Pb = 0.114;
const float a = 0.93;
const float b = 0.11;
const float c = 0.66;

float doubleCircleSigmoid(float x, float shape)
{
   shape = clamp(shape, 0.0, 1.0);

   if (x <= shape)
      return shape - sqrt(shape * shape - x * x);
   else
      return shape +
             sqrt(pow(1.0 - shape, 2.0) - pow(x - 1.0, 2.0));
}

vec3 rgb2hsl(vec3 col)
{
   float red = col.r;
   float green = col.g;
   float blue = col.b;
   float minc = min(col.r, min(col.g, col.b));
   float maxc = max(col.r, max(col.g, col.b));
   float delta = maxc - minc;
   float lum = (minc + maxc) * 0.5;
   float sat = 0.0;
   float hue = 0.0;

   if (lum > 0.0 && lum < 1.0) {
      float mul = lum < 0.5 ? lum : 1.0 - lum;
      sat = delta / (mul * 2.0);
   }

   if (delta > 0.0) {
      vec3 maxcVec = vec3(maxc);
      vec3 masks = vec3(equal(maxcVec, col)) *
                   vec3(notEqual(maxcVec, vec3(green, blue, red)));
      vec3 adds = vec3(0.0, 2.0, 4.0) +
                  vec3(green - blue, blue - red, red - green) / delta;
      hue += dot(adds, masks);
      hue /= 6.0;
      if (hue < 0.0)
         hue += 1.0;
   }

   return vec3(hue, sat, lum);
}

vec3 hsl2rgb(vec3 col)
{
   const float onethird = 1.0 / 3.0;
   const float twothird = 2.0 / 3.0;
   const float rcpsixth = 6.0;
   float hue = col.x;
   float sat = col.y;
   float lum = col.z;
   vec3 xt = vec3(0.0);

   if (hue < onethird) {
      xt.r = rcpsixth * (onethird - hue);
      xt.g = rcpsixth * hue;
   } else if (hue < twothird) {
      xt.g = rcpsixth * (twothird - hue);
      xt.b = rcpsixth * (hue - onethird);
   } else {
      xt = vec3(rcpsixth * (hue - twothird), 0.0,
                rcpsixth * (1.0 - hue));
   }

   xt = min(xt, 1.0);
   float sat2 = 2.0 * sat;
   float satinv = 1.0 - sat;
   float luminv = 1.0 - lum;
   float lum2m1 = 2.0 * lum - 1.0;
   vec3 ct = sat2 * xt + satinv;

   return lum >= 0.5 ? luminv * ct + lum2m1 : lum * ct;
}

void main()
{
   /* Frozen from the retained Omarchy recovery config: size=3, passes=1,
    * vibrancy=.1696, default vibrancy_darkness=0. The source's mip 1 is the
    * same size as the destination, matching Hyprland's first down pass. */
   const float radius = 3.0;
   const float vibrancy = 0.1696;
   const float vibrancy_darkness = 0.0;
#ifdef FIXED_TEXTURE_SIZE
   vec2 halfpixel = 1.0 / vec2(FIXED_TEXTURE_SIZE);
#else
   vec2 halfpixel = 1.0 / vec2(textureSize(tex, 1));
#endif
   vec2 uv = v_texcoord * 2.0;

   vec4 sum = texture(tex, uv) * 4.0;
   sum += texture(tex, uv - halfpixel.xy * radius);
   sum += texture(tex, uv + halfpixel.xy * radius);
   sum += texture(tex, uv + vec2(halfpixel.x, -halfpixel.y) * radius);
   sum += texture(tex, uv - vec2(halfpixel.x, -halfpixel.y) * radius);
   vec4 color = sum / 8.0;

   float vibrancy_darkness1 = 1.0 - vibrancy_darkness;
   vec3 hsl = rgb2hsl(color.rgb);
   float perceivedBrightness = doubleCircleSigmoid(
      sqrt(color.r * color.r * Pr + color.g * color.g * Pg +
           color.b * color.b * Pb),
      0.8 * vibrancy_darkness1);
   float b1 = b * vibrancy_darkness1;
   float boostBase = hsl.y > 0.0
                        ? smoothstep(b1 - c * 0.5, b1 + c * 0.5,
                                     1.0 -
                                        (pow(1.0 - hsl.y * cos(a), 2.0) +
                                         pow(1.0 - perceivedBrightness * sin(a),
                                             2.0)))
                        : 0.0;
   float saturation = clamp(hsl.y + boostBase * vibrancy, 0.0, 1.0);
   fragColor = vec4(hsl2rgb(vec3(hsl.x, saturation, hsl.z)), color.a);
}
