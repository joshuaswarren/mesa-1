#version 450

#ifndef TEX_BINDING
#define TEX_BINDING 0
#endif

layout(set = 0, binding = TEX_BINDING) uniform sampler2D tex;
layout(location = 0) in vec2 v_texcoord;
layout(location = 0) out vec4 fragColor;

void main()
{
   const float radius = 3.0;
#ifdef FIXED_TEXTURE_SIZE
   vec2 halfpixel = 0.25 / vec2(FIXED_TEXTURE_SIZE);
#else
   vec2 halfpixel = 0.25 / vec2(textureSize(tex, 1));
#endif
   vec2 uv = v_texcoord / 2.0;

   vec4 sum = texture(tex, uv + vec2(-halfpixel.x * 2.0, 0.0) * radius);
   sum += texture(tex, uv + vec2(-halfpixel.x, halfpixel.y) * radius) * 2.0;
   sum += texture(tex, uv + vec2(0.0, halfpixel.y * 2.0) * radius);
   sum += texture(tex, uv + vec2(halfpixel.x, halfpixel.y) * radius) * 2.0;
   sum += texture(tex, uv + vec2(halfpixel.x * 2.0, 0.0) * radius);
   sum += texture(tex, uv + vec2(halfpixel.x, -halfpixel.y) * radius) * 2.0;
   sum += texture(tex, uv + vec2(0.0, -halfpixel.y * 2.0) * radius);
   sum += texture(tex, uv + vec2(-halfpixel.x, -halfpixel.y) * radius) * 2.0;
   fragColor = sum / 12.0;
}
