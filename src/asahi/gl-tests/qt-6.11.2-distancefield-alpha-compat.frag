#version 120

struct buf
{
    mat4 matrix;
    vec2 textureScale;
    vec4 color;
    float alphaMin;
    float alphaMax;
};

uniform buf _14;

uniform sampler2D _qt_texture;

varying vec2 sampleCoord;

void main()
{
    gl_FragData[0] = _14.color * smoothstep(_14.alphaMin, _14.alphaMax,
                                            texture2D(_qt_texture, sampleCoord).w);
}
