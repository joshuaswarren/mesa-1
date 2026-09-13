#version 120

struct buf
{
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
    vec2 textureScale;
    float dpr;
    vec4 color;
};

uniform buf _14;

uniform sampler2D _qt_texture;

varying vec2 sampleCoord;

void main()
{
    gl_FragData[0] = _14.color * texture2D(_qt_texture, sampleCoord).w;
}
