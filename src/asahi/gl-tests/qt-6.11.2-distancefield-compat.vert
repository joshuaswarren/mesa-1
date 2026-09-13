#version 120

struct buf
{
    mat4 matrix;
    vec2 textureScale;
    vec4 color;
    float alphaMin;
    float alphaMax;
};

uniform buf _17;

varying vec2 sampleCoord;
attribute vec2 tCoord;
attribute vec4 vCoord;
attribute float _qt_order;

void main()
{
    sampleCoord = tCoord * _17.textureScale;
    gl_Position = _17.matrix * vCoord;
    gl_Position.z = _qt_order * gl_Position.w;
}
