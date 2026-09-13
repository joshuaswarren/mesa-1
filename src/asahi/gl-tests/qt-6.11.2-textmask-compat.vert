#version 120

struct buf
{
    mat4 modelViewMatrix;
    mat4 projectionMatrix;
    vec2 textureScale;
    float dpr;
    vec4 color;
};

uniform buf _17;

varying vec2 sampleCoord;
attribute vec2 tCoord;
attribute vec4 vCoord;
attribute float _qt_order;

void main()
{
    sampleCoord = tCoord * _17.textureScale;
    vec4 xformed = _17.modelViewMatrix * vCoord;
    gl_Position = _17.projectionMatrix * vec4(floor((xformed.xyz * _17.dpr) + vec3(0.5)) / vec3(_17.dpr), xformed.w);
    gl_Position.z = _qt_order * gl_Position.w;
}
