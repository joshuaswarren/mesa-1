/* Minimal DirectXMath stand-in for the HelloWorld samples (mingw has none). */
#pragma once
namespace DirectX {
struct XMFLOAT2 { float x, y; XMFLOAT2() = default; XMFLOAT2(float a, float b) : x(a), y(b) {} };
struct XMFLOAT3 { float x, y, z; XMFLOAT3() = default; XMFLOAT3(float a, float b, float c) : x(a), y(b), z(c) {} };
struct XMFLOAT4 { float x, y, z, w; XMFLOAT4() = default; XMFLOAT4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {} };
}
