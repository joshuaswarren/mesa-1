/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform Tag { uint tag; } pc;
layout(location=0) in uvec4 provided;
layout(xfb_buffer=0, xfb_stride=16, xfb_offset=0, location=0) flat out uvec4 captured;
void main() {
  captured = uvec4(uint(gl_VertexIndex), provided.x ^ pc.tag,
                   provided.y + pc.tag, provided.z);
  gl_Position = vec4(0, 0, 0, 1);
}
