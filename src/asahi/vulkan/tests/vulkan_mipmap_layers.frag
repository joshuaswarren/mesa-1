/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform Color { ivec4 value; } pc;
layout(location=0) out ivec4 color;
void main() { color = pc.value; }
