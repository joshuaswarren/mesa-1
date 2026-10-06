// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Reference root signatures, compiled by dxc to real RTS0 blobs so our
// parser is checked against Microsoft's own encoder rather than against
// our own serializer's idea of the format.
#define RS_DESCRIPTORS "UAV(u0), UAV(u1)"
#define RS_TABLE "DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0, numDescriptors=2))"
#define RS_MIXED "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
                 "CBV(b0), " \
                 "DescriptorTable(SRV(t1, numDescriptors=3, space=2), UAV(u4)), " \
                 "RootConstants(num32BitConstants=4, b1), " \
                 "SRV(t9, space=1)"
