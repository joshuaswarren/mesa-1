/*
 * Apple witnesses for the G17P sparse texel read, and for what the earlier
 * dense-vs-sparse field table credited to sparseness.  See
 * src/asahi/isa/README-agx3-packer-limits.md, "The sparse texel-buffer fetch,
 * and a shift that lost its mask".
 *
 *   MT=$(dirname $(xcrun -f metal))            # MetalToolchain v27.1.5194.15
 *   xcrun metal -c g17p-sparse-read-witness.metal -o w.air
 *   xcrun metallib w.air -o w.metallib
 *   printf '{"pipelines":{"compute_pipelines":[{"compute_function":"K"}]}}' \
 *       > K.mtlp-json
 *   "$MT/metal-nt" -arch applegpu_g17p -platform_version macos 27.0 27.0 \
 *       -N K.mtlp-json w.metallib -o K.bin
 *   # code lives in the inner Mach-O's __TEXT,__text
 *
 * k_2d_dread  05800cb8b0001701000000000100                      smp ep3
 * k_2d_sread  2dc00c38b0001701900282000200002028c00018          smp ep4, f142=1
 *             ... then icmp/cndst/bn, a coordinate rescale, a SECOND smp
 *             05820ca010011701a00280000200002000800080 reading a different
 *             descriptor at mask 1, then bit_andc on the residency register.
 *             That second op, not sparseness, is why the first smp's f147
 *             (the final-texture-op marker) reads 0.
 * k_idx_dread   05a00cb8b0001701000280000100002800a00098        one dense read
 * k_idx_dread2  05d00c38b0001701000280000100002800d00018        two dense reads
 *               25c20cb830011701000280000100002820c00098
 *             Both of the pair decode with f120=1 while the single read above
 *             has it clear, so f120 is not a sparse bit.
 */
#include <metal_stdlib>
using namespace metal;

kernel void k_2d_dread(texture2d<uint> t [[texture(0)]],
                       device const uint2 *coord [[buffer(0)]],
                       device uint4 *out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]])
{
   out[tid] = t.read(coord[tid], 0);
}

kernel void k_2d_sread(texture2d<uint> t [[texture(0)]],
                       device const uint2 *coord [[buffer(0)]],
                       device uint4 *out [[buffer(1)]],
                       device uint *res [[buffer(2)]],
                       uint tid [[thread_position_in_grid]])
{
   sparse_color<uint4> r = t.sparse_read(coord[tid], 0);
   out[tid] = r.value();
   res[tid] = r.resident();
}

kernel void k_2d_ssample(texture2d<float> t [[texture(0)]],
                         sampler s [[sampler(0)]],
                         device const float2 *coord [[buffer(0)]],
                         device float4 *out [[buffer(1)]],
                         device uint *res [[buffer(2)]],
                         uint tid [[thread_position_in_grid]])
{
   sparse_color<float4> r = t.sparse_sample(s, coord[tid], level(0.0f));
   out[tid] = r.value();
   res[tid] = r.resident();
}

/* A run-time texture index forces the ep4 form for a DENSE read, which is
 * what separates f120 from sparseness. */
kernel void k_idx_dread(array<texture2d<uint>, 4> t [[texture(0)]],
                        device const uint2 *coord [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        device const uint *sel [[buffer(2)]],
                        uint tid [[thread_position_in_grid]])
{
   out[tid] = t[sel[tid]].read(coord[tid], 0);
}

kernel void k_idx_dread2(array<texture2d<uint>, 4> t [[texture(0)]],
                         device const uint4 *coord [[buffer(0)]],
                         device uint4 *out [[buffer(1)]],
                         device const uint *sel [[buffer(2)]],
                         uint tid [[thread_position_in_grid]])
{
   uint i = sel[tid];
   out[tid] = t[i].read(coord[tid].xy, 0) + t[i + 1].read(coord[tid].zw, 0);
}
