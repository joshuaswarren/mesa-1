/*
 * Apple's G17P encoding of a one-bit extract at a per-lane offset -- the shape
 * agx_nir_fuse_algebraic_late folds `(word >> shift) & 1` into.  Same
 * invocation as g17p-sparse-read-witness.metal.
 *
 *   k_bit_reg     bit_and, then a710540002001000f0131500  srl, f57=1
 *   k_bits3_reg   bit_and, then a710540002001000f0133500  srl, f57=3
 *   k_shift_reg   bit_and, then a710540002001000f0130500  srl, f57 absent
 *   k_bit_imm20   a700560002005000f0111100                srl, f30=20, f57=1
 *
 * agx3_alu_bfeil() composes the same field vector, so the encoding is not
 * where the sparse residency bug was: the offset that reached it had simply
 * never been masked to 0..31.  Note that Apple materialises the source-level
 * `& 31` as a separate bit_and rather than relying on the srl to mask.
 */
#include <metal_stdlib>
using namespace metal;

kernel void k_bit_reg(device const uint *word [[buffer(0)]],
                      device const uint *sh [[buffer(1)]],
                      device uint *out [[buffer(2)]],
                      uint tid [[thread_position_in_grid]])
{
   out[tid] = (word[tid] >> (sh[tid] & 31u)) & 1u;
}

kernel void k_bit_imm20(device const uint *word [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
   out[tid] = (word[tid] >> 20) & 1u;
}

kernel void k_bits3_reg(device const uint *word [[buffer(0)]],
                        device const uint *sh [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        uint tid [[thread_position_in_grid]])
{
   out[tid] = (word[tid] >> (sh[tid] & 31u)) & 7u;
}

kernel void k_shift_reg(device const uint *word [[buffer(0)]],
                        device const uint *sh [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        uint tid [[thread_position_in_grid]])
{
   out[tid] = word[tid] >> (sh[tid] & 31u);
}
