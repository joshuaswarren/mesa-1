/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef D12_NIR_INTEGER_H
#define D12_NIR_INTEGER_H

#include "nir_builder.h"

/* DXIL integer division/remainder instructions have the DXBC zero-divisor
 * result: all bits set. This is distinct from source-level HLSL/LLVM
 * undefined behaviour. See DirectXShaderCompiler issue 3150 and the MIT
 * dxil-spirv opcodes_llvm_builtins.cpp discussion of integer division.
 *
 * Give NIR a nonzero divisor as well as selecting the specified result.
 * This keeps constant folding and backend reciprocal lowering away from
 * division by zero, including after scalarisation of a mixed vector.
 */
static inline nir_def *
d12_nir_integer_divmod(nir_builder *b, nir_def *x, nir_def *y,
                      bool is_signed, bool remainder)
{
   nir_def *zero = nir_ieq_imm(b, y, 0);
   nir_def *divisor = nir_bcsel(b, zero, nir_imm_intN_t(b, 1, y->bit_size), y);
   nir_def *result = remainder
      ? (is_signed ? nir_irem(b, x, divisor) : nir_umod(b, x, divisor))
      : (is_signed ? nir_idiv(b, x, divisor) : nir_udiv(b, x, divisor));
   return nir_bcsel(b, zero, nir_imm_intN_t(b, -1, x->bit_size), result);
}

#endif
