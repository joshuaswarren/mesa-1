#!/usr/bin/env python3
"""Keep the evidence-backed AGX2 IR -> AGX3 mnemonic map fail-closed."""

import sys
from pathlib import Path

COMPILER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMPILER))

from agx_opcodes import opcodes


assert opcodes["mov_imm"].agx3_mnemonic == "ldimm"
assert opcodes["stop"].agx3_mnemonic == "end"
assert opcodes["threadgroup_barrier"].agx3_mnemonic == "barrier"
assert opcodes["scoped_barrier_agx3"].agx3_mnemonic == "barrier"
assert opcodes["scs_load_agx3"].agx3_mnemonic == "ldscs"
assert opcodes["scs_store_agx3"].agx3_mnemonic == "stscs"
assert opcodes["scs_load_uniform_indexed_agx3"].agx3_mnemonic == "ldscs"
assert opcodes["scs_store_uniform_indexed_agx3"].agx3_mnemonic == "stscs"
assert opcodes["fltnorm_agx3"].agx3_mnemonic == "fltnorm"
assert opcodes["normflt_agx3"].agx3_mnemonic == "normflt"
assert opcodes["pack_rgb10a2_agx3"].agx3_mnemonic == "pack"
assert opcodes["unpack_rgb10a2_agx3"].agx3_mnemonic == "unpack"

expected = {
    "mov_imm", "stop", "threadgroup_barrier", "scoped_barrier_agx3",
    "scs_load_agx3", "scs_store_agx3", "scs_load_uniform_indexed_agx3",
    "scs_store_uniform_indexed_agx3", "fltnorm_agx3", "normflt_agx3",
    "pack_rgb10a2_agx3", "unpack_rgb10a2_agx3",
    "sinc_agx3", "bittest_select_agx3",
}
assert {name for name, op in opcodes.items()
        if op.agx3_mnemonic is not None} == expected
print("agx3 opcode mapping: mov_imm -> ldimm, stop -> end, "
      "threadgroup/scoped barriers -> barrier, direct/uniform-indexed SCS "
      "-> ldscs/stscs, tile normalization -> fltnorm/normflt, "
      "RGB10A2 conversion -> pack/unpack, native trig -> f32sinc/isel; "
      "all other opcodes fail closed")
