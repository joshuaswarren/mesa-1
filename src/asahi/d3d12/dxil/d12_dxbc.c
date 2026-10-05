/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DXBC (Shader Model 4/5) -> NIR.
 *
 * WHY THIS EXISTS. Measured against vkd3d-proton's suite, 202 of 557
 * tests fail at the very first step with "no DXIL part in DXBC
 * container" -- and at nothing else first. They embed fxc-compiled
 * Shader Model 5.x, which has no DXIL anywhere:
 *
 *     chunks: 3
 *        ISGN 44
 *        OSGN 44
 *        SHEX 240   sm5.0 kind=1
 *
 * SM 5.1 is legal D3D12, so this is a real conformance gap. It is also
 * the single largest one: no other fix competes.
 *
 * WHAT DXBC IS, and how it differs from what we already have. DXIL is
 * LLVM 3.7 bitcode -- SSA, typed, with dx.op calls. DXBC is a flat TOKEN
 * STREAM over a fixed register file: r# temporaries, v# inputs, o#
 * outputs, cb# constant buffers, t#/s# resources and samplers. Nothing is
 * SSA. So this is a genuinely separate front end and shares only the
 * container walk with the DXIL one.
 *
 * The encoding, from the public shader-model documentation:
 *
 *   opcode token   [10:0]  opcode
 *                  [23:11] opcode-specific control bits
 *                  [30:24] instruction length in tokens
 *                  [31]    an extended opcode token follows
 *
 *   operand token  [1:0]   number of components (0=none, 1=one, 2=four)
 *                  [11:2]  component selection / swizzle / mask
 *                  [19:12] operand type (the register file)
 *                  [21:20] index dimension (0..3)
 *                  [30:22] how each index is represented, 3 bits each
 *                  [31]    an extended operand token follows
 *
 * Registers are read and written as vec4s; scalars come out of a swizzle.
 * That maps onto NIR the way any pre-SSA IR does: a temporary file of
 * nir_variables that later passes promote.
 */

#include "d12_debug.h"
#include "d12_dxil.h"
#include "../d12_native_descriptor_core.h"

#include "agx_compile.h"
#include "agx_nir_texture.h"
#include "nir.h"
#include "nir_builder.h"
#include "d12_nir_sparse_buffer.h"
#include "util/log.h"
#include "util/u_math.h"

#include <string.h>

#define D12_DXBC_MAX_TEMPS    4096
#define D12_DXBC_MAX_OPERANDS 8

/* Register files (operand types) we understand. */
enum d12_dxbc_file {
   D12_DXBC_FILE_TEMP = 0,
   D12_DXBC_FILE_INPUT = 1,
   D12_DXBC_FILE_OUTPUT = 2,
   D12_DXBC_FILE_INDEXABLE_TEMP = 3,
   D12_DXBC_FILE_IMMEDIATE32 = 4,
   D12_DXBC_FILE_SAMPLER = 6,
   D12_DXBC_FILE_RESOURCE = 7,
   D12_DXBC_FILE_CONSTANT_BUFFER = 8,
   D12_DXBC_FILE_IMMEDIATE_CONSTANT_BUFFER = 9,
   D12_DXBC_FILE_INPUT_PRIMITIVE_ID = 11,
   D12_DXBC_FILE_OUTPUT_DEPTH = 12,
   D12_DXBC_FILE_NULL = 13,
   D12_DXBC_FILE_RASTERIZER = 14,
   /* Compute system values. These are not varyings and have no
    * declaration in the token stream -- the register file IS the
    * meaning. */
   D12_DXBC_FILE_OUTPUT_COVERAGE_MASK = 15,
   D12_DXBC_FILE_OUTPUT_CONTROL_POINT_ID = 22,
   D12_DXBC_FILE_INPUT_FORK_INSTANCE_ID = 23,
   D12_DXBC_FILE_INPUT_JOIN_INSTANCE_ID = 24,
   D12_DXBC_FILE_INPUT_CONTROL_POINT = 25,
   D12_DXBC_FILE_OUTPUT_CONTROL_POINT = 26,
   D12_DXBC_FILE_INPUT_PATCH_CONSTANT = 27,
   D12_DXBC_FILE_INPUT_DOMAIN_POINT = 28,
   D12_DXBC_FILE_UAV = 30,
   D12_DXBC_FILE_TGSM = 31,
   D12_DXBC_FILE_THREAD_ID = 32,
   D12_DXBC_FILE_THREAD_GROUP_ID = 33,
   D12_DXBC_FILE_THREAD_ID_IN_GROUP = 34,
   D12_DXBC_FILE_COVERAGE_MASK = 35,
   D12_DXBC_FILE_THREAD_ID_IN_GROUP_FLAT = 36,
};

/*
 * A relative index is encoded as a complete operand nested inside the
 * containing operand.  Keep a non-recursive copy of it: the public token
 * grammar permits the nested operand to carry its own ordinary indices, but
 * another relative index there would require unbounded recursion and is not
 * produced by the shaders this front end accepts.
 */
struct d12_dxbc_relative_operand {
   bool valid;
   enum d12_dxbc_file file;
   unsigned num_indices;
   uint32_t index[3];
   unsigned num_components;
   unsigned sel_mode;
   uint8_t swizzle[4];
   uint8_t mask;
   bool neg;
   bool abs;
   uint32_t imm[4];
};

/* The opcodes this front end covers. Anything else fails the shader
 * loudly rather than silently miscompiling it. */
enum d12_dxbc_op {
   D12_DXBC_OP_ADD = 0,
   D12_DXBC_OP_AND = 1,
   D12_DXBC_OP_BREAK = 2,
   D12_DXBC_OP_BREAKC = 3,
   D12_DXBC_OP_CASE = 6,
   D12_DXBC_OP_DERIV_RTX = 11,
   D12_DXBC_OP_DERIV_RTY = 12,
   D12_DXBC_OP_DEFAULT = 10,
   D12_DXBC_OP_DISCARD = 13,
   D12_DXBC_OP_DIV = 14,
   D12_DXBC_OP_DP2 = 15,
   D12_DXBC_OP_DP3 = 16,
   D12_DXBC_OP_DP4 = 17,
   D12_DXBC_OP_ELSE = 18,
   D12_DXBC_OP_ENDIF = 21,
   D12_DXBC_OP_ENDLOOP = 22,
   D12_DXBC_OP_ENDSWITCH = 23,
   D12_DXBC_OP_EQ = 24,
   D12_DXBC_OP_EXP = 25,
   D12_DXBC_OP_FRC = 26,
   D12_DXBC_OP_FTOI = 27,
   D12_DXBC_OP_FTOU = 28,
   D12_DXBC_OP_GE = 29,
   D12_DXBC_OP_IADD = 30,
   D12_DXBC_OP_IF = 31,
   D12_DXBC_OP_IEQ = 32,
   D12_DXBC_OP_IGE = 33,
   D12_DXBC_OP_ILT = 34,
   D12_DXBC_OP_IMAD = 35,
   D12_DXBC_OP_IMAX = 36,
   D12_DXBC_OP_IMIN = 37,
   D12_DXBC_OP_IMUL = 38,
   D12_DXBC_OP_INE = 39,
   D12_DXBC_OP_INEG = 40,
   D12_DXBC_OP_ISHL = 41,
   D12_DXBC_OP_ISHR = 42,
   D12_DXBC_OP_ITOF = 43,
   D12_DXBC_OP_LD = 45,
   D12_DXBC_OP_LD_MS = 46,
   D12_DXBC_OP_LOG = 47,
   D12_DXBC_OP_LOOP = 48,
   D12_DXBC_OP_LT = 49,
   D12_DXBC_OP_MAD = 50,
   D12_DXBC_OP_MIN = 51,
   D12_DXBC_OP_MAX = 52,
   D12_DXBC_OP_MOV = 54,
   D12_DXBC_OP_MOVC = 55,
   D12_DXBC_OP_MUL = 56,
   D12_DXBC_OP_NE = 57,
   D12_DXBC_OP_NOT = 59,
   D12_DXBC_OP_OR = 60,
   D12_DXBC_OP_RESINFO = 61,
   D12_DXBC_OP_BUFINFO = 121,
   D12_DXBC_OP_DERIV_RTX_COARSE = 122,
   D12_DXBC_OP_DERIV_RTX_FINE = 123,
   D12_DXBC_OP_DERIV_RTY_COARSE = 124,
   D12_DXBC_OP_DERIV_RTY_FINE = 125,
   D12_DXBC_OP_GATHER4_C = 126,
   D12_DXBC_OP_GATHER4_PO = 127,
   D12_DXBC_OP_GATHER4_PO_C = 128,
   D12_DXBC_OP_RET = 62,
   D12_DXBC_OP_RETC = 63,
   D12_DXBC_OP_ROUND_NE = 64,
   D12_DXBC_OP_ROUND_NI = 65,
   D12_DXBC_OP_ROUND_PI = 66,
   D12_DXBC_OP_ROUND_Z = 67,
   D12_DXBC_OP_RSQ = 68,
   D12_DXBC_OP_SAMPLE = 69,
   D12_DXBC_OP_SAMPLE_C = 70,
   D12_DXBC_OP_SAMPLE_C_LZ = 71,
   D12_DXBC_OP_SAMPLE_L = 72,
   D12_DXBC_OP_SAMPLE_D = 73,
   D12_DXBC_OP_SAMPLE_B = 74,
   D12_DXBC_OP_GATHER4 = 109,
   D12_DXBC_OP_SQRT = 75,
   D12_DXBC_OP_SWITCH = 76,
   D12_DXBC_OP_SINCOS = 77,
   D12_DXBC_OP_UDIV = 78,
   D12_DXBC_OP_ULT = 79,
   D12_DXBC_OP_UGE = 80,
   D12_DXBC_OP_UMUL = 81,
   D12_DXBC_OP_UMAX = 83,
   D12_DXBC_OP_UMIN = 84,
   D12_DXBC_OP_USHR = 85,
   D12_DXBC_OP_UTOF = 86,
   D12_DXBC_OP_XOR = 87,
   D12_DXBC_OP_SAMPLE_INFO = 111,
   D12_DXBC_OP_SAMPLE_POS = 110,
   D12_DXBC_OP_EVAL_SAMPLE_INDEX = 204,
   D12_DXBC_OP_EVAL_CENTROID = 205,
   D12_DXBC_OP_MSAD = 213,
   D12_DXBC_OP_CONTINUE = 7,
   D12_DXBC_OP_CONTINUEC = 8,
   D12_DXBC_OP_CUSTOMDATA = 53,
   D12_DXBC_OP_NOP = 58,
   D12_DXBC_OP_RCP = 129,
   D12_DXBC_OP_F32TOF16 = 130,
   D12_DXBC_OP_F16TOF32 = 131,
   D12_DXBC_OP_COUNTBITS = 134,
   D12_DXBC_OP_FIRSTBIT_HI = 135,
   D12_DXBC_OP_FIRSTBIT_LO = 136,
   D12_DXBC_OP_FIRSTBIT_SHI = 137,
   D12_DXBC_OP_UBFE = 138,
   D12_DXBC_OP_IBFE = 139,
   D12_DXBC_OP_BFI = 140,
   D12_DXBC_OP_BFREV = 141,
   D12_DXBC_OP_SWAPC = 142,
   D12_DXBC_OP_UMAD = 82,
   /* Hull-shader phase delimiters. They carry no operands; the phase bodies
    * follow as ordinary instructions/declarations. */
   D12_DXBC_OP_HS_DECLS = 113,
   D12_DXBC_OP_HS_CONTROL_POINT_PHASE = 114,
   D12_DXBC_OP_HS_FORK_PHASE = 115,
   D12_DXBC_OP_HS_JOIN_PHASE = 116,
   D12_DXBC_OP_DCL_INDEX_RANGE = 91,
   D12_DXBC_OP_DCL_OUTPUT_SGV = 102,
   /* The SM5 declaration block. Bindings all come from the root
    * signature and the container's signature chunks, so these carry
    * nothing the driver needs -- but they must be STEPPED OVER
    * correctly, or everything after them decodes as garbage. */
   D12_DXBC_OP_DCL_UAV_TYPED = 156,
   D12_DXBC_OP_DCL_UAV_RAW = 157,
   D12_DXBC_OP_DCL_UAV_STRUCTURED = 158,
   D12_DXBC_OP_DCL_TGSM_RAW = 159,
   D12_DXBC_OP_DCL_TGSM_STRUCTURED = 160,
   D12_DXBC_OP_DCL_RESOURCE_RAW = 161,
   D12_DXBC_OP_DCL_RESOURCE_STRUCTURED = 162,
   D12_DXBC_OP_DCL_STREAM = 143,
   D12_DXBC_OP_DCL_GS_INPUT_PRIMITIVE = 93,
   D12_DXBC_OP_DCL_GS_OUTPUT_TOPOLOGY = 92,
   D12_DXBC_OP_DCL_MAX_OUTPUT_VERTEX_COUNT = 94,
   D12_DXBC_OP_DCL_GS_INSTANCE_COUNT = 206,
   D12_DXBC_OP_DCL_INPUT_CONTROL_POINT_COUNT = 147,
   D12_DXBC_OP_DCL_OUTPUT_CONTROL_POINT_COUNT = 148,
   D12_DXBC_OP_DCL_TESS_DOMAIN = 149,
   D12_DXBC_OP_DCL_TESS_PARTITIONING = 150,
   D12_DXBC_OP_DCL_TESS_OUTPUT_PRIMITIVE = 151,
   D12_DXBC_OP_DCL_HS_MAX_TESSFACTOR = 152,
   D12_DXBC_OP_DCL_HS_FORK_PHASE_INSTANCE_COUNT = 153,
   D12_DXBC_OP_DCL_HS_JOIN_PHASE_INSTANCE_COUNT = 154,
   /* Public SM5 raw/structured buffer access opcodes. Raw loads carry
    * dest, byte offset, resource; structured loads additionally carry a
    * member offset. */
   D12_DXBC_OP_LD_RAW = 165,
   D12_DXBC_OP_LD_STRUCTURED = 167,
   D12_DXBC_OP_STORE_STRUCTURED = 168,
   /* UAV/TGSM atomic family from the public SM5 token enum. */
   D12_DXBC_OP_ATOMIC_AND = 169,
   D12_DXBC_OP_ATOMIC_OR = 170,
   D12_DXBC_OP_ATOMIC_XOR = 171,
   D12_DXBC_OP_ATOMIC_CMP_STORE = 172,
   D12_DXBC_OP_ATOMIC_IADD = 173,
   D12_DXBC_OP_ATOMIC_IMAX = 174,
   D12_DXBC_OP_ATOMIC_IMIN = 175,
   D12_DXBC_OP_ATOMIC_UMAX = 176,
   D12_DXBC_OP_ATOMIC_UMIN = 177,
   /* Append/consume counters: destination and UAV, no element coordinate. */
   D12_DXBC_OP_IMM_ATOMIC_ALLOC = 178,
   D12_DXBC_OP_IMM_ATOMIC_CONSUME = 179,
   /* Value-returning forms: dest, UAV, coordinate, operands. */
   D12_DXBC_OP_IMM_ATOMIC_IADD = 180,
   D12_DXBC_OP_IMM_ATOMIC_AND = 181,
   D12_DXBC_OP_IMM_ATOMIC_OR = 182,
   D12_DXBC_OP_IMM_ATOMIC_XOR = 183,
   D12_DXBC_OP_IMM_ATOMIC_EXCH = 184,
   D12_DXBC_OP_IMM_ATOMIC_CMP_EXCH = 185,
   D12_DXBC_OP_IMM_ATOMIC_IMAX = 186,
   D12_DXBC_OP_IMM_ATOMIC_IMIN = 187,
   D12_DXBC_OP_IMM_ATOMIC_UMAX = 188,
   D12_DXBC_OP_IMM_ATOMIC_UMIN = 189,
   D12_DXBC_OP_SYNC = 190,
   /* Both write to file 30 with three operands. 164 takes an element
    * coordinate, 166 a byte offset -- typed versus raw. */
   D12_DXBC_OP_STORE_UAV_TYPED = 164,
   D12_DXBC_OP_STORE_RAW = 166,
   /* The load counterpart to 164: same operand shape, dest in a
    * temporary and the UAV last. */
   D12_DXBC_OP_LD_UAV_TYPED = 163,
   /* Geometry shader vertex emission. The _STREAM forms name a stream
    * in their first operand; the plain ones mean stream 0. */
   D12_DXBC_OP_CUT = 9,
   D12_DXBC_OP_EMIT = 19,
   D12_DXBC_OP_EMITTHENCUT = 20,
   D12_DXBC_OP_EMIT_STREAM = 117,
   D12_DXBC_OP_CUT_STREAM = 118,
   D12_DXBC_OP_EMITTHENCUT_STREAM = 119,
   D12_DXBC_OP_DCL_RESOURCE = 88,
   D12_DXBC_OP_DCL_CONSTANT_BUFFER = 89,
   D12_DXBC_OP_DCL_SAMPLER = 90,
   D12_DXBC_OP_DCL_INPUT = 95,
   D12_DXBC_OP_DCL_INPUT_SGV = 96,
   D12_DXBC_OP_DCL_INPUT_SIV = 97,
   D12_DXBC_OP_DCL_INPUT_PS = 98,
   D12_DXBC_OP_DCL_INPUT_PS_SGV = 99,
   D12_DXBC_OP_DCL_INPUT_PS_SIV = 100,
   D12_DXBC_OP_DCL_OUTPUT = 101,
   D12_DXBC_OP_DCL_OUTPUT_SIV = 103,
   D12_DXBC_OP_DCL_TEMPS = 104,
   D12_DXBC_OP_DCL_INDEXABLE_TEMP = 105,
   D12_DXBC_OP_DCL_GLOBAL_FLAGS = 106,
   D12_DXBC_OP_DCL_THREAD_GROUP = 155,
   D12_DXBC_OP_GATHER4_S = 219,
   D12_DXBC_OP_GATHER4_C_S = 220,
   D12_DXBC_OP_GATHER4_PO_S = 221,
   D12_DXBC_OP_GATHER4_PO_C_S = 222,
   D12_DXBC_OP_LD_S = 223,
   D12_DXBC_OP_LD_MS_S = 224,
   D12_DXBC_OP_LD_UAV_TYPED_S = 225,
   D12_DXBC_OP_LD_RAW_S = 226,
   D12_DXBC_OP_LD_STRUCTURED_S = 227,
   D12_DXBC_OP_SAMPLE_L_S = 228,
   D12_DXBC_OP_SAMPLE_C_LZ_S = 229,
   D12_DXBC_OP_SAMPLE_CLAMP_S = 230,
   D12_DXBC_OP_SAMPLE_B_CLAMP_S = 231,
   D12_DXBC_OP_SAMPLE_D_CLAMP_S = 232,
   D12_DXBC_OP_SAMPLE_C_CLAMP_S = 233,
   D12_DXBC_OP_CHECK_ACCESS_FULLY_MAPPED = 234,
};

/* Public token encoding from Microsoft's D3D12TokenizedProgramFormat.hpp.
 * Unlike sampled textures, Buffer<T> must become a NIR buffer texture so
 * agx_nir_lower_texture applies the texel-buffer addressing rules.
 */
#define D12_DXBC_RESOURCE_DIMENSION_MASK  0x0000f800u
#define D12_DXBC_RESOURCE_DIMENSION_SHIFT 11
#define D12_DXBC_RESOURCE_DIMENSION_BUFFER 1
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D          2
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D 3
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS 4
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_3D 5
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE 6
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY    7
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY 8
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY 9
#define D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY 10

#define D12_DXBC_RESINFO_RETURN_TYPE_MASK  0x00001800u
#define D12_DXBC_RESINFO_RETURN_TYPE_SHIFT 11
#define D12_DXBC_RESINFO_RETURN_FLOAT      0
#define D12_DXBC_RESINFO_RETURN_UINT       2
#define D12_DXBC_RESOURCE_RETURN_UNORM     1
#define D12_DXBC_RESOURCE_RETURN_SNORM     2
#define D12_DXBC_RESOURCE_RETURN_SINT      3
#define D12_DXBC_RESOURCE_RETURN_UINT      4
#define D12_DXBC_RESOURCE_RETURN_FLOAT     5

struct d12_dxbc_operand {
   enum d12_dxbc_file file;
   unsigned num_indices;
   uint32_t index[3];
   bool index_is_relative[3];
   struct d12_dxbc_relative_operand relative[3];

   /* Component count is bits 1:0. For four-component operands only,
    * selection mode is the separate bits 3:2. */
   unsigned num_components;
   unsigned sel_mode;
   uint8_t swizzle[4]; /* source form */
   uint8_t mask;       /* destination form */

   bool neg;
   bool abs;

   uint32_t imm[4]; /* IMMEDIATE32 */
};

/*
 * One ISGN/OSGN element. The chunk is a flat table: a count, then 24-byte
 * records, with the semantic name a byte offset from the chunk's data.
 *
 * This is where a DXBC output learns WHAT it is. The token stream only
 * says "o2"; that a given register is SV_Position rather than a generic
 * varying lives here, and agx_nir_lower_uvs and the tilebuffer lowering
 * both key on the resulting io_semantics.location. Without this mapping
 * every store_output reached the backend as location 0 and asserted.
 */
struct d12_dxbc_sig_elem {
   char name[32];
   uint32_t semantic_index;
   uint32_t sysval;
   uint32_t component_type;
   uint32_t min_precision;
   uint32_t reg;
   uint8_t mask;
};

/* SM 5.1 range IDs and register spaces are not limited to the 32 ordinary
 * signature registers.  Full-root tests legitimately declare 62 resource
 * ranges; truncating the declaration tables at 32 made range 32 look like an
 * undeclared t32 in space 0. */
#define D12_DXBC_MAX_SIG 256

struct d12_dxbc_binding_range {
   uint32_t lower_bound;
   uint32_t upper_bound;
   uint32_t space;
   bool valid;
};

struct d12_dxbc_tgsm_decl {
   uint32_t base;
   uint32_t size;
   uint32_t stride;
   bool valid;
};

enum d12_dxbc_cf_kind {
   D12_DXBC_CF_IF,
   D12_DXBC_CF_LOOP,
   D12_DXBC_CF_SWITCH,
};

enum d12_dxbc_hs_phase {
   D12_DXBC_HS_DECLS,
   D12_DXBC_HS_CONTROL_POINT,
   D12_DXBC_HS_FORK,
   D12_DXBC_HS_JOIN,
};

struct d12_dxbc_cf {
   enum d12_dxbc_cf_kind kind;
   nir_if *nif;
   bool in_else;
   bool case_open;
   nir_def *switch_value;
   nir_def *switch_any_match;
   nir_variable *switch_matched;
};

struct d12_dxbc_ctx {
   char errbuf[64];
   const struct d12_binding_map *map;
   uint32_t descriptor_generation;
   struct d12_native_descriptor_layout native_layout;

   struct d12_dxbc_sig_elem in_sig[D12_DXBC_MAX_SIG];
   unsigned num_in_sig;
   struct d12_dxbc_sig_elem out_sig[D12_DXBC_MAX_SIG];
   unsigned num_out_sig;
   struct d12_dxbc_sig_elem patch_sig[D12_DXBC_MAX_SIG];
   unsigned num_patch_sig;
   uint8_t input_interp[D12_DXBC_MAX_SIG];
   enum d12_dxbc_hs_phase hs_phase;
   unsigned hs_input_control_points;
   unsigned hs_max_invocations;
   bool hs_has_control_point_phase;
   bool hs_instance_phase_open;
   const uint32_t *immediate_constants;
   unsigned num_immediate_constants;

   /* Element stride of each structured buffer, by register. Declared
    * once and needed at every access, since the address is
    * index * stride + offset. */
   uint32_t srv_stride[D12_DXBC_MAX_SIG];
   uint32_t uav_stride[D12_DXBC_MAX_SIG];
   uint8_t srv_dim[D12_DXBC_MAX_SIG];
   uint8_t uav_dim[D12_DXBC_MAX_SIG];
   uint8_t uav_return_type[D12_DXBC_MAX_SIG];
   bool uav_raw[D12_DXBC_MAX_SIG];
   bool srv_raw[D12_DXBC_MAX_SIG];
   /* SM 5.1 names constant buffers as cb<range-id>[register][row].  The
    * declaration is the only place that maps that range ID to a register
    * space and to the bounds needed for safe dynamic slot arithmetic. */
   struct d12_dxbc_binding_range cb_range[D12_DXBC_MAX_SIG];
   struct d12_dxbc_binding_range srv_range[D12_DXBC_MAX_SIG];
   struct d12_dxbc_binding_range sampler_range[D12_DXBC_MAX_SIG];
   /* SM 5.1 UAV operands likewise carry a range ID plus an optional
    * relative register.  Retain the declaration's bounds and space so a
    * dynamic counter access maps into the root signature's dense slots.
    */
   struct d12_dxbc_binding_range uav_range[D12_DXBC_MAX_SIG];
   /* g# declarations occupy one flat NIR workgroup-shared allocation. Raw
    * declarations have stride zero; structured declarations retain their
    * public byte stride for element/member addressing. */
   struct d12_dxbc_tgsm_decl tgsm[D12_DXBC_MAX_SIG];
   uint32_t shared_size;
   nir_builder b;
   const uint32_t *tok;
   const uint32_t *end;

   nir_variable *temps[D12_DXBC_MAX_TEMPS];
   unsigned num_temps;
   nir_variable *indexable_temps[D12_DXBC_MAX_SIG];
   uint32_t indexable_temp_size[D12_DXBC_MAX_SIG];
   nir_def **indexable_temp_shadow[D12_DXBC_MAX_SIG];
   bool indexable_temp_shadow_valid[D12_DXBC_MAX_SIG];

   /* DXBC control flow is STRUCTURED -- if/else/endif and loop/endloop
    * nest properly -- which is exactly what nir_push_if/nir_push_loop
    * want, so a stack of what we opened is enough to close them. */
   struct d12_dxbc_cf cf[64];
   unsigned cf_depth;
   nir_if *conditional_return_if;

   const char *error;
};

static void
fail(struct d12_dxbc_ctx *ctx, const char *why)
{
   if (!ctx->error)
      ctx->error = why;
}

static unsigned
parse_signature(const uint8_t *base, size_t size_B,
                struct d12_dxbc_sig_elem *out, unsigned max,
                bool has_stream, bool has_min_precision)
{
   if (size_B < 8)
      return 0;

   uint32_t count;
   memcpy(&count, base, 4);
   if (count > max)
      count = max;

   unsigned n = 0;
   for (uint32_t i = 0; i < count; i++) {
      /* Legacy SGN records are six dwords. OSG5 prefixes the stream index;
       * SG1 prefixes the stream and appends minimum precision. The semantic
       * fields therefore start one dword later in both extended layouts. */
      const size_t stride = 24 + (has_stream ? 4 : 0) +
                            (has_min_precision ? 4 : 0);
      const size_t fields = has_stream ? 4 : 0;
      size_t off = 8 + (size_t)i * stride;
      if (off + stride > size_B)
         break;

      uint32_t name_off, sem_idx, sysval, component_type, reg;
      memcpy(&name_off, base + off + fields + 0, 4);
      memcpy(&sem_idx, base + off + fields + 4, 4);
      memcpy(&sysval, base + off + fields + 8, 4);
      memcpy(&component_type, base + off + fields + 12, 4);
      memcpy(&reg, base + off + fields + 16, 4);

      struct d12_dxbc_sig_elem *e = &out[n++];
      memset(e, 0, sizeof(*e));
      e->semantic_index = sem_idx;
      e->sysval = sysval;
      e->component_type = component_type;
      e->reg = reg;
      e->mask = base[off + fields + 20];
      if (has_min_precision)
         memcpy(&e->min_precision, base + off + fields + 24, 4);

      if (name_off < size_B) {
         const char *nm = (const char *)base + name_off;
         size_t avail = size_B - name_off;
         size_t len = strnlen(nm, MIN2(avail, sizeof(e->name) - 1));
         memcpy(e->name, nm, len);
         e->name[len] = 0;
      }
   }

   return n;
}

static const struct d12_dxbc_sig_elem *
find_sig(const struct d12_dxbc_sig_elem *sig, unsigned n, uint32_t reg)
{
   for (unsigned i = 0; i < n; i++) {
      if (sig[i].reg == reg)
         return &sig[i];
   }
   return NULL;
}

static const struct d12_dxbc_sig_elem *
find_sig_component(const struct d12_dxbc_sig_elem *sig, unsigned n,
                   uint32_t reg, unsigned component)
{
   for (unsigned i = 0; i < n; i++) {
      if (sig[i].reg == reg && (sig[i].mask & BITFIELD_BIT(component)))
         return &sig[i];
   }

   return find_sig(sig, n, reg);
}

/* Clip/cull signature elements may share a register at arbitrary component
 * masks. Their semantic indices enumerate declarations, not vec4 slots, so
 * `semantic_index * 4 + component` leaves holes (float3 + float is the common
 * counterexample). D3D's system-value ABI is a compact clip array followed by
 * a compact cull array. Return the compact component selected by this physical
 * register component. */
static unsigned
system_value_compact_component(const struct d12_dxbc_sig_elem *sig,
                               unsigned n,
                               const struct d12_dxbc_sig_elem *target,
                               unsigned component)
{
   unsigned base = 0;

   for (unsigned i = 0; i < n; ++i) {
      const struct d12_dxbc_sig_elem *e = &sig[i];
      if (strcasecmp(e->name, target->name))
         continue;

      if (e == target) {
         unsigned prior = e->mask & BITFIELD_MASK(component);
         return base + util_bitcount(prior);
      }

      base += util_bitcount(e->mask);
   }

   return base;
}

static unsigned
system_value_component_count(const struct d12_dxbc_sig_elem *sig, unsigned n,
                             const char *name)
{
   unsigned count = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (!strcasecmp(sig[i].name, name))
         count += util_bitcount(sig[i].mask);
   }
   return count;
}

/* Where a signature element lands in NIR's location space. */
static unsigned
output_location(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_sig_elem *e,
                uint32_t reg)
{
   mesa_shader_stage stage = ctx->b.shader->info.stage;

   if (stage == MESA_SHADER_FRAGMENT) {
      /* Every fragment output is a render target; SV_Depth is the one
       * exception and is named. */
      if (e && !strcasecmp(e->name, "SV_Depth"))
         return FRAG_RESULT_DEPTH;
      return FRAG_RESULT_DATA0 + (e ? e->semantic_index : reg);
   }

   if (e) {
      if (!strcasecmp(e->name, "SV_Position") || e->sysval == 1)
         return VARYING_SLOT_POS;
      if (!strcasecmp(e->name, "SV_RenderTargetArrayIndex"))
         return VARYING_SLOT_LAYER;
      if (!strcasecmp(e->name, "SV_ClipDistance"))
         return VARYING_SLOT_CLIP_DIST0 + e->semantic_index;
   }

   return VARYING_SLOT_VAR0 + reg;
}

static unsigned
input_location(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_sig_elem *e,
               uint32_t reg)
{
   if (ctx->b.shader->info.stage == MESA_SHADER_VERTEX)
      return VERT_ATTRIB_GENERIC0 + reg;

   if (e) {
      if (!strcasecmp(e->name, "SV_Position") || e->sysval == 1)
         return VARYING_SLOT_POS;
      if (!strcasecmp(e->name, "SV_RenderTargetArrayIndex"))
         return VARYING_SLOT_LAYER;
      if (!strcasecmp(e->name, "SV_ClipDistance"))
         return VARYING_SLOT_CLIP_DIST0 + e->semantic_index;
   }

   return VARYING_SLOT_VAR0 + reg;
}

static unsigned
patch_location(const struct d12_dxbc_sig_elem *e, uint32_t reg)
{
   if (e) {
      if (!strcasecmp(e->name, "SV_TessFactor"))
         return VARYING_SLOT_TESS_LEVEL_OUTER;
      if (!strcasecmp(e->name, "SV_InsideTessFactor"))
         return VARYING_SLOT_TESS_LEVEL_INNER;
   }

   return VARYING_SLOT_PATCH0 + reg;
}

/*
 * Decode one operand. Returns the number of tokens consumed, or 0 on a
 * malformed stream -- which is a hard error, since a miscounted operand
 * desynchronises everything after it.
 */
static unsigned
decode_operand(struct d12_dxbc_ctx *ctx, const uint32_t *p, unsigned avail,
               struct d12_dxbc_operand *out)
{
   if (!avail)
      return 0;

   memset(out, 0, sizeof(*out));

   uint32_t tok = p[0];
   unsigned used = 1;

   out->num_components = tok & 0x3;
   out->sel_mode = (tok >> 2) & 0x3;
   unsigned sel = (tok >> 4) & 0xff;
   out->file = (tok >> 12) & 0xff;
   out->num_indices = (tok >> 20) & 0x3;

   if (out->num_components == 1) {
      /* A one-component operand has no selection field. Represent it as
       * an x scalar replicated to the vec4 register shape used here. */
      out->swizzle[0] = out->swizzle[1] = out->swizzle[2] =
         out->swizzle[3] = 0;
      out->mask = 1;
   } else if (out->num_components == 2) {
      switch (out->sel_mode) {
      case 0: /* destination mask */
         out->mask = sel & 0xf;
         for (unsigned i = 0; i < 4; i++)
            out->swizzle[i] = i;
         break;
      case 1: /* source swizzle */
         out->mask = 0xf;
         for (unsigned i = 0; i < 4; i++)
            out->swizzle[i] = (sel >> (i * 2)) & 0x3;
         break;
      case 2: { /* source select-one */
         unsigned c = sel & 0x3;
         out->mask = 1 << c;
         out->swizzle[0] = out->swizzle[1] = out->swizzle[2] =
            out->swizzle[3] = c;
         break;
      }
      default:
         return 0;
      }
   } else if (out->num_components == 0) {
      out->mask = 0xf;
      for (unsigned i = 0; i < 4; i++)
         out->swizzle[i] = i;
   } else {
      /* N-component operands are reserved in the public token format. */
      return 0;
   }

   /* Extended operand token: modifiers (negate / absolute). */
   if (tok & 0x80000000u) {
      if (used >= avail)
         return 0;
      uint32_t ext = p[used++];
      if ((ext & 0x3f) == 1) {
         unsigned mod = (ext >> 6) & 0xff;
         out->neg = mod == 1 || mod == 3;
         out->abs = mod == 2 || mod == 3;
      }
   }

   if (out->file == D12_DXBC_FILE_IMMEDIATE32) {
      unsigned n = out->num_components == 1 ? 1 : 4;
      if (used + n > avail)
         return 0;
      for (unsigned i = 0; i < n; i++)
         out->imm[i] = p[used++];
      if (n == 1)
         out->imm[1] = out->imm[2] = out->imm[3] = out->imm[0];
      return used;
   }

   for (unsigned i = 0; i < out->num_indices; i++) {
      unsigned rep = (tok >> (22 + i * 3)) & 0x7;

      switch (rep) {
      case 0: /* immediate32 */
         if (used >= avail)
            return 0;
         out->index[i] = p[used++];
         break;
      case 2: /* relative */
         out->index_is_relative[i] = true;
         {
            struct d12_dxbc_operand rel;
            unsigned n = decode_operand(ctx, p + used, avail - used, &rel);
            if (!n)
               return 0;
            used += n;

            /* Relative indices are integer register values. Float
             * source modifiers have no defined place in address math. */
            if (rel.neg || rel.abs)
               return 0;

            for (unsigned j = 0; j < rel.num_indices; j++) {
               if (rel.index_is_relative[j])
                  return 0;
            }

            out->relative[i] = (struct d12_dxbc_relative_operand) {
               .valid = true,
               .file = rel.file,
               .num_indices = rel.num_indices,
               .num_components = rel.num_components,
               .sel_mode = rel.sel_mode,
               .mask = rel.mask,
               .neg = rel.neg,
               .abs = rel.abs,
            };
            memcpy(out->relative[i].index, rel.index,
                   sizeof(out->relative[i].index));
            memcpy(out->relative[i].swizzle, rel.swizzle,
                   sizeof(out->relative[i].swizzle));
            memcpy(out->relative[i].imm, rel.imm,
                   sizeof(out->relative[i].imm));
         }
         break;
      case 3: /* immediate32 + relative */
         if (used >= avail)
            return 0;
         out->index[i] = p[used++];
         out->index_is_relative[i] = true;
         {
            struct d12_dxbc_operand rel;
            unsigned n = decode_operand(ctx, p + used, avail - used, &rel);
            if (!n)
               return 0;
            used += n;

            if (rel.neg || rel.abs)
               return 0;

            for (unsigned j = 0; j < rel.num_indices; j++) {
               if (rel.index_is_relative[j])
                  return 0;
            }

            out->relative[i] = (struct d12_dxbc_relative_operand) {
               .valid = true,
               .file = rel.file,
               .num_indices = rel.num_indices,
               .num_components = rel.num_components,
               .sel_mode = rel.sel_mode,
               .mask = rel.mask,
               .neg = rel.neg,
               .abs = rel.abs,
            };
            memcpy(out->relative[i].index, rel.index,
                   sizeof(out->relative[i].index));
            memcpy(out->relative[i].swizzle, rel.swizzle,
                   sizeof(out->relative[i].swizzle));
            memcpy(out->relative[i].imm, rel.imm,
                   sizeof(out->relative[i].imm));
         }
         break;
      default:
         return 0;
      }
   }

   return used;
}

static nir_variable *
temp_var(struct d12_dxbc_ctx *ctx, unsigned index)
{
   if (index >= D12_DXBC_MAX_TEMPS) {
      fail(ctx, "temporary register out of range");
      return NULL;
   }

   if (!ctx->temps[index]) {
      ctx->temps[index] = nir_local_variable_create(
         ctx->b.impl, glsl_vec4_type(), "r");
   }

   return ctx->temps[index];
}

static nir_def *read_src_typed(struct d12_dxbc_ctx *ctx,
                               const struct d12_dxbc_operand *op,
                               bool integer);

static nir_def *
read_src(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *op)
{
   return read_src_typed(ctx, op, false);
}

static nir_def *
read_src_int(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *op)
{
   return read_src_typed(ctx, op, true);
}

static nir_def *
emit_derivative(struct d12_dxbc_ctx *ctx, unsigned opcode,
                const struct d12_dxbc_operand *src)
{
   nir_builder *b = &ctx->b;
   bool hoist = src->file == D12_DXBC_FILE_INPUT &&
                (ctx->cf_depth || ctx->conditional_return_if);

   /* AGX evaluates derivatives using the currently active quad lanes. D3D
    * keeps direct pixel-input derivatives valid across structured divergent
    * branches, so place that pure input/derivative chain before the outermost
    * if. Restrict this to direct inputs and all-if nesting: temporaries and
    * loop-carried values need real dependence analysis rather than guessing. */
   for (unsigned i = 0; hoist && i < ctx->cf_depth; i++)
      hoist &= ctx->cf[i].kind == D12_DXBC_CF_IF && ctx->cf[i].nif;

   nir_if *outer = ctx->conditional_return_if
                      ? ctx->conditional_return_if
                      : ctx->cf[0].nif;
   nir_cursor saved = b->cursor;
   if (hoist)
      b->cursor = nir_before_cf_node(&outer->cf_node);

   nir_def *value = read_src(ctx, src);
   switch (opcode) {
   case D12_DXBC_OP_DERIV_RTX:        value = nir_ddx(b, value); break;
   case D12_DXBC_OP_DERIV_RTY:        value = nir_ddy(b, value); break;
   case D12_DXBC_OP_DERIV_RTX_COARSE: value = nir_ddx_coarse(b, value); break;
   case D12_DXBC_OP_DERIV_RTX_FINE:   value = nir_ddx_fine(b, value); break;
   case D12_DXBC_OP_DERIV_RTY_COARSE: value = nir_ddy_coarse(b, value); break;
   case D12_DXBC_OP_DERIV_RTY_FINE:   value = nir_ddy_fine(b, value); break;
   default: UNREACHABLE("not a derivative opcode");
   }

   if (hoist)
      b->cursor = saved;

   return value;
}

/* Materialize one immediate, relative, or immediate-plus-relative operand
 * index.  SM 5.1 permits this independently for the selected cbuffer and
 * for the vector within it. */
static nir_def *
read_operand_index(struct d12_dxbc_ctx *ctx,
                   const struct d12_dxbc_operand *op, unsigned index)
{
   nir_builder *b = &ctx->b;
   nir_def *value = nir_imm_int(b, op->index[index]);

   if (!op->index_is_relative[index])
      return value;

   const struct d12_dxbc_relative_operand *ri = &op->relative[index];
   if (!ri->valid) {
      fail(ctx, "missing relative operand index");
      return NULL;
   }

   struct d12_dxbc_operand rel = {
      .file = ri->file,
      .num_indices = ri->num_indices,
      .num_components = ri->num_components,
      .sel_mode = ri->sel_mode,
      .mask = ri->mask,
      .neg = ri->neg,
      .abs = ri->abs,
   };
   memcpy(rel.index, ri->index, sizeof(rel.index));
   memcpy(rel.swizzle, ri->swizzle, sizeof(rel.swizzle));
   memcpy(rel.imm, ri->imm, sizeof(rel.imm));

   nir_def *relative = read_src(ctx, &rel);
   if (ctx->error)
      return NULL;

   return nir_iadd(b, value, nir_channel(b, relative, 0));
}

static const struct d12_dxbc_tgsm_decl *
tgsm_decl(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *op)
{
   if (op->file != D12_DXBC_FILE_TGSM || op->num_indices != 1 ||
       op->index_is_relative[0] || op->index[0] >= D12_DXBC_MAX_SIG ||
       !ctx->tgsm[op->index[0]].valid) {
      fail(ctx, "invalid thread-group shared-memory operand");
      return NULL;
   }

   return &ctx->tgsm[op->index[0]];
}

/* Return a byte offset in NIR's flat shared-memory address space. A raw g#
 * consumes the byte offset directly; a structured g# consumes
 * element * declared_stride + member_offset. */
static nir_def *
tgsm_byte_offset(struct d12_dxbc_ctx *ctx,
                 const struct d12_dxbc_operand *op, nir_def *index,
                 nir_def *offset)
{
   nir_builder *b = &ctx->b;
   const struct d12_dxbc_tgsm_decl *decl = tgsm_decl(ctx, op);
   if (!decl)
      return NULL;

   nir_def *byte = nir_channel(b, offset, 0);
   if (decl->stride) {
      byte = nir_iadd(
         b, nir_imul_imm(b, nir_channel(b, index, 0), decl->stride), byte);
   }

   return nir_iadd_imm(b, byte, decl->base);
}

static nir_def *
tgsm_atomic_offset(struct d12_dxbc_ctx *ctx,
                   const struct d12_dxbc_operand *op, nir_def *coord)
{
   const struct d12_dxbc_tgsm_decl *decl = tgsm_decl(ctx, op);
   if (!decl)
      return NULL;

   nir_builder *b = &ctx->b;
   nir_def *index = decl->stride ? coord : nir_imm_int(b, 0);
   nir_def *offset =
      decl->stride ? nir_channel(b, coord, 1) : nir_channel(b, coord, 0);
   return tgsm_byte_offset(ctx, op, index, offset);
}

/* DXBC loads carry a destination mask and a resource swizzle. Load only the
 * physical dwords that can reach a written destination component. Besides
 * avoiding needless traffic, this keeps a scalar load from reading beyond
 * the final member of a tightly packed shared array. */
static nir_def *
tgsm_load_vec4(struct d12_dxbc_ctx *ctx,
               const struct d12_dxbc_operand *resource, nir_def *byte,
               unsigned dst_mask)
{
   nir_builder *b = &ctx->b;
   nir_def *component[4] = {
      nir_imm_int(b, 0), nir_imm_int(b, 0),
      nir_imm_int(b, 0), nir_imm_int(b, 0),
   };
   bool loaded[4] = {false};

   dst_mask = dst_mask ? dst_mask : 0xf;
   u_foreach_bit(c, dst_mask) {
      unsigned physical = resource->swizzle[c];
      if (loaded[physical])
         continue;

      component[physical] = nir_load_shared(
         b, 1, 32, nir_iadd_imm(b, byte, physical * 4), .align_mul = 4);
      loaded[physical] = true;
   }

   return nir_vec4(b, component[0], component[1], component[2],
                   component[3]);
}

/*
 * Select one of the root-table planes. The first-level header keeps
 * the hot resource plane direct; its second pointer names a compact metadata
 * header containing the counter and BUFINFO planes.
 */
static nir_def *
root_plane(struct d12_dxbc_ctx *ctx, unsigned plane)
{
   nir_builder *b = &ctx->b;
   nir_def *root = nir_load_root_agx(b);
   nir_def *header = nir_load_global(b, 3, 64, root, .align_mul = 8);

   if (plane == D12_ROOT_RESOURCE_BASE)
      return nir_channel(b, header, 0);
   if (plane == D12_ROOT_SAMPLER_BASE)
      return nir_channel(b, header, 2);

   nir_def *metadata = nir_load_global(
      b, 2, 64, nir_channel(b, header, 1), .align_mul = 8);

   if (plane == D12_ROOT_COUNTER_BASE)
      return nir_channel(b, metadata, 0);
   if (plane == D12_ROOT_BUFINFO_BASE)
      return nir_channel(b, metadata, 1);

   fail(ctx, "invalid root-table plane");
   return nir_imm_int64(b, 0);
}

/* Byte offset of the permanently-null PBE descriptor appended to the
 * shader-visible heap.  Keep this separate from root_plane(): that helper
 * returns GPU addresses, while an AGX bindless image handle is an offset
 * from its uniform-selected descriptor heap. */
static nir_def *
root_heap_pbe_null_offset(struct d12_dxbc_ctx *ctx)
{
   nir_builder *b = &ctx->b;
   nir_def *root = nir_load_root_agx(b);
   nir_def *header = nir_load_global(b, 4, 64, root, .align_mul = 8);
   return nir_u2u32(b, nir_channel(b, header, 3));
}

/* Read an operand as a vec4 of 32-bit values, swizzle and modifiers applied.
 * Registers are typeless, but neg/abs are not: integer ALU instructions need
 * integer modifiers rather than flipping the IEEE-754 sign bit. */
static nir_def *
read_src_typed(struct d12_dxbc_ctx *ctx,
               const struct d12_dxbc_operand *op, bool integer)
{
   nir_builder *b = &ctx->b;
   nir_def *v = NULL;

   switch (op->file) {
   case D12_DXBC_FILE_IMMEDIATE32: {
      nir_const_value c[4];
      for (unsigned i = 0; i < 4; i++)
         c[i] = nir_const_value_for_uint(op->imm[i], 32);
      v = nir_build_imm(b, 4, 32, c);
      break;
   }

   case D12_DXBC_FILE_IMMEDIATE_CONSTANT_BUFFER: {
      if (op->num_indices != 1 || !ctx->immediate_constants ||
          !ctx->num_immediate_constants) {
         fail(ctx, "invalid immediate constant-buffer operand");
         return nir_imm_zero(b, 4, 32);
      }

      nir_def *index = read_operand_index(ctx, op, 0);
      if (!index)
         return nir_imm_zero(b, 4, 32);

      nir_def *result = nir_imm_zero(b, 4, 32);
      for (unsigned i = 0; i < ctx->num_immediate_constants; i++) {
         nir_const_value c[4];
         for (unsigned j = 0; j < 4; j++)
            c[j] = nir_const_value_for_uint(
               ctx->immediate_constants[i * 4 + j], 32);
         nir_def *value = nir_build_imm(b, 4, 32, c);
         result = nir_bcsel(b, nir_ieq_imm(b, index, i), value, result);
      }
      v = result;
      break;
   }

   case D12_DXBC_FILE_TEMP: {
      nir_variable *var = temp_var(ctx, op->index[0]);
      if (!var)
         return nir_imm_zero(b, 4, 32);
      v = nir_load_var(b, var);
      break;
   }

   case D12_DXBC_FILE_INDEXABLE_TEMP: {
      if (op->num_indices != 2 || op->index_is_relative[0] ||
          op->index[0] >= D12_DXBC_MAX_SIG ||
          !ctx->indexable_temps[op->index[0]]) {
         fail(ctx, "invalid indexable temporary source");
         return nir_imm_zero(b, 4, 32);
      }

      nir_def *index = read_operand_index(ctx, op, 1);
      if (!index)
         return nir_imm_zero(b, 4, 32);

      /* fxc commonly materializes a small source array as an indexable
       * temporary and then reads it under divergent fragment control flow.
       * When every element has a dominating top-level SSA store, select the
       * values directly. This avoids lowering a dynamic local-array read to
       * AGX fragment stack traffic without weakening general memory
       * semantics. */
      unsigned file = op->index[0];
      nir_def **shadow = ctx->indexable_temp_shadow[file];
      unsigned count = ctx->indexable_temp_size[file];
      bool complete = ctx->indexable_temp_shadow_valid[file] && shadow;
      for (unsigned i = 0; complete && i < count; i++)
         complete &= shadow[i] != NULL;

      if (complete) {
         if (nir_src_is_const(nir_src_for_ssa(index))) {
            unsigned element = nir_src_as_uint(nir_src_for_ssa(index));
            if (element < count) {
               v = shadow[element];
               break;
            }
         } else {
            v = nir_undef(b, 4, 32);
            for (unsigned i = 0; i < count; i++)
               v = nir_bcsel(b, nir_ieq_imm(b, index, i), shadow[i], v);
            break;
         }
      }

      nir_deref_instr *array =
         nir_build_deref_var(b, ctx->indexable_temps[op->index[0]]);
      v = nir_load_deref(b, nir_build_deref_array(b, array, index));
      break;
   }

   /*
    * A compute shader's thread identifiers. Unlike a vertex or pixel
    * input these carry no declaration and no signature entry: the
    * register FILE is the whole meaning, which is why they arrive here
    * as "unsupported register file" rather than as a missing opcode.
    */
   case D12_DXBC_FILE_THREAD_ID:
      v = nir_pad_vector(b, nir_load_global_invocation_id(b, 32), 4);
      break;

   case D12_DXBC_FILE_THREAD_GROUP_ID:
      v = nir_pad_vector(b, nir_load_workgroup_id(b), 4);
      break;

   case D12_DXBC_FILE_THREAD_ID_IN_GROUP:
      v = nir_pad_vector(b, nir_load_local_invocation_id(b), 4);
      break;

   case D12_DXBC_FILE_THREAD_ID_IN_GROUP_FLAT:
      return nir_replicate(b, nir_load_local_invocation_index(b), 4);

   case D12_DXBC_FILE_COVERAGE_MASK:
      return nir_replicate(b, nir_load_sample_mask_in(b), 4);

   case D12_DXBC_FILE_INPUT_PRIMITIVE_ID:
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         nir_def *id = nir_load_input(
            b, 1, 32, nir_imm_int(b, 0), .base = 0,
            .dest_type = nir_type_uint32,
            .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
            .io_semantics.num_slots = 1);
         b->shader->info.inputs_read |= VARYING_BIT_PRIMITIVE_ID;
         return nir_replicate(b, id, 4);
      }
      return nir_replicate(b, nir_load_primitive_id(b), 4);

   case D12_DXBC_FILE_OUTPUT_CONTROL_POINT_ID:
      return nir_replicate(b, nir_load_invocation_id(b), 4);

   /* Fork/join phases are distributed over the hull workgroup's invocations;
    * their declaration wraps the phase body to the requested instance count. */
   case D12_DXBC_FILE_INPUT_FORK_INSTANCE_ID:
   case D12_DXBC_FILE_INPUT_JOIN_INSTANCE_ID:
      return nir_replicate(b, nir_load_invocation_id(b), 4);

   case D12_DXBC_FILE_INPUT_CONTROL_POINT: {
      if (op->num_indices != 2 || op->index_is_relative[1]) {
         fail(ctx, "invalid input control-point operand");
         return nir_imm_zero(b, 4, 32);
      }

      nir_def *vertex = read_operand_index(ctx, op, 0);
      if (!vertex)
         return nir_imm_zero(b, 4, 32);

      uint32_t reg = op->index[1];
      const struct d12_dxbc_sig_elem *e =
         find_sig(ctx->in_sig, ctx->num_in_sig, reg);
      unsigned loc = input_location(ctx, e, reg);
      v = nir_load_per_vertex_input(
         b, 4, 32, nir_u2u32(b, vertex), nir_imm_int(b, 0), .base = reg,
         .dest_type = nir_type_float32, .io_semantics.location = loc,
         .io_semantics.num_slots = 1);
      b->shader->info.inputs_read |= BITFIELD64_BIT(loc);
      break;
   }

   case D12_DXBC_FILE_OUTPUT_CONTROL_POINT: {
      if (op->num_indices != 2 || op->index_is_relative[1]) {
         fail(ctx, "invalid output control-point operand");
         return nir_imm_zero(b, 4, 32);
      }

      nir_def *vertex = read_operand_index(ctx, op, 0);
      if (!vertex)
         return nir_imm_zero(b, 4, 32);

      uint32_t reg = op->index[1];
      const struct d12_dxbc_sig_elem *e =
         find_sig(ctx->out_sig, ctx->num_out_sig, reg);
      unsigned loc = output_location(ctx, e, reg);
      v = nir_load_per_vertex_output(
         b, 4, 32, nir_u2u32(b, vertex), nir_imm_int(b, 0), .base = reg,
         .dest_type = nir_type_float32, .io_semantics.location = loc,
         .io_semantics.num_slots = 1);
      break;
   }

   case D12_DXBC_FILE_INPUT_PATCH_CONSTANT: {
      if (op->num_indices != 1 || op->index_is_relative[0]) {
         fail(ctx, "invalid patch-constant operand");
         return nir_imm_zero(b, 4, 32);
      }

      uint32_t reg = op->index[0];
      nir_def *components[4];
      for (unsigned c = 0; c < 4; c++) {
         const struct d12_dxbc_sig_elem *e = find_sig_component(
            ctx->patch_sig, ctx->num_patch_sig, reg, c);
         unsigned loc = patch_location(e, reg);
         unsigned component =
            loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
                  loc == VARYING_SLOT_TESS_LEVEL_INNER
               ? e->semantic_index
               : c;

         components[c] = nir_load_input(
            b, 1, 32, nir_imm_int(b, 0), .base = reg,
            .component = component, .dest_type = nir_type_float32,
            .io_semantics.location = loc, .io_semantics.num_slots = 1);
         if (loc >= VARYING_SLOT_PATCH0)
            b->shader->info.patch_inputs_read |=
               BITFIELD64_BIT(loc - VARYING_SLOT_PATCH0);
         else
            b->shader->info.inputs_read |= BITFIELD64_BIT(loc);
      }
      v = nir_vec(b, components, ARRAY_SIZE(components));
      break;
   }

   case D12_DXBC_FILE_INPUT_DOMAIN_POINT:
      v = nir_pad_vector(b, nir_load_tess_coord(b), 4);
      break;

   case D12_DXBC_FILE_INPUT: {
      uint32_t reg = op->index[op->num_indices ? op->num_indices - 1 : 0];
      const struct d12_dxbc_sig_elem *e =
         find_sig(ctx->in_sig, ctx->num_in_sig, reg);

      /* System values are not varyings: they have their own intrinsics,
       * and reading them as inputs gives the shader a slot nothing ever
       * writes. */
      if (e) {
         if (!strcasecmp(e->name, "SV_VertexID"))
            return nir_replicate(b, nir_load_vertex_id(b), 4);
         if (!strcasecmp(e->name, "SV_InstanceID"))
            return nir_replicate(b, nir_load_instance_id(b), 4);
         if (!strcasecmp(e->name, "SV_IsFrontFace"))
            return nir_replicate(
               b, nir_ineg(b, nir_b2i32(b, nir_load_front_face(b, 1))), 4);
         if (!strcasecmp(e->name, "SV_PrimitiveID") &&
             b->shader->info.stage == MESA_SHADER_FRAGMENT) {
            nir_def *id = nir_load_input(
               b, 1, 32, nir_imm_int(b, 0), .base = reg,
               .dest_type = nir_type_uint32,
               .io_semantics.location = VARYING_SLOT_PRIMITIVE_ID,
               .io_semantics.num_slots = 1);
            b->shader->info.inputs_read |= VARYING_BIT_PRIMITIVE_ID;
            return nir_replicate(b, id, 4);
         }
         if (!strcasecmp(e->name, "SV_PrimitiveID"))
            return nir_replicate(b, nir_load_primitive_id(b), 4);
         if (!strcasecmp(e->name, "SV_SampleIndex")) {
            b->shader->info.fs.uses_sample_shading = true;
            return nir_replicate(b, nir_load_sample_id(b), 4);
         }
         if (!strcasecmp(e->name, "SV_RenderTargetArrayIndex") &&
             b->shader->info.stage == MESA_SHADER_FRAGMENT)
            return nir_replicate(b, nir_load_layer_id(b), 4);
      }

      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         if (e && !strcasecmp(e->name, "SV_Position")) {
            v = nir_load_frag_coord(b);
            /* NIR frag_coord.w is reciprocal clip-W; D3D SV_Position.w
             * exposes clip-W itself. */
            v = nir_vector_insert_imm(
               b, v, nir_frcp(b, nir_channel(b, v, 3)), 3);
            break;
         }

         /* A fragment varying must arrive INTERPOLATED: the backend
          * consumes load_interpolated_input with a barycentric, not a
          * raw load_input. */
         unsigned loc = VARYING_SLOT_VAR0 + reg;
         const unsigned mode = reg < D12_DXBC_MAX_SIG
                                  ? ctx->input_interp[reg]
                                  : 0;
         if (mode == 1) {
            v = nir_load_input(
               b, 4, 32, nir_imm_int(b, 0), .base = reg,
               .dest_type = nir_type_uint32,
               .io_semantics.location = loc,
               .io_semantics.num_slots = 1);
            b->shader->info.inputs_read |= BITFIELD64_BIT(loc);
            break;
         }
         const enum glsl_interp_mode interp =
            mode == 4 || mode == 5 || mode == 7
               ? INTERP_MODE_NOPERSPECTIVE
               : INTERP_MODE_SMOOTH;
         if (mode == 6 || mode == 7)
            b->shader->info.fs.uses_sample_shading = true;
         nir_def *bary = mode == 3 || mode == 5
                            ? nir_load_barycentric_centroid(
                                 b, 32, .interp_mode = interp)
                         : mode == 6 || mode == 7
                            ? nir_load_barycentric_sample(
                                 b, 32, .interp_mode = interp)
                            : nir_load_barycentric_pixel(
                                 b, 32, .interp_mode = interp);
         v = nir_load_interpolated_input(
            b, 4, 32, bary, nir_imm_int(b, 0),
            .base = reg, .dest_type = nir_type_float32,
            .io_semantics.location = loc, .io_semantics.num_slots = 1);
         b->shader->info.inputs_read |= BITFIELD64_BIT(loc);
         break;
      }

      unsigned loc = input_location(ctx, e, reg);

      if (b->shader->info.stage == MESA_SHADER_GEOMETRY ||
          b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
          b->shader->info.stage == MESA_SHADER_TESS_EVAL) {
         if (op->num_indices != 2) {
            fail(ctx, "invalid arrayed shader input operand");
            return nir_imm_zero(b, 4, 32);
         }
         nir_def *vertex = read_operand_index(ctx, op, 0);
         if (!vertex)
            return nir_imm_zero(b, 4, 32);
         v = nir_load_per_vertex_input(
            b, 4, 32, nir_u2u32(b, vertex), nir_imm_int(b, 0), .base = reg,
            .dest_type = nir_type_float32, .io_semantics.location = loc,
            .io_semantics.num_slots = 1);
      } else {
         v = nir_load_input(b, 4, 32, nir_imm_int(b, 0), .base = reg,
                            .dest_type = nir_type_float32,
                            .io_semantics.location = loc,
                            .io_semantics.num_slots = 1);
      }
      b->shader->info.inputs_read |=
         BITFIELD64_BIT(b->shader->info.stage == MESA_SHADER_VERTEX ? reg
                                                                    : loc);
      break;
   }

   case D12_DXBC_FILE_CONSTANT_BUFFER: {
      /*
       * SM 4/5.0 uses cb<slot>[<row>].  SM 5.1 uses
       * cb<range-id>[<absolute-register>][<row>], and permits both the
       * register and row indices to be relative.
       *
       * This driver does not have UBOs: a constant buffer is reached
       * through the ROOT TABLE, one indirection to the address the bind
       * put in the slot -- exactly what the DXIL front end does for
       * cbufferLoadLegacy. Emitting nir_load_ubo instead produced NIR the
       * AGX backend has no lowering for, and it asserted.
       *
       * The declaration supplies the register space and range bounds for
       * SM 5.1.  Dynamic registers are legal only when the declared range
       * maps to one contiguous root-signature descriptor range, matching
       * the validation used by the DXIL front end.
       */
      if (op->num_indices != 2 && op->num_indices != 3) {
         fail(ctx, "invalid constant buffer operand dimensions");
         return nir_imm_zero(b, 4, 32);
      }

      const bool sm51 = op->num_indices == 3;
      const unsigned binding_index = sm51 ? 1 : 0;
      const unsigned row_index = sm51 ? 2 : 1;
      uint32_t space = 0;
      const struct d12_dxbc_binding_range *range = NULL;

      if (sm51) {
         if (op->index_is_relative[0] ||
             op->index[0] >= D12_DXBC_MAX_SIG ||
             !ctx->cb_range[op->index[0]].valid) {
            fail(ctx, "undeclared SM 5.1 constant buffer range");
            return nir_imm_zero(b, 4, 32);
         }
         range = &ctx->cb_range[op->index[0]];
         space = range->space;
      } else if (op->index_is_relative[binding_index]) {
         fail(ctx, "dynamically selected SM 5.0 constant buffer");
         return nir_imm_zero(b, 4, 32);
      }

      bool rooted_heap =
         sm51 && ctx->map &&
         d12_binding_map_unbounded(ctx->map, D12_RES_CBV, space,
                                   range->lower_bound);
      unsigned rooted_base_slot = ~0u;
      nir_def *descriptor_index = NULL;
      nir_def *slot = NULL;

      if (rooted_heap) {
         rooted_base_slot = d12_binding_map_find(
            ctx->map, D12_RES_CBV, space, range->lower_bound);
         if (rooted_base_slot == ~0u) {
            fail(ctx, "unbounded constant buffer range has no root base");
            return nir_imm_zero(b, 4, 32);
         }

         if (!op->index_is_relative[binding_index]) {
            uint32_t reg = op->index[binding_index];
            if (reg < range->lower_bound ||
                (range->upper_bound != UINT32_MAX &&
                 reg > range->upper_bound)) {
               fail(ctx, "constant buffer register is outside its declared "
                         "range");
               return nir_imm_zero(b, 4, 32);
            }
            descriptor_index = nir_imm_int(b, reg - range->lower_bound);
         } else {
            nir_def *reg = read_operand_index(ctx, op, binding_index);
            if (!reg)
               return nir_imm_zero(b, 4, 32);
            descriptor_index =
               nir_iadd_imm(b, reg, -(int64_t)range->lower_bound);
         }
      } else if (!op->index_is_relative[binding_index]) {
         if (sm51 &&
             (op->index[binding_index] < range->lower_bound ||
              (range->upper_bound != UINT32_MAX &&
               op->index[binding_index] > range->upper_bound))) {
            fail(ctx, "constant buffer register is outside its declared range");
            return nir_imm_zero(b, 4, 32);
         }

         unsigned static_slot =
            ctx->map ? d12_binding_map_find(ctx->map, D12_RES_CBV, space,
                                             op->index[binding_index])
                     : op->index[binding_index];
         if (static_slot == ~0u) {
            fail(ctx, "constant buffer is not bound by the root signature");
            return nir_imm_zero(b, 4, 32);
         }
         slot = nir_imm_int(b, static_slot);
      } else {
         assert(sm51 && range);

         unsigned first_slot =
            ctx->map ? d12_binding_map_find(ctx->map, D12_RES_CBV, space,
                                             range->lower_bound)
                     : range->lower_bound;
         if (first_slot == ~0u) {
            fail(ctx, "constant buffer range is not bound by the root signature");
            return nir_imm_zero(b, 4, 32);
         }

         if (ctx->map && range->upper_bound != UINT32_MAX) {
            uint64_t count =
               (uint64_t)range->upper_bound - range->lower_bound + 1;
            if (range->upper_bound < range->lower_bound ||
                count > D12_MAX_RESOURCE_BINDINGS) {
               fail(ctx, "invalid constant buffer range bounds");
               return nir_imm_zero(b, 4, 32);
            }

            unsigned last_slot = d12_binding_map_find(
               ctx->map, D12_RES_CBV, space, range->upper_bound);
            unsigned root_range = d12_binding_map_root_range(
               ctx->map, D12_RES_CBV, space, range->lower_bound);
            if (last_slot != first_slot + (unsigned)count - 1 ||
                root_range == ~0u ||
                d12_binding_map_root_range(ctx->map, D12_RES_CBV, space,
                                           range->upper_bound) != root_range) {
               fail(ctx, "constant buffer array crosses root descriptor ranges");
               return nir_imm_zero(b, 4, 32);
            }
         }

         nir_def *reg = read_operand_index(ctx, op, binding_index);
         if (!reg)
            return nir_imm_zero(b, 4, 32);
         slot = nir_iadd_imm(b, reg,
                            (int64_t)first_slot - range->lower_bound);
      }

      nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
      nir_def *base;
      if (rooted_heap) {
         nir_def *heap = nir_load_global(
            b, 1, 64,
            nir_iadd_imm(b, resource_table,
                         rooted_base_slot * sizeof(uint64_t)),
            .align_mul = 8);
         nir_def *descriptor = nir_iadd(
            b, heap,
            nir_imul_imm(b, nir_u2u64(b, descriptor_index),
                         D12_HEAP_DESC_SIZE));
         base = nir_load_global(b, 1, 64, descriptor, .align_mul = 8);
      } else {
         base = nir_load_global(
            b, 1, 64,
            nir_iadd(b, resource_table,
                     nir_imul_imm(b, nir_u2u64(b, slot), 8)),
            .align_mul = 8);
      }

      nir_def *row = read_operand_index(ctx, op, row_index);
      if (!row)
         return nir_imm_zero(b, 4, 32);

      nir_def *byte_offset = nir_u2u64(b, nir_imul_imm(b, row, 16));
      v = nir_load_global(b, 4, 32, nir_iadd(b, base, byte_offset),
                          .align_mul = 16);
      break;
   }

   default:
      snprintf(ctx->errbuf, sizeof(ctx->errbuf),
               "unsupported source register file %u", op->file);
      fail(ctx, ctx->errbuf);
      return nir_imm_zero(b, 4, 32);
   }

   if (!v)
      return nir_imm_zero(b, 4, 32);

   unsigned sw[4] = {op->swizzle[0], op->swizzle[1], op->swizzle[2],
                     op->swizzle[3]};
   v = nir_swizzle(b, v, sw, 4);

   if (op->abs)
      v = integer ? nir_iabs(b, v) : nir_fabs(b, v);
   if (op->neg)
      v = integer ? nir_ineg(b, v) : nir_fneg(b, v);

   return v;
}

static nir_def *
read_interpolated_src(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *op, nir_def *bary)
{
   nir_builder *b = &ctx->b;
   if (b->shader->info.stage != MESA_SHADER_FRAGMENT ||
       op->file != D12_DXBC_FILE_INPUT || op->num_indices != 1 ||
       op->index_is_relative[0]) {
      fail(ctx, "invalid evaluated fragment input");
      return nir_imm_zero(b, 4, 32);
   }

   const uint32_t reg = op->index[0];
   const struct d12_dxbc_sig_elem *e =
      find_sig(ctx->in_sig, ctx->num_in_sig, reg);
   if (e && !strncasecmp(e->name, "SV_", 3)) {
      fail(ctx, "cannot evaluate a fragment system value");
      return nir_imm_zero(b, 4, 32);
   }

   const unsigned loc = VARYING_SLOT_VAR0 + reg;
   nir_def *v = nir_load_interpolated_input(
      b, 4, 32, bary, nir_imm_int(b, 0), .base = reg,
      .dest_type = nir_type_float32, .io_semantics.location = loc,
      .io_semantics.num_slots = 1);
   b->shader->info.inputs_read |= BITFIELD64_BIT(loc);

   unsigned sw[4] = {op->swizzle[0], op->swizzle[1], op->swizzle[2],
                     op->swizzle[3]};
   v = nir_swizzle(b, v, sw, 4);
   if (op->abs)
      v = nir_fabs(b, v);
   if (op->neg)
      v = nir_fneg(b, v);
   return v;
}

static nir_def *
uav_binding_slot(struct d12_dxbc_ctx *ctx,
                 const struct d12_dxbc_operand *uav);
static nir_def *
buffer_element_in_bounds_slot(struct d12_dxbc_ctx *ctx, nir_def *slot,
                              nir_def *index);

/* Shared with the DXIL translator: resolve a shader register to the rooted
 * descriptor-array base that actually contains it, honouring the extent of a
 * range the root signature sized explicitly. */
static bool
unbounded_root_for_register(const struct d12_binding_map *map,
                            unsigned klass, unsigned space, unsigned reg,
                            unsigned *out_slot, unsigned *out_reg)
{
   return d12_binding_map_rooted_base(map, klass, space, reg, out_slot,
                                      out_reg);
}

/* Address through a possibly dynamic dense root-table slot. */
static nir_def *
buffer_address_slot(struct d12_dxbc_ctx *ctx, nir_def *slot,
                    nir_def *index, nir_def *offset, unsigned stride)
{
   nir_builder *b = &ctx->b;
   nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
   nir_def *base = nir_load_global(
      b, 1, 64,
      nir_iadd(b, resource_table,
               nir_imul_imm(b, nir_u2u64(b, slot), sizeof(uint64_t))),
      .align_mul = 8);

   nir_def *byte =
      stride
         ? nir_iadd(b,
                    nir_imul_imm(
                       b, nir_u2u64(b, nir_channel(b, index, 0)), stride),
                    nir_u2u64(b, nir_channel(b, offset, 0)))
         : nir_u2u64(
              b, nir_iadd(b,
                          nir_imul_imm(b, nir_channel(b, index, 0), 4),
                          nir_channel(b, offset, 0)));

   return nir_iadd(b, base, byte);
}

/* GPU address of the d12_descriptor selected by an SM 5.1 heap-indexed UAV
 * operand, or NULL for the ordinary finite root-table model. */
static nir_def *
uav_rooted_descriptor(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *uav)
{
   nir_builder *b = &ctx->b;

   if (uav->file != D12_DXBC_FILE_UAV || uav->num_indices != 2 ||
       uav->index[0] >= D12_DXBC_MAX_SIG ||
       !ctx->uav_range[uav->index[0]].valid || !ctx->map)
      return NULL;

   const struct d12_dxbc_binding_range *range =
      &ctx->uav_range[uav->index[0]];
   unsigned base_slot = ~0u;
   unsigned rooted_reg = 0;
   if (!unbounded_root_for_register(ctx->map, D12_RES_UAV, range->space,
                                    range->lower_bound, &base_slot,
                                    &rooted_reg))
      return NULL;

   nir_def *reg = read_operand_index(ctx, uav, 1);
   if (!reg)
      return NULL;
   nir_def *descriptor_index =
      nir_iadd_imm(b, reg, -(int64_t)rooted_reg);

   nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
   nir_def *heap = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, resource_table, base_slot * sizeof(uint64_t)),
      .align_mul = 8);
   return nir_iadd(b, heap,
                   nir_imul_imm(b, nir_u2u64(b, descriptor_index),
                                D12_HEAP_DESC_SIZE));
}

static nir_def *
srv_rooted_descriptor(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *srv)
{
   nir_builder *b = &ctx->b;

   if (srv->file != D12_DXBC_FILE_RESOURCE || srv->num_indices != 2 ||
       srv->index[0] >= D12_DXBC_MAX_SIG ||
       !ctx->srv_range[srv->index[0]].valid || !ctx->map)
      return NULL;

   const struct d12_dxbc_binding_range *range =
      &ctx->srv_range[srv->index[0]];
   unsigned base_slot = ~0u;
   unsigned rooted_reg = 0;
   if (!unbounded_root_for_register(ctx->map, D12_RES_SRV, range->space,
                                    range->lower_bound, &base_slot,
                                    &rooted_reg))
      return NULL;
   nir_def *reg = read_operand_index(ctx, srv, 1);
   if (!reg)
      return NULL;

   nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
   nir_def *heap = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, resource_table, base_slot * sizeof(uint64_t)),
      .align_mul = 8);
   nir_def *descriptor_index =
      nir_iadd_imm(b, reg, -(int64_t)rooted_reg);
   return nir_iadd(b, heap,
                   nir_imul_imm(b, nir_u2u64(b, descriptor_index),
                                D12_HEAP_DESC_SIZE));
}

static nir_def *
rooted_native_index(struct d12_dxbc_ctx *ctx, nir_def *descriptor)
{
   nir_builder *b = &ctx->b;
   nir_def *heap = nir_load_descriptor_heap_agx(b);
   nir_def *offset = nir_isub(b, descriptor, heap);
   return nir_u2u32(b, nir_udiv_imm(b, offset, D12_HEAP_DESC_SIZE));
}

/* Global index selected by an SM 5.1 heap-indexed sampler operand. The root
 * sampler plane stores the descriptor heap index at the start of each
 * oversized API range; the operand supplies the register-relative offset. */
static nir_def *
sampler_rooted_index(struct d12_dxbc_ctx *ctx,
                     const struct d12_dxbc_operand *sampler)
{
   nir_builder *b = &ctx->b;
   if (sampler->file != D12_DXBC_FILE_SAMPLER ||
       sampler->num_indices != 2 ||
       sampler->index[0] >= D12_DXBC_MAX_SIG ||
       !ctx->sampler_range[sampler->index[0]].valid || !ctx->map)
      return NULL;

   const struct d12_dxbc_binding_range *range =
      &ctx->sampler_range[sampler->index[0]];
   unsigned base_slot = ~0u;
   unsigned rooted_reg = 0;
   if (!unbounded_root_for_register(ctx->map, D12_RES_SAMPLER,
                                    range->space, range->lower_bound,
                                    &base_slot, &rooted_reg))
      return NULL;

   nir_def *reg = read_operand_index(ctx, sampler, 1);
   if (!reg)
      return NULL;

   nir_def *sampler_table = root_plane(ctx, D12_ROOT_SAMPLER_BASE);
   nir_def *base = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, sampler_table, base_slot * sizeof(uint64_t)),
      .align_mul = 8);
   return nir_iadd(b, nir_u2u32(b, base),
                   nir_iadd_imm(b, reg, -(int64_t)rooted_reg));
}

/* Resolve an ordinary finite SM 5.0/5.1 texture or sampler operand. SM 5.1
 * use sites carry {range id, absolute register}; the declaration, not the
 * use-site token, owns the register space. Dynamic finite arrays need a NIR
 * handle and remain an explicit refusal here rather than aliasing range id
 * zero in space zero. */
static unsigned
finite_binding_slot(struct d12_dxbc_ctx *ctx, unsigned klass,
                    const struct d12_dxbc_operand *operand,
                    const struct d12_dxbc_binding_range *ranges)
{
   if (!ctx->map)
      return operand->num_indices == 2 ? operand->index[1]
                                       : operand->index[0];

   if (operand->num_indices == 1 && !operand->index_is_relative[0])
      return d12_binding_map_find(ctx->map, klass, 0, operand->index[0]);

   if (operand->num_indices != 2 || operand->index[0] >= D12_DXBC_MAX_SIG ||
       !ranges[operand->index[0]].valid || operand->index_is_relative[1])
      return ~0u;

   const struct d12_dxbc_binding_range *range = &ranges[operand->index[0]];
   unsigned reg = operand->index[1];
   if (reg < range->lower_bound ||
       (range->upper_bound != UINT32_MAX && reg > range->upper_bound))
      return ~0u;

   return d12_binding_map_find(ctx->map, klass, range->space, reg);
}

static nir_def *
uav_buffer_address(struct d12_dxbc_ctx *ctx,
                   const struct d12_dxbc_operand *uav, nir_def *index,
                   nir_def *offset, unsigned stride)
{
   nir_builder *b = &ctx->b;
   nir_def *descriptor = uav_rooted_descriptor(ctx, uav);
   nir_def *base;

   if (descriptor) {
      base = nir_load_global(b, 1, 64, descriptor, .align_mul = 8);
   } else {
      nir_def *slot = uav_binding_slot(ctx, uav);
      if (!slot)
         return NULL;
      nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
      base = nir_load_global(
         b, 1, 64,
         nir_iadd(b, resource_table,
                  nir_imul_imm(b, nir_u2u64(b, slot), sizeof(uint64_t))),
         .align_mul = 8);
   }

   nir_def *byte =
      stride
         ? nir_iadd(b,
                    nir_imul_imm(
                       b, nir_u2u64(b, nir_channel(b, index, 0)), stride),
                    nir_u2u64(b, nir_channel(b, offset, 0)))
         : nir_u2u64(
              b, nir_iadd(b,
                          nir_imul_imm(b, nir_channel(b, index, 0), 4),
                          nir_channel(b, offset, 0)));
   return nir_iadd(b, base, byte);
}

static nir_def *
srv_buffer_address(struct d12_dxbc_ctx *ctx,
                   const struct d12_dxbc_operand *srv, nir_def *index,
                   nir_def *offset, unsigned stride)
{
   nir_builder *b = &ctx->b;

   nir_def *descriptor = srv_rooted_descriptor(ctx, srv);
   if (descriptor) {
      nir_def *base =
         nir_load_global(b, 1, 64, descriptor, .align_mul = 8);
      nir_def *byte =
         stride
            ? nir_iadd(b,
                       nir_imul_imm(
                          b, nir_u2u64(b, nir_channel(b, index, 0)), stride),
                       nir_u2u64(b, nir_channel(b, offset, 0)))
            : nir_u2u64(
                 b, nir_iadd(b,
                             nir_imul_imm(b, nir_channel(b, index, 0), 4),
                             nir_channel(b, offset, 0)));
      return nir_iadd(b, base, byte);
   }

   unsigned slot =
      finite_binding_slot(ctx, D12_RES_SRV, srv, ctx->srv_range);
   if (slot == ~0u) {
      fail(ctx, "buffer is not bound by the root signature");
      return nir_imm_int64(b, 0);
   }

   return buffer_address_slot(ctx, nir_imm_int(b, slot), index, offset,
                              stride);
}

/*
 * All SM5 buffer atomics share the same UAV-coordinate rules. Raw buffers
 * supply a byte offset, structured buffers supply {element, member offset},
 * and typed buffers use a four-byte element index. Texture UAV atomics need
 * image descriptors and stay an explicit refusal rather than being mistaken
 * for linear memory.
 */
static nir_def *
atomic_buffer_address(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *uav, nir_def *coord)
{
   nir_builder *b = &ctx->b;

   if (uav->file != D12_DXBC_FILE_UAV ||
       uav->index[0] >= D12_DXBC_MAX_SIG) {
      fail(ctx, "atomic operand is not a declared UAV");
      return NULL;
   }

   uint32_t reg = uav->index[0];
   if (ctx->uav_dim[reg] != D12_DXBC_RESOURCE_DIMENSION_BUFFER) {
      fail(ctx, "texture UAV atomic unsupported");
      return NULL;
   }

   nir_def *index = coord;
   nir_def *offset = nir_imm_int(b, 0);
   unsigned stride = ctx->uav_stride[reg];

   if (ctx->uav_raw[reg]) {
      index = nir_imm_int(b, 0);
      offset = coord;
   } else if (stride) {
      offset = nir_channel(b, coord, 1);
   }

   return uav_buffer_address(ctx, uav, index, offset, stride);
}

/* Resolve SM 5.0 t#/u# and SM 5.1 range-indexed resource operands to the
 * dense slot assigned by the root signature. Treating an SM 5.1 range ID as
 * a register aliases every dynamic array access to its first descriptor. */
static nir_def *
resource_binding_slot(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *operand,
                      enum d12_dxbc_file file, unsigned klass,
                      const struct d12_dxbc_binding_range *ranges)
{
   nir_builder *b = &ctx->b;

   if (operand->file != file || operand->index[0] >= D12_DXBC_MAX_SIG) {
      fail(ctx, "operand is not a declared resource");
      return NULL;
   }

   nir_def *slot;
   const struct d12_dxbc_binding_range *range = NULL;

   if (operand->num_indices == 2) {
      /* SM 5.1 use-site syntax is u<range-id>[<absolute register>].
       * The second index, not the range ID, may be relative.
       */
      if (!ranges[operand->index[0]].valid) {
         fail(ctx, "SM 5.1 resource has no declared range");
         return NULL;
      }
      range = &ranges[operand->index[0]];

      unsigned first_slot =
         ctx->map ? d12_binding_map_find(ctx->map, klass, range->space,
                                         range->lower_bound)
                  : range->lower_bound;
      if (first_slot == ~0u) {
         fail(ctx, "resource range is not bound by the root signature");
         return NULL;
      }

      if (ctx->map && range->upper_bound != UINT32_MAX) {
         uint64_t count =
            (uint64_t)range->upper_bound - range->lower_bound + 1;
         unsigned last_slot = d12_binding_map_find(
            ctx->map, klass, range->space, range->upper_bound);
         unsigned root_range = d12_binding_map_root_range(
            ctx->map, klass, range->space, range->lower_bound);

         if (range->upper_bound < range->lower_bound ||
             count > D12_MAX_RESOURCE_BINDINGS ||
             last_slot != first_slot + (unsigned)count - 1 ||
             root_range == ~0u ||
             d12_binding_map_root_range(ctx->map, klass, range->space,
                                        range->upper_bound) != root_range) {
            fail(ctx, "resource array crosses root descriptor ranges");
            return NULL;
         }
      }

      nir_def *reg = read_operand_index(ctx, operand, 1);
      if (!reg)
         return NULL;
      slot = nir_iadd_imm(b, reg,
                         (int64_t)first_slot - range->lower_bound);
   } else if (operand->num_indices == 1 && !operand->index_is_relative[0]) {
      unsigned static_slot =
         ctx->map ? d12_binding_map_find(ctx->map, klass, 0, operand->index[0])
                  : operand->index[0];
      if (static_slot == ~0u) {
         fail(ctx, "resource is not bound by the root signature");
         return NULL;
      }
      slot = nir_imm_int(b, static_slot);
   } else {
      fail(ctx, "unsupported resource index shape");
      return NULL;
   }

   return slot;
}

static nir_def *
uav_binding_slot(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *uav)
{
   return resource_binding_slot(ctx, uav, D12_DXBC_FILE_UAV, D12_RES_UAV,
                                ctx->uav_range);
}

static nir_def *
srv_binding_slot(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *srv)
{
   return resource_binding_slot(ctx, srv, D12_DXBC_FILE_RESOURCE, D12_RES_SRV,
                                ctx->srv_range);
}

/* A dynamically indexed finite texture range is not bounded by the dense
 * slots which describe its root-signature declaration. The command list
 * publishes a byte offset to a declaration-shaped native segment in the
 * first dense resource slot; add the runtime register displacement there.
 * Static operands keep the compact slot ABI. */
static nir_def *
resource_texture_offset(struct d12_dxbc_ctx *ctx,
                        const struct d12_dxbc_operand *operand,
                        enum d12_dxbc_file file, unsigned klass,
                        const struct d12_dxbc_binding_range *ranges,
                        unsigned native_stride)
{
   nir_builder *b = &ctx->b;
   nir_def *slot = resource_binding_slot(ctx, operand, file, klass, ranges);
   if (!slot)
      return NULL;

   if (operand->num_indices != 2)
      return nir_imul_imm(b, slot, native_stride);

   const struct d12_dxbc_binding_range *range = &ranges[operand->index[0]];
   unsigned first_slot =
      ctx->map ? d12_binding_map_find(ctx->map, klass, range->space,
                                      range->lower_bound)
               : range->lower_bound;
   if (first_slot == ~0u) {
      fail(ctx, "dynamic texture range has no native-segment base");
      return NULL;
   }

   nir_def *resource_table = root_plane(ctx, D12_ROOT_RESOURCE_BASE);
   nir_def *base = nir_load_global(
      b, 1, 64, nir_iadd_imm(b, resource_table, first_slot * sizeof(uint64_t)),
      .align_mul = 8);
   nir_def *relative = nir_iadd_imm(b, slot, -(int64_t)first_slot);
   nir_def *index = nir_iadd(b, base, nir_u2u64(b, relative));
   if (ctx->descriptor_generation == D12_NATIVE_DESCRIPTOR_GEN7) {
      unsigned dim = file == D12_DXBC_FILE_UAV
         ? ctx->uav_dim[operand->index[0]] : ctx->srv_dim[operand->index[0]];
      const bool array = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY ||
                         dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY;
      nir_def *count = nir_udiv_imm(b, root_heap_pbe_null_offset(ctx),
                                   ctx->native_layout.pbe_stride_B);
      nir_def *count64 = nir_u2u64(b, count);
      index = nir_u2u32(b, nir_bcsel(b, nir_ult(b, index, count64),
                                   index, count64));
      index = nir_iadd(b, index,
         nir_imul_imm(b, nir_iadd_imm(b, count, 1), array ? 2 : 1));
   }
   return nir_imul_imm(b, nir_u2u32(b, index), native_stride);
}

static nir_def *
srv_texture_offset(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *srv)
{
   return resource_texture_offset(ctx, srv, D12_DXBC_FILE_RESOURCE, D12_RES_SRV,
                                  ctx->srv_range,
                                  ctx->native_layout.texture_stride_B);
}

static nir_def *
uav_texture_offset(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *uav)
{
   return resource_texture_offset(ctx, uav, D12_DXBC_FILE_UAV, D12_RES_UAV,
                                  ctx->uav_range,
                                  ctx->native_layout.texture_stride_B);
}

/*
 * Writable texture operations use the parallel PBE heap.  It is indexed by
 * the same dense binding slot as the read-only texture and buffer heaps, but
 * each entry has AGX's PBE descriptor size.  Restrict the first DXBC path to
 * the 2D dimensions the command-list uploader can describe and to finite
 * descriptor ranges; an unbounded range needs a GPU-visible descriptor heap,
 * not the command-local flattened table used here.
 */
static nir_def *
uav_image_handle(struct d12_dxbc_ctx *ctx,
                 const struct d12_dxbc_operand *uav,
                 enum glsl_sampler_dim *image_dim, bool *is_array)
{
   nir_builder *b = &ctx->b;

   if (uav->file != D12_DXBC_FILE_UAV ||
       uav->index[0] >= D12_DXBC_MAX_SIG) {
      fail(ctx, "image operand is not a declared UAV");
      return NULL;
   }

   unsigned reg = uav->index[0];
   unsigned dim = ctx->uav_dim[reg];
   *is_array = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY ||
               dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY;
   *image_dim = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_3D
                   ? GLSL_SAMPLER_DIM_3D
                : dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D ||
                      dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY
                   ? GLSL_SAMPLER_DIM_1D
                   : GLSL_SAMPLER_DIM_2D;
   if (dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D &&
       dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D && !*is_array &&
       dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_3D) {
      fail(ctx, "unsupported texture UAV dimension");
      return NULL;
   }

   nir_def *descriptor = uav_rooted_descriptor(ctx, uav);
   if (descriptor) {
      nir_def *index = rooted_native_index(ctx, descriptor);
      return nir_bindless_image_agx(
         b, nir_imul_imm(b, index, ctx->native_layout.pbe_stride_B),
         .desc_set = D12_UNIFORM_HEAP_PBES);
   }

   nir_def *offset = resource_texture_offset(
      ctx, uav, D12_DXBC_FILE_UAV, D12_RES_UAV, ctx->uav_range,
      ctx->native_layout.pbe_stride_B);
   if (!offset)
      return NULL;
   return nir_bindless_image_agx(b, offset,
      .desc_set = uav->num_indices == 2 ? D12_UNIFORM_HEAP_PBES : D12_UNIFORM_PBES);
}

static nir_def *
uav_image_coord(struct d12_dxbc_ctx *ctx, nir_def *coord,
                enum glsl_sampler_dim image_dim, bool is_array)
{
   nir_builder *b = &ctx->b;
   nir_def *zero = nir_imm_int(b, 0);

   return nir_vec4(b, nir_channel(b, coord, 0), nir_channel(b, coord, 1),
                   (is_array || image_dim == GLSL_SAMPLER_DIM_3D)
                      ? nir_channel(b, coord, 2)
                      : zero,
                   zero);
}

/*
 * AppendStructuredBuffer/ConsumeStructuredBuffer counters are separate
 * allocations.  Command recording publishes their addresses in the counter
 * root plane at the same dense binding slot as the UAV resource itself.
 */
static nir_def *
uav_counter_address(struct d12_dxbc_ctx *ctx,
                    const struct d12_dxbc_operand *uav)
{
   nir_builder *b = &ctx->b;
   nir_def *descriptor = uav_rooted_descriptor(ctx, uav);
   if (descriptor) {
      return nir_load_global(
         b, 1, 64, nir_iadd_imm(b, descriptor, D12_DESC_COUNTER_OFFSET),
         .align_mul = 8);
   }

   nir_def *slot = uav_binding_slot(ctx, uav);
   if (!slot)
      return NULL;

   nir_def *counter_table = root_plane(ctx, D12_ROOT_COUNTER_BASE);
   return nir_load_global(
      b, 1, 64,
      nir_iadd(b, counter_table,
               nir_imul_imm(b, nir_u2u64(b, slot), sizeof(uint64_t))),
      .align_mul = 8);
}

/* The AGX buffer-image atomic path does not suppress an out-of-range
 * coordinate itself.  For an SM 5.1 unbounded typed UAV, select either its
 * live PBE descriptor or the extra null descriptor at the end of the bound
 * heap.  The selection must stay inside the bindless-image intrinsic so its
 * producer retains the shape required by the AGX backend. */
static nir_def *
uav_rooted_typed_atomic_handle(struct d12_dxbc_ctx *ctx,
                               nir_def *descriptor, nir_def *element)
{
   nir_builder *b = &ctx->b;
   nir_def *count = nir_load_global(
      b, 1, 32,
      nir_iadd_imm(b, descriptor, D12_DESC_BUFFER_COUNT_OFFSET),
      .align_mul = 4);
   nir_def *valid = nir_ult(
      b, nir_u2u32(b, nir_channel(b, element, 0)), count);
   nir_def *live_offset = nir_imul_imm(
      b, rooted_native_index(ctx, descriptor),
      ctx->native_layout.pbe_stride_B);

   return nir_bindless_image_agx(
      b, nir_bcsel(b, valid, live_offset,
                   root_heap_pbe_null_offset(ctx)),
      .desc_set = D12_UNIFORM_HEAP_PBES);
}

/*
 * Return whether a structured element index lies inside the logical buffer
 * view.  The BUFINFO plane carries NumElements in its low word; checking the
 * backing allocation would incorrectly permit accesses beyond a smaller UAV
 * or SRV view.
 */
static nir_def *
buffer_element_in_bounds_slot(struct d12_dxbc_ctx *ctx, nir_def *slot,
                              nir_def *index)
{
   nir_builder *b = &ctx->b;
   nir_def *bufinfo_table = root_plane(ctx, D12_ROOT_BUFINFO_BASE);
   nir_def *count = nir_load_global(
      b, 1, 32,
      nir_iadd(b, bufinfo_table,
               nir_imul_imm(b, nir_u2u64(b, slot), sizeof(uint64_t))),
      .align_mul = 8);

   return nir_ult(b, nir_u2u32(b, nir_channel(b, index, 0)), count);
}

/* Heap descriptors and finite root tables expose {NumElements, stride} for
 * actual buffer views. A zero stride is the bare-root sentinel: in that case
 * the low word is already a byte extent. */
static void
buffer_view_info(struct d12_dxbc_ctx *ctx,
                 const struct d12_dxbc_operand *resource,
                 nir_def **count_or_bytes, nir_def **stride_B)
{
   nir_builder *b = &ctx->b;
   nir_def *descriptor = resource->file == D12_DXBC_FILE_UAV
                            ? uav_rooted_descriptor(ctx, resource)
                            : srv_rooted_descriptor(ctx, resource);
   if (descriptor) {
      nir_def *info = nir_load_global(
         b, 2, 32,
         nir_iadd_imm(b, descriptor, D12_DESC_BUFFER_COUNT_OFFSET),
         .align_mul = 8);
      *count_or_bytes = nir_channel(b, info, 0);
      *stride_B = nir_channel(b, info, 1);
      return;
   }

   nir_def *slot = NULL;
   if (resource->file == D12_DXBC_FILE_UAV) {
      slot = uav_binding_slot(ctx, resource);
   } else if (resource->file == D12_DXBC_FILE_RESOURCE) {
      unsigned static_slot = finite_binding_slot(
         ctx, D12_RES_SRV, resource, ctx->srv_range);
      if (static_slot != ~0u)
         slot = nir_imm_int(b, static_slot);
   }

   if (!slot) {
      fail(ctx, "buffer view metadata has no resolvable binding slot");
      *count_or_bytes = nir_imm_int(b, 0);
      *stride_B = nir_imm_int(b, 0);
      return;
   }

   nir_def *bufinfo = root_plane(ctx, D12_ROOT_BUFINFO_BASE);
   nir_def *info = nir_load_global(
      b, 2, 32,
      nir_iadd(b, bufinfo,
               nir_imul_imm(b, nir_u2u64(b, slot), sizeof(uint64_t))),
      .align_mul = 8);
   *count_or_bytes = nir_channel(b, info, 0);
   *stride_B = nir_iand_imm(b, nir_channel(b, info, 1), 0xffff);
}

static nir_def *
buffer_view_count_or_bytes(struct d12_dxbc_ctx *ctx,
                           const struct d12_dxbc_operand *resource)
{
   nir_def *count_or_bytes, *stride_B;
   buffer_view_info(ctx, resource, &count_or_bytes, &stride_B);
   return count_or_bytes;
}

static nir_def *
buffer_view_size_bytes(struct d12_dxbc_ctx *ctx,
                       const struct d12_dxbc_operand *resource)
{
   nir_builder *b = &ctx->b;
   nir_def *count_or_bytes, *stride_B;
   buffer_view_info(ctx, resource, &count_or_bytes, &stride_B);
   nir_def *count64 = nir_u2u64(b, count_or_bytes);
   return nir_bcsel(b, nir_ine_imm(b, stride_B, 0),
                    nir_imul(b, count64, nir_u2u64(b, stride_B)), count64);
}

static nir_def *
robust_load_buffer_vec4(struct d12_dxbc_ctx *ctx, nir_def *address,
                        nir_def *byte_offset, nir_def *size_bytes)
{
   nir_builder *b = &ctx->b;
   nir_def *values[4];
   nir_def *offset64 = nir_u2u64(b, nir_channel(b, byte_offset, 0));
   nir_def *size64 = nir_u2u64(b, size_bytes);

   for (unsigned c = 0; c < 4; c++) {
      nir_def *component_offset = nir_iadd_imm(b, offset64, c * 4);
      nir_push_if(b, nir_ult(b, component_offset, size64));
      nir_def *in_bounds = nir_load_global(
         b, 1, 32, nir_iadd_imm(b, address, c * 4), .align_mul = 4);
      nir_push_else(b, NULL);
      nir_def *out_of_bounds = nir_imm_int(b, 0);
      nir_pop_if(b, NULL);
      values[c] = nir_if_phi(b, in_bounds, out_of_bounds);
   }

   return nir_vec(b, values, ARRAY_SIZE(values));
}

/* The metadata header and the prebiased coherent map use the same ABI as
 * DXIL. Its zero page is entry 2; the two residency arrays and read-only
 * sparse VA offset follow the four original header entries. */
static nir_def *
buffer_metadata_pointer(struct d12_dxbc_ctx *ctx, unsigned offset_B)
{
   nir_builder *b = &ctx->b;
   nir_def *header = nir_load_global(b, 3, 64, nir_load_root_agx(b),
                                    .align_mul = 8);
   return nir_load_global(b, 1, 64,
                         nir_iadd_imm(b, nir_channel(b, header, 1), offset_B),
                         .align_mul = 8);
}

static nir_def *
buffer_feedback_code(struct d12_dxbc_ctx *ctx,
                      const struct d12_dxbc_operand *resource,
                      nir_def *byte, nir_def *access_size_B,
                      nir_def **read_address)
{
   nir_builder *b = &ctx->b;
   bool uav = resource->file == D12_DXBC_FILE_UAV;
   nir_def *descriptor = uav ? uav_rooted_descriptor(ctx, resource)
                             : srv_rooted_descriptor(ctx, resource);
   nir_def *index = descriptor ? rooted_native_index(ctx, descriptor)
                    : uav ? uav_binding_slot(ctx, resource)
                          : srv_binding_slot(ctx, resource);
   if (!index)
      return NULL;
   nir_def *zero_page = buffer_metadata_pointer(ctx, 16);
   nir_def *table = buffer_metadata_pointer(ctx, descriptor ? 40 : 32);
   nir_def *has_table = nir_ine_imm(b, table, 0);
   nir_def *safe_table = nir_bcsel(b, has_table, table, zero_page);
   nir_def *safe_index = nir_bcsel(b, has_table, index, nir_imm_int(b, 0));
   nir_def *map = nir_load_global(
      b, 1, 64, nir_iadd(b, safe_table,
                         nir_imul_imm(b, nir_u2u64(b, safe_index), 8)),
      .align_mul = 8);
   map = nir_bcsel(b, has_table, map, nir_imm_int64(b, 0));
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *base = uav ? uav_buffer_address(ctx, resource, zero, zero, 0)
                       : srv_buffer_address(ctx, resource, zero, zero, 0);
   nir_def *size = buffer_view_size_bytes(ctx, resource);
   if (ctx->error || !base)
      return NULL;
   nir_def *status = d12_nir_buffer_residency_status(
      b, map, base, byte, size, access_size_B, NULL, zero_page);
   if (read_address) {
      nir_def *ro_offset = buffer_metadata_pointer(ctx, 48);
      *read_address = nir_bcsel(b, nir_ine_imm(b, map, 0),
                                nir_ior(b, *read_address, ro_offset),
                                *read_address);
   }
   /* Match the existing AGX opaque status convention (resident bit 0 is
    * clear), so texture and buffer codes share CHECK_ACCESS_FULLY_MAPPED. */
   return nir_iand_imm(b, nir_inot(b, status), 1);
}

static nir_def *
buffer_vector_feedback(struct d12_dxbc_ctx *ctx,
                        const struct d12_dxbc_operand *resource,
                        const struct d12_dxbc_operand *dst, nir_def *byte,
                        nir_def **read_address)
{
   if (dst->file == D12_DXBC_FILE_NULL || !dst->mask) {
      fail(ctx, "buffer feedback needs a data destination component mask");
      return NULL;
   }
   unsigned first = 4, last = 0;
   for (unsigned c = 0; c < 4; c++) {
      if (dst->mask & (1u << c)) {
         first = MIN2(first, resource->swizzle[c]);
         last = MAX2(last, resource->swizzle[c]);
      }
   }
   nir_builder *b = &ctx->b;
   return buffer_feedback_code(ctx, resource,
      nir_iadd_imm(b, nir_u2u64(b, nir_channel(b, byte, 0)), first * 4),
      nir_imm_int(b, (last - first + 1) * 4), read_address);
}

static void
robust_store_buffer_masked(struct d12_dxbc_ctx *ctx, nir_def *address,
                           nir_def *byte_offset, nir_def *size_bytes,
                           nir_def *value, unsigned mask)
{
   nir_builder *b = &ctx->b;
   nir_def *offset64 = nir_u2u64(b, nir_channel(b, byte_offset, 0));
   nir_def *size64 = nir_u2u64(b, size_bytes);

   for (unsigned c = 0; c < 4; c++) {
      if (!(mask & (1 << c)))
         continue;

      nir_def *component_offset = nir_iadd_imm(b, offset64, c * 4);
      nir_push_if(b, nir_ult(b, component_offset, size64));
      nir_store_global(b, nir_channel(b, value, c),
                       nir_iadd_imm(b, address, c * 4), .write_mask = 1,
                       .align_mul = 4);
      nir_pop_if(b, NULL);
   }
}

static void
store_patch_output_component(struct d12_dxbc_ctx *ctx, nir_def *value,
                             unsigned source_component,
                             const struct d12_dxbc_sig_elem *e)
{
   nir_builder *b = &ctx->b;
   unsigned loc = patch_location(e, e->reg);
   unsigned component = loc == VARYING_SLOT_TESS_LEVEL_OUTER ||
                              loc == VARYING_SLOT_TESS_LEVEL_INNER
                           ? e->semantic_index
                           : source_component;

   unsigned outer, inner;
   mesa_count_tess_level_components(b->shader->info.tess._primitive_mode,
                                    &outer, &inner);
   unsigned limit = loc == VARYING_SLOT_TESS_LEVEL_OUTER
                       ? outer
                       : loc == VARYING_SLOT_TESS_LEVEL_INNER ? inner : 4;
   if (component >= limit) {
      fail(ctx, "tessellation factor component exceeds domain");
      return;
   }

   nir_store_output(b, nir_channel(b, value, source_component),
                    nir_imm_int(b, 0), .base = e->reg, .write_mask = 1,
                    .component = component, .src_type = nir_type_float32,
                    .io_semantics.location = loc,
                    .io_semantics.num_slots = 1);
   if (loc >= VARYING_SLOT_PATCH0)
      b->shader->info.patch_outputs_written |=
         BITFIELD64_BIT(loc - VARYING_SLOT_PATCH0);
   else
      b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
}

static void
write_dst(struct d12_dxbc_ctx *ctx, const struct d12_dxbc_operand *op,
          nir_def *value)
{
   nir_builder *b = &ctx->b;
   unsigned mask = op->mask ? op->mask : 0xf;

   switch (op->file) {
   case D12_DXBC_FILE_TEMP: {
      nir_variable *var = temp_var(ctx, op->index[0]);
      if (var)
         nir_store_var(b, var, value, mask);
      break;
   }

   case D12_DXBC_FILE_INDEXABLE_TEMP: {
      if (op->num_indices != 2 || op->index_is_relative[0] ||
          op->index[0] >= D12_DXBC_MAX_SIG ||
          !ctx->indexable_temps[op->index[0]]) {
         fail(ctx, "invalid indexable temporary destination");
         break;
      }

      nir_def *index = read_operand_index(ctx, op, 1);
      if (!index)
         break;

      unsigned file = op->index[0];
      nir_def **shadow = ctx->indexable_temp_shadow[file];
      if (ctx->cf_depth || !shadow ||
          !nir_src_is_const(nir_src_for_ssa(index))) {
         ctx->indexable_temp_shadow_valid[file] = false;
      } else {
         unsigned element = nir_src_as_uint(nir_src_for_ssa(index));
         if (element >= ctx->indexable_temp_size[file]) {
            ctx->indexable_temp_shadow_valid[file] = false;
         } else {
            nir_def *old = shadow[element];
            if (!old)
               old = nir_undef(b, 4, 32);

            nir_def *components[4];
            for (unsigned c = 0; c < 4; c++)
               components[c] = nir_channel(b, mask & BITFIELD_BIT(c) ? value
                                                                     : old,
                                           c);
            shadow[element] = nir_vec(b, components, 4);
         }
      }

      nir_deref_instr *array =
         nir_build_deref_var(b, ctx->indexable_temps[op->index[0]]);
      nir_store_deref(b, nir_build_deref_array(b, array, index), value, mask);
      break;
   }

   case D12_DXBC_FILE_OUTPUT: {
      uint32_t reg = op->index[op->num_indices ? op->num_indices - 1 : 0];
      if (b->shader->info.stage == MESA_SHADER_TESS_CTRL &&
          (ctx->hs_phase == D12_DXBC_HS_FORK ||
           ctx->hs_phase == D12_DXBC_HS_JOIN)) {
         if (op->num_indices != 1) {
            fail(ctx, "invalid hull patch-output operand");
            break;
         }

         if (op->index_is_relative[0]) {
            nir_def *dynamic_reg = read_operand_index(ctx, op, 0);
            if (!dynamic_reg)
               break;

            for (unsigned i = 0; i < ctx->num_patch_sig; i++) {
               const struct d12_dxbc_sig_elem *e = &ctx->patch_sig[i];
               unsigned component_mask = mask & e->mask;
               if (!component_mask)
                  continue;

               nir_push_if(b, nir_ieq_imm(b, dynamic_reg, e->reg));
               u_foreach_bit(c, component_mask)
                  store_patch_output_component(ctx, value, c, e);
               nir_pop_if(b, NULL);
            }
         } else {
            for (unsigned c = 0; c < 4; c++) {
               if (!(mask & BITFIELD_BIT(c)))
                  continue;
               const struct d12_dxbc_sig_elem *e = find_sig_component(
                  ctx->patch_sig, ctx->num_patch_sig, reg, c);
               if (!e) {
                  fail(ctx, "hull patch output has no signature element");
                  break;
               }
               store_patch_output_component(ctx, value, c, e);
            }
         }
      } else {
         unsigned packed_clip_cull_mask = 0;
         if (b->shader->info.stage != MESA_SHADER_TESS_CTRL) {
            const unsigned clip_count = system_value_component_count(
               ctx->out_sig, ctx->num_out_sig, "SV_ClipDistance");
            const unsigned cull_count = system_value_component_count(
               ctx->out_sig, ctx->num_out_sig, "SV_CullDistance");

            u_foreach_bit(c, mask) {
               const struct d12_dxbc_sig_elem *ce = find_sig_component(
                  ctx->out_sig, ctx->num_out_sig, reg, c);
               const bool is_clip =
                  ce && !strcasecmp(ce->name, "SV_ClipDistance");
               const bool is_cull =
                  ce && !strcasecmp(ce->name, "SV_CullDistance");
               if (!is_clip && !is_cull)
                  continue;

               packed_clip_cull_mask |= BITFIELD_BIT(c);

               unsigned compact = system_value_compact_component(
                  ctx->out_sig, ctx->num_out_sig, ce, c);
               unsigned combined = compact + (is_cull ? clip_count : 0);
               unsigned loc = VARYING_SLOT_CLIP_DIST0 + (combined / 4);

               nir_store_output(
                  b, nir_channel(b, value, c), nir_imm_int(b, 0),
                  .base = 0, .write_mask = 1, .component = combined % 4,
                  .src_type = nir_type_float32,
                  .io_semantics.location = loc,
                  .io_semantics.num_slots = DIV_ROUND_UP(
                     clip_count + cull_count, 4));
               b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
            }

            if (packed_clip_cull_mask) {
               b->shader->info.clip_distance_array_size = clip_count;
               b->shader->info.cull_distance_array_size = cull_count;

               /* Signature packing may place an ordinary output beside a
                * clip/cull component in the same physical register. Preserve
                * those components with scalar stores to their own varying
                * locations rather than dropping the whole remainder. */
               u_foreach_bit(c, mask & ~packed_clip_cull_mask) {
                  const struct d12_dxbc_sig_elem *ce = find_sig_component(
                     ctx->out_sig, ctx->num_out_sig, reg, c);
                  unsigned loc = output_location(ctx, ce, reg);

                  nir_store_output(
                     b, nir_channel(b, value, c), nir_imm_int(b, 0),
                     .base = reg, .write_mask = 1, .component = c,
                     .src_type = nir_type_float32,
                     .io_semantics.location = loc,
                     .io_semantics.num_slots = 1);
                  b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
               }
            }
         }

         if (packed_clip_cull_mask)
            break;

         const struct d12_dxbc_sig_elem *e =
            find_sig(ctx->out_sig, ctx->num_out_sig, reg);
         unsigned loc = output_location(ctx, e, reg);

         if (b->shader->info.stage == MESA_SHADER_TESS_CTRL) {
            nir_store_per_vertex_output(
               b, value, nir_load_invocation_id(b), nir_imm_int(b, 0),
               .base = reg, .write_mask = mask,
               .src_type = nir_type_float32,
               .io_semantics.location = loc,
               .io_semantics.num_slots = 1);
         } else {
            nir_store_output(b, value, nir_imm_int(b, 0), .base = reg,
                             .write_mask = mask,
                             .src_type = nir_type_float32,
                             .io_semantics.location = loc,
                             .io_semantics.num_slots = 1);
         }
         b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
      }
      break;
   }

   case D12_DXBC_FILE_OUTPUT_DEPTH:
      nir_store_output(b, nir_channel(b, value, 0), nir_imm_int(b, 0),
                       .base = 0, .write_mask = 1,
                       .src_type = nir_type_float32,
                       .io_semantics.location = FRAG_RESULT_DEPTH,
                       .io_semantics.num_slots = 1);
      b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
      break;

   case D12_DXBC_FILE_OUTPUT_COVERAGE_MASK:
      nir_store_output(b, nir_channel(b, value, 0), nir_imm_int(b, 0),
                       .base = 0, .write_mask = 1, .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_SAMPLE_MASK,
                       .io_semantics.num_slots = 1);
      b->shader->info.outputs_written |=
         BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      break;

   case D12_DXBC_FILE_NULL:
      break;

   default:
      snprintf(ctx->errbuf, sizeof(ctx->errbuf),
               "unsupported destination register file %u", op->file);
      fail(ctx, ctx->errbuf);
      break;
   }
}

static nir_def *
apply_dst_saturate(nir_builder *b, uint32_t op_token, nir_def *value)
{
   /* Public token bit 13 is the instruction's _sat result modifier. */
   return (op_token & BITFIELD_BIT(13)) ? nir_fsat(b, value) : value;
}

/*
 * Locate the shader chunk. SHEX is Shader Model 5, SHDR is Shader Model
 * 4; the token stream is the same shape, so either is accepted.
 */
bool
d12_dxbc_find_shader(const void *data, size_t size_B, const uint8_t **out,
                     size_t *out_size_B, uint32_t *out_version)
{
   const uint8_t *p = data;

   if (!p || size_B < 32)
      return false;

   uint32_t magic;
   memcpy(&magic, p, 4);
   if (magic != 0x43425844u) /* 'DXBC' */
      return false;

   uint32_t part_count;
   memcpy(&part_count, p + 28, 4);
   if (part_count > 64 || 32 + (size_t)part_count * 4 > size_B)
      return false;

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t off;
      memcpy(&off, p + 32 + i * 4, 4);
      if ((size_t)off + 8 > size_B)
         return false;

      if (memcmp(p + off, "SHEX", 4) && memcmp(p + off, "SHDR", 4))
         continue;

      uint32_t sz;
      memcpy(&sz, p + off + 4, 4);
      if ((size_t)off + 8 + sz > size_B || sz < 8)
         return false;

      if (out_version)
         memcpy(out_version, p + off + 8, 4);

      *out = p + off + 8;
      *out_size_B = sz;
      return true;
   }

   return false;
}

/* Locate any chunk by fourcc. */
static bool
find_chunk(const void *data, size_t size_B, const char *a, const char *b,
           const uint8_t **out, size_t *out_size_B)
{
   const uint8_t *p = data;

   if (!p || size_B < 32)
      return false;

   uint32_t part_count;
   memcpy(&part_count, p + 28, 4);
   if (part_count > 64 || 32 + (size_t)part_count * 4 > size_B)
      return false;

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t off;
      memcpy(&off, p + 32 + i * 4, 4);
      if ((size_t)off + 8 > size_B)
         return false;

      if (memcmp(p + off, a, 4) && (!b || memcmp(p + off, b, 4)))
         continue;

      uint32_t sz;
      memcpy(&sz, p + off + 4, 4);
      if ((size_t)off + 8 + sz > size_B)
         return false;

      *out = p + off + 8;
      *out_size_B = sz;
      return true;
   }

   return false;
}

/*
 * INPUT SIGNATURE of a DXBC shader, in the same shape the DXIL path
 * reports. resolve_input_layout needs it to match D3D12 input-layout
 * elements to shader registers; without it nothing resolves, no vertex
 * attributes are fetched, and load_input survives to the AGX backend as
 * "Unhandled intrinsic". The DXIL version reads the LLVM bitcode module,
 * which a DXBC container does not have -- this reads the ISGN chunk the
 * translator already parses for its own use.
 */
unsigned
d12_dxbc_input_signature(const void *data, size_t size_B,
                         struct d12_input_sig_elem *out, unsigned max)
{
   const uint8_t *sig;
   size_t sig_size;
   bool extended = find_chunk(data, size_B, "ISG1", NULL, &sig, &sig_size);

   if (!extended &&
       !find_chunk(data, size_B, "ISGN", NULL, &sig, &sig_size))
      return 0;

   struct d12_dxbc_sig_elem elems[D12_DXBC_MAX_SIG];
   unsigned n = parse_signature(sig, sig_size, elems, D12_DXBC_MAX_SIG,
                                extended, extended);
   n = MIN2(n, max);

   for (unsigned i = 0; i < n; i++) {
      /* `reg` is the register the shader reads with load_input, which is
       * what the DXIL path calls `id`. */
      out[i].id = elems[i].reg;
      out[i].register_index = elems[i].reg;
      out[i].semantic_index = elems[i].semantic_index;
      out[i].start_component = elems[i].mask ? ffs(elems[i].mask) - 1 : 0;
      out[i].component_count = util_bitcount(elems[i].mask);
      out[i].rows = 1;
      out[i].component_type = elems[i].component_type;
      out[i].min_precision = elems[i].min_precision;
      out[i].system_value = elems[i].sysval;
      strncpy(out[i].name, elems[i].name, sizeof(out[i].name) - 1);
      out[i].name[sizeof(out[i].name) - 1] = '\0';
   }

   return n;
}

unsigned
d12_dxbc_output_signature(const void *data, size_t size_B,
                          struct d12_input_sig_elem *out, unsigned max)
{
   const uint8_t *sig;
   size_t sig_size;
   bool min_precision =
      find_chunk(data, size_B, "OSG1", NULL, &sig, &sig_size);
   bool stream = min_precision;

   if (!stream) {
      stream = find_chunk(data, size_B, "OSG5", NULL, &sig, &sig_size);
      if (!stream &&
          !find_chunk(data, size_B, "OSGN", NULL, &sig, &sig_size))
         return 0;
   }

   struct d12_dxbc_sig_elem elems[D12_DXBC_MAX_SIG];
   unsigned n = parse_signature(sig, sig_size, elems, D12_DXBC_MAX_SIG,
                                stream, min_precision);
   n = MIN2(n, max);

   for (unsigned i = 0; i < n; i++) {
      out[i].id = elems[i].reg;
      out[i].register_index = elems[i].reg;
      out[i].semantic_index = elems[i].semantic_index;
      out[i].start_component = elems[i].mask ? ffs(elems[i].mask) - 1 : 0;
      out[i].component_count = util_bitcount(elems[i].mask);
      out[i].rows = 1;
      out[i].component_type = elems[i].component_type;
      out[i].min_precision = elems[i].min_precision;
      out[i].system_value = elems[i].sysval;
      strncpy(out[i].name, elems[i].name, sizeof(out[i].name) - 1);
      out[i].name[sizeof(out[i].name) - 1] = '\0';
   }

   return n;
}

unsigned
d12_dxbc_patch_signature(const void *data, size_t size_B,
                         struct d12_input_sig_elem *out, unsigned max)
{
   const uint8_t *sig;
   size_t sig_size;
   bool extended = find_chunk(data, size_B, "PSG1", NULL, &sig, &sig_size);

   if (!extended &&
       !find_chunk(data, size_B, "PCSG", NULL, &sig, &sig_size))
      return 0;

   struct d12_dxbc_sig_elem elems[D12_DXBC_MAX_SIG];
   unsigned n = parse_signature(sig, sig_size, elems, D12_DXBC_MAX_SIG,
                                extended, extended);
   n = MIN2(n, max);

   for (unsigned i = 0; i < n; i++) {
      out[i].id = elems[i].reg;
      out[i].register_index = elems[i].reg;
      out[i].semantic_index = elems[i].semantic_index;
      out[i].start_component = elems[i].mask ? ffs(elems[i].mask) - 1 : 0;
      out[i].component_count = util_bitcount(elems[i].mask);
      out[i].rows = 1;
      out[i].component_type = elems[i].component_type;
      out[i].min_precision = elems[i].min_precision;
      out[i].system_value = elems[i].sysval;
      strncpy(out[i].name, elems[i].name, sizeof(out[i].name) - 1);
      out[i].name[sizeof(out[i].name) - 1] = '\0';
   }

   return n;
}

bool
d12_dxbc_is_dxbc_shader(const void *data, size_t size_B)
{
   const uint8_t *chunk;
   size_t chunk_size;
   return d12_dxbc_find_shader(data, size_B, &chunk, &chunk_size, NULL);
}

bool
d12_dxbc_container_stage(const void *data, size_t size_B,
                         mesa_shader_stage *stage)
{
   const uint8_t *chunk;
   size_t chunk_size;
   uint32_t version;

   if (!stage ||
       !d12_dxbc_find_shader(data, size_B, &chunk, &chunk_size, &version))
      return false;

   /* Microsoft D3D12TokenizedProgramFormat.hpp: the program type is bits
    * 31:16 of the SHEX/SHDR version token. Keep the numeric ABI local so
    * this C source does not depend on the Windows C++ header.
    */
   switch (version >> 16) {
   case 0:  *stage = MESA_SHADER_FRAGMENT;  return true;
   case 1:  *stage = MESA_SHADER_VERTEX;    return true;
   case 2:  *stage = MESA_SHADER_GEOMETRY;  return true;
   case 3:  *stage = MESA_SHADER_TESS_CTRL; return true;
   case 4:  *stage = MESA_SHADER_TESS_EVAL; return true;
   case 5:  *stage = MESA_SHADER_COMPUTE;   return true;
   case 13: *stage = MESA_SHADER_MESH;      return true;
   case 14: *stage = MESA_SHADER_TASK;      return true;
   default: return false;
   }
}

/* A DXBC switch may place DEFAULT before later CASE labels.  Its predicate
 * is therefore "no case in this switch matches", not merely "no earlier
 * case matched".  Scan the structured token range once and build that
 * predicate before opening the synthetic one-iteration NIR loop used for
 * break/fallthrough semantics.  Nested switches belong to their own scan. */
static nir_def *
d12_dxbc_switch_any_match(struct d12_dxbc_ctx *ctx, const uint32_t *tok,
                          nir_def *value)
{
   nir_builder *b = &ctx->b;
   nir_def *any = nir_imm_false(b);
   unsigned nested = 0;

   while (tok < ctx->end) {
      uint32_t op_tok = *tok;
      unsigned opcode = op_tok & 0x7ff;
      unsigned len = (op_tok >> 24) & 0x7f;
      unsigned opcode_words = 1;

      if (opcode == D12_DXBC_OP_CUSTOMDATA) {
         if (tok + 2 > ctx->end || tok[1] < 2 || tok + tok[1] > ctx->end) {
            fail(ctx, "malformed custom data in switch");
            return NULL;
         }
         tok += tok[1];
         continue;
      }

      if (op_tok & 0x80000000u) {
         while ((tok[opcode_words - 1] & 0x80000000u) &&
                tok + opcode_words < ctx->end)
            opcode_words++;
         if (!len)
            len = opcode_words;
      }

      if (!len || opcode_words > len || tok + len > ctx->end) {
         fail(ctx, "malformed instruction in switch");
         return NULL;
      }

      if (opcode == D12_DXBC_OP_SWITCH) {
         nested++;
      } else if (opcode == D12_DXBC_OP_ENDSWITCH) {
         if (!nested)
            return any;
         nested--;
      } else if (!nested && opcode == D12_DXBC_OP_CASE) {
         struct d12_dxbc_operand operand;
         unsigned used = decode_operand(ctx, tok + opcode_words,
                                        len - opcode_words, &operand);
         if (!used || operand.file != D12_DXBC_FILE_IMMEDIATE32) {
            fail(ctx, "switch case is not an immediate constant");
            return NULL;
         }
         any = nir_ior(b, any, nir_ieq_imm(b, value, operand.imm[0]));
      }

      tok += len;
   }

   fail(ctx, "unterminated switch");
   return NULL;
}

/*
 * Walk the token stream. Every instruction carries its own length, so an
 * opcode this front end does not implement can still be STEPPED OVER --
 * but it is not, deliberately: silently skipping an instruction produces
 * a shader that runs and computes the wrong answer, which is the single
 * worst outcome. Unknown opcodes fail the whole translation.
 */
static void
translate_tokens(struct d12_dxbc_ctx *ctx)
{
   nir_builder *b = &ctx->b;

   while (ctx->tok < ctx->end && !ctx->error) {
      uint32_t op_tok = *ctx->tok;
      unsigned opcode = op_tok & 0x7ff;
      unsigned len = (op_tok >> 24) & 0x7f;
      unsigned opcode_words = 1;

      /* Bit 31 starts a chain of extended OPCODE words. They are part of
       * the instruction length but are not operands. Decoding the chain as
       * operands made the last extension word a fake destination (the LD in
       * test_uninit_root_parameters consequently wrote only .z).
       */
      if (op_tok & 0x80000000u) {
         const uint32_t *q = ctx->tok;
         unsigned n = 1;
         while ((q[n - 1] & 0x80000000u) && ctx->tok + n < ctx->end)
            n++;
         opcode_words = n;
         if (!len)
            len = n;
      }

      /* CUSTOMDATA does NOT carry its length in the opcode token: the
       * count is the token after it. Treating it like an ordinary
       * instruction desynchronises the whole rest of the stream, which
       * then decodes as plausible garbage. */
      if (opcode == D12_DXBC_OP_CUSTOMDATA) {
         if (ctx->tok + 2 > ctx->end) {
            fail(ctx, "truncated custom data block");
            return;
         }
         unsigned n = ctx->tok[1];
         if (n < 2 || ctx->tok + n > ctx->end) {
            fail(ctx, "bad custom data length");
            return;
         }
         unsigned custom_class = op_tok >> 11;
         if (custom_class == 3) {
            if ((n - 2) % 4) {
               fail(ctx, "misaligned immediate constant buffer");
               return;
            }
            ctx->immediate_constants = ctx->tok + 2;
            ctx->num_immediate_constants = (n - 2) / 4;
         }
         ctx->tok += n;
         continue;
      }

      if (!len || ctx->tok + len > ctx->end) {
         fail(ctx, "malformed instruction length");
         return;
      }

      if (opcode_words > len) {
         fail(ctx, "extended opcode chain exceeds instruction length");
         return;
      }

      const uint32_t *args = ctx->tok + opcode_words;
      unsigned avail = len - opcode_words;

      struct d12_dxbc_operand o[D12_DXBC_MAX_OPERANDS];
      unsigned nops = 0;

      /* Declarations carry their payload inline rather than as operands,
       * and are handled before any operand decoding. */
      switch (opcode) {
      case D12_DXBC_OP_HS_DECLS:
         ctx->hs_phase = D12_DXBC_HS_DECLS;
         ctx->tok += len;
         continue;
      case D12_DXBC_OP_HS_CONTROL_POINT_PHASE:
         if (ctx->hs_instance_phase_open) {
            nir_pop_if(b, NULL);
            ctx->hs_instance_phase_open = false;
         }
         ctx->hs_phase = D12_DXBC_HS_CONTROL_POINT;
         ctx->hs_has_control_point_phase = true;
         nir_push_if(b, nir_ult_imm(
                           b, nir_load_invocation_id(b),
                           b->shader->info.tess.tcs_vertices_out));
         ctx->hs_instance_phase_open = true;
         ctx->tok += len;
         continue;
      case D12_DXBC_OP_HS_FORK_PHASE:
         if (ctx->hs_instance_phase_open) {
            nir_pop_if(b, NULL);
            ctx->hs_instance_phase_open = false;
         }
         ctx->hs_phase = D12_DXBC_HS_FORK;
         ctx->tok += len;
         continue;
      case D12_DXBC_OP_HS_JOIN_PHASE:
         if (ctx->hs_instance_phase_open) {
            nir_pop_if(b, NULL);
            ctx->hs_instance_phase_open = false;
         }
         ctx->hs_phase = D12_DXBC_HS_JOIN;
         ctx->tok += len;
         continue;

      case D12_DXBC_OP_DCL_OUTPUT_CONTROL_POINT_COUNT: {
         unsigned count = (op_tok >> 11) & 0x3f;
         if (!count || count > 32) {
            fail(ctx, "invalid hull output control-point count");
            return;
         }
         b->shader->info.tess.tcs_vertices_out = count;
         ctx->hs_max_invocations = MAX2(ctx->hs_max_invocations, count);
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_INPUT_CONTROL_POINT_COUNT: {
         unsigned count = (op_tok >> 11) & 0x3f;
         if (!count || count > 32) {
            fail(ctx, "invalid hull input control-point count");
            return;
         }
         ctx->hs_input_control_points = count;
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_TESS_DOMAIN: {
         switch ((op_tok >> 11) & 0x3) {
         case 1: b->shader->info.tess._primitive_mode =
                    TESS_PRIMITIVE_ISOLINES; break;
         case 2: b->shader->info.tess._primitive_mode =
                    TESS_PRIMITIVE_TRIANGLES; break;
         case 3: b->shader->info.tess._primitive_mode =
                    TESS_PRIMITIVE_QUADS; break;
         default:
            fail(ctx, "invalid tessellator domain");
            return;
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_TESS_PARTITIONING: {
         switch ((op_tok >> 11) & 0x7) {
         case 1: b->shader->info.tess.spacing = TESS_SPACING_EQUAL; break;
         case 3: b->shader->info.tess.spacing =
                    TESS_SPACING_FRACTIONAL_ODD; break;
         case 4: b->shader->info.tess.spacing =
                    TESS_SPACING_FRACTIONAL_EVEN; break;
         default:
            fail(ctx, "unsupported tessellator partitioning");
            return;
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_TESS_OUTPUT_PRIMITIVE: {
         unsigned primitive = (op_tok >> 11) & 0x7;
         if (primitive < 1 || primitive > 4) {
            fail(ctx, "invalid tessellator output primitive");
            return;
         }
         b->shader->info.tess.point_mode = primitive == 1;
         b->shader->info.tess.ccw = primitive == 4;
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_HS_FORK_PHASE_INSTANCE_COUNT:
      case D12_DXBC_OP_DCL_HS_JOIN_PHASE_INSTANCE_COUNT:
         if (avail != 1 || !args[0] || args[0] > 32 ||
             ctx->hs_instance_phase_open) {
            fail(ctx, "invalid hull fork/join instance count");
            return;
         }
         nir_push_if(b, nir_ult_imm(b, nir_load_invocation_id(b), args[0]));
         ctx->hs_instance_phase_open = true;
         ctx->hs_max_invocations = MAX2(ctx->hs_max_invocations, args[0]);
         ctx->tok += len;
         continue;

      case D12_DXBC_OP_DCL_INPUT_PS:
      case D12_DXBC_OP_DCL_INPUT_PS_SGV:
      case D12_DXBC_OP_DCL_INPUT_PS_SIV: {
         struct d12_dxbc_operand input;
         unsigned used = decode_operand(ctx, args, avail, &input);
         if (!used || input.file != D12_DXBC_FILE_INPUT ||
             !input.num_indices || input.index_is_relative[0] ||
             input.index[0] >= D12_DXBC_MAX_SIG) {
            fail(ctx, "invalid pixel-shader input declaration");
            return;
         }
         ctx->input_interp[input.index[0]] = (op_tok >> 11) & 0xf;
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_TGSM_RAW:
      case D12_DXBC_OP_DCL_TGSM_STRUCTURED: {
         struct d12_dxbc_operand g;
         unsigned used = decode_operand(ctx, args, avail, &g);
         bool structured = opcode == D12_DXBC_OP_DCL_TGSM_STRUCTURED;
         unsigned trailing = structured ? 2 : 1;
         if (!used || used + trailing != avail ||
             g.file != D12_DXBC_FILE_TGSM || g.num_indices != 1 ||
             g.index_is_relative[0] || g.index[0] >= D12_DXBC_MAX_SIG ||
             ctx->tgsm[g.index[0]].valid) {
            fail(ctx, "invalid thread-group shared-memory declaration");
            return;
         }

         uint32_t stride = structured ? args[used] : 0;
         uint32_t count_or_size = args[used + structured];
         uint64_t size = structured
                            ? (uint64_t)stride * count_or_size
                            : count_or_size;
         /* D3D11/D3D12 expose 32 KiB of groupshared memory per group. The
          * public declaration sizes are dword-aligned; reject malformed or
          * overflowing byte counts before they reach the backend launch. */
         if (!size || size > 32 * 1024 || (size & 3) ||
             (structured && (!stride || (stride & 3))) ||
             (uint64_t)ctx->shared_size + size > 32 * 1024) {
            fail(ctx, "invalid thread-group shared-memory size");
            return;
         }

         ctx->shared_size = ALIGN_POT(ctx->shared_size, 4);
         ctx->tgsm[g.index[0]] = (struct d12_dxbc_tgsm_decl) {
            .base = ctx->shared_size,
            .size = size,
            .stride = stride,
            .valid = true,
         };
         ctx->shared_size += size;
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_GLOBAL_FLAGS:
         /* Public DXBC bit 13 is [earlydepthstencil].  Preserve it in NIR
          * so AGX schedules depth/stencil tests before fragment side effects
          * instead of allowing depth-failing invocations to reach UAV
          * atomics. */
         if (b->shader->info.stage == MESA_SHADER_FRAGMENT &&
             (op_tok & BITFIELD_BIT(13)))
            b->shader->info.fs.early_fragment_tests = true;
         ctx->tok += len;
         continue;

      case D12_DXBC_OP_DCL_INDEXABLE_TEMP: {
         /* The declaration payload is x-register id, array length and
          * component count.  Keep vec4 elements internally; destination
          * masks and source swizzles preserve declarations with fewer live
          * components while making dynamic indexing straightforward. */
         if (avail < 3 || args[0] >= D12_DXBC_MAX_SIG || !args[1] ||
             args[1] > D12_DXBC_MAX_TEMPS || !args[2] || args[2] > 4 ||
             ctx->indexable_temps[args[0]]) {
            fail(ctx, "invalid indexable temporary declaration");
            return;
         }

         ctx->indexable_temps[args[0]] = nir_local_variable_create(
            b->impl, glsl_array_type(glsl_vec4_type(), args[1], 0), "x");
         ctx->indexable_temp_size[args[0]] = args[1];
         ctx->indexable_temp_shadow[args[0]] =
            rzalloc_array(b->shader, nir_def *, args[1]);
         if (!ctx->indexable_temp_shadow[args[0]]) {
            fail(ctx, "out of memory for indexable temporary shadow");
            return;
         }
         ctx->indexable_temp_shadow_valid[args[0]] = true;
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_TEMPS:
      case D12_DXBC_OP_DCL_INPUT:
      case D12_DXBC_OP_DCL_INPUT_SGV:
      case D12_DXBC_OP_DCL_INPUT_SIV:
      case D12_DXBC_OP_DCL_OUTPUT:
      case D12_DXBC_OP_DCL_OUTPUT_SIV:
      case D12_DXBC_OP_DCL_INDEX_RANGE:
      case D12_DXBC_OP_DCL_OUTPUT_SGV:
      case D12_DXBC_OP_DCL_STREAM:
      case D12_DXBC_OP_DCL_HS_MAX_TESSFACTOR:
         /* Signature and binding information all comes from the
          * container's ISGN/OSGN chunks and the root signature, so the
          * declarations themselves carry nothing this driver needs. */
         ctx->tok += len;
         continue;

      case D12_DXBC_OP_DCL_GS_OUTPUT_TOPOLOGY: {
         /* Public D3D10_SB_PRIMITIVE_TOPOLOGY values in opcode bits 17:11.
          * D3D geometry shaders may emit point lists, line strips, or
          * triangle strips. Poly needs the exact output topology before it
          * can count emitted primitives and build its rasterization shader. */
         switch ((op_tok >> 11) & 0x7f) {
         case 1:
            b->shader->info.gs.output_primitive = MESA_PRIM_POINTS;
            break;
         case 3:
            b->shader->info.gs.output_primitive = MESA_PRIM_LINE_STRIP;
            break;
         case 5:
            b->shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
            break;
         default:
            fail(ctx, "invalid geometry output topology");
            return;
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_MAX_OUTPUT_VERTEX_COUNT: {
         if (avail < 1 || !args[0] || args[0] > 1024) {
            fail(ctx, "invalid geometry maximum output vertex count");
            return;
         }
         b->shader->info.gs.vertices_out = args[0];
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_GS_INSTANCE_COUNT: {
         if (avail < 1 || !args[0] || args[0] > 32) {
            fail(ctx, "invalid geometry instance count");
            return;
         }
         b->shader->info.gs.invocations = args[0];
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_GS_INPUT_PRIMITIVE: {
         /* Public D3D10_SB_PRIMITIVE values in opcode bits 16:11. */
         unsigned primitive = (op_tok >> 11) & 0x1f;
         switch (primitive) {
         case 1:
            b->shader->info.gs.input_primitive = MESA_PRIM_POINTS;
            break;
         case 2:
            b->shader->info.gs.input_primitive = MESA_PRIM_LINES;
            break;
         case 3:
            b->shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
            break;
         case 6:
            b->shader->info.gs.input_primitive =
               MESA_PRIM_LINES_ADJACENCY;
            break;
         case 7:
            b->shader->info.gs.input_primitive =
               MESA_PRIM_TRIANGLES_ADJACENCY;
            break;
         default:
            fail(ctx, "invalid geometry input primitive");
            return;
         }
         b->shader->info.gs.vertices_in =
            mesa_vertices_per_prim(b->shader->info.gs.input_primitive);
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_SAMPLER: {
         /* SM 5.1 sampler declarations spell
          * s<range-id>[lower:upper], space. Preserve that mapping so a
          * relative use can index the bound sampler heap rather than
          * aliasing every access to range id zero. */
         struct d12_dxbc_operand r;
         unsigned n = decode_operand(ctx, args, avail, &r);
         if (n && r.file == D12_DXBC_FILE_SAMPLER &&
             r.num_indices == 3 && !r.index_is_relative[0] &&
             !r.index_is_relative[1] && !r.index_is_relative[2] &&
             r.index[0] < D12_DXBC_MAX_SIG && n < avail) {
            ctx->sampler_range[r.index[0]] =
               (struct d12_dxbc_binding_range) {
                  .lower_bound = r.index[1],
                  .upper_bound = r.index[2],
                  .space = args[n],
                  .valid = true,
               };
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_CONSTANT_BUFFER: {
         /* SM 5.1 declarations are cb<id>[lower:upper], size, space.
          * Preserve the ID mapping; accesses carry only the ID and absolute
          * register, so the space cannot otherwise be recovered. */
         struct d12_dxbc_operand cb;
         unsigned n = decode_operand(ctx, args, avail, &cb);
         if (n && cb.file == D12_DXBC_FILE_CONSTANT_BUFFER &&
             cb.num_indices == 3 && !cb.index_is_relative[0] &&
             !cb.index_is_relative[1] && !cb.index_is_relative[2] &&
             cb.index[0] < D12_DXBC_MAX_SIG && n + 2 <= avail) {
            ctx->cb_range[cb.index[0]] = (struct d12_dxbc_binding_range) {
               .lower_bound = cb.index[1],
               .upper_bound = cb.index[2],
               .space = args[n + 1],
               .valid = true,
            };
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_RESOURCE: {
         /* The resource dimension is opcode control state, while the t#
          * register is an ordinary operand. Keep both so LD can distinguish
          * Buffer<T> from Texture2D<T> instead of treating every SRV as 2D.
          */
         struct d12_dxbc_operand r;
         unsigned n = decode_operand(ctx, args, avail, &r);
         if (n && r.file == D12_DXBC_FILE_RESOURCE &&
             r.index[0] < D12_DXBC_MAX_SIG) {
            ctx->srv_dim[r.index[0]] =
               (op_tok & D12_DXBC_RESOURCE_DIMENSION_MASK) >>
               D12_DXBC_RESOURCE_DIMENSION_SHIFT;

            /* SM 5.1 typed-resource declarations spell
             * t<range-id>[lower:upper], return-types, space. Texture array
             * operands retain only the range id plus absolute register, so
             * preserve the bounds/space just like structured declarations
             * do below. */
            if (r.num_indices == 3 && !r.index_is_relative[0] &&
                !r.index_is_relative[1] && !r.index_is_relative[2] &&
                n + 2 <= avail) {
               ctx->srv_range[r.index[0]] =
                  (struct d12_dxbc_binding_range) {
                     .lower_bound = r.index[1],
                     .upper_bound = r.index[2],
                     .space = args[n + 1],
                     .valid = true,
                  };
            }
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_RESOURCE_RAW: {
         struct d12_dxbc_operand r;
         unsigned n = decode_operand(ctx, args, avail, &r);
         if (n && r.file == D12_DXBC_FILE_RESOURCE &&
             r.index[0] < D12_DXBC_MAX_SIG) {
            ctx->srv_dim[r.index[0]] = D12_DXBC_RESOURCE_DIMENSION_BUFFER;
            ctx->srv_raw[r.index[0]] = true;

            /* SM 5.1 raw declarations append the register space directly
             * after t<range-id>[lower:upper]. */
            if (r.num_indices == 3 && !r.index_is_relative[0] &&
                !r.index_is_relative[1] && !r.index_is_relative[2] &&
                n < avail) {
               ctx->srv_range[r.index[0]] =
                  (struct d12_dxbc_binding_range) {
                     .lower_bound = r.index[1],
                     .upper_bound = r.index[2],
                     .space = args[n],
                     .valid = true,
                  };
            }
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_UAV_TYPED:
      case D12_DXBC_OP_DCL_UAV_RAW: {
         struct d12_dxbc_operand r;
         unsigned n = decode_operand(ctx, args, avail, &r);
         if (n && r.file == D12_DXBC_FILE_UAV &&
             r.index[0] < D12_DXBC_MAX_SIG) {
            if (opcode == D12_DXBC_OP_DCL_UAV_RAW) {
               ctx->uav_dim[r.index[0]] = D12_DXBC_RESOURCE_DIMENSION_BUFFER;
               ctx->uav_raw[r.index[0]] = true;
            } else {
               ctx->uav_dim[r.index[0]] =
                  (op_tok & D12_DXBC_RESOURCE_DIMENSION_MASK) >>
                  D12_DXBC_RESOURCE_DIMENSION_SHIFT;
               if (n < avail)
                  ctx->uav_return_type[r.index[0]] = args[n] & 0xf;
            }

            unsigned space_word =
               n + (opcode == D12_DXBC_OP_DCL_UAV_TYPED ? 1 : 0);
            if (r.num_indices == 3 && !r.index_is_relative[0] &&
                !r.index_is_relative[1] && !r.index_is_relative[2] &&
                space_word < avail) {
               ctx->uav_range[r.index[0]] =
                  (struct d12_dxbc_binding_range) {
                     .lower_bound = r.index[1],
                     .upper_bound = r.index[2],
                     .space = args[space_word],
                     .valid = true,
                  };
            }
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_UAV_STRUCTURED:
      case D12_DXBC_OP_DCL_RESOURCE_STRUCTURED: {
         /* operand(u#/t#) then the stride. Decode the operand to find
          * the register, since its own length varies. */
         struct d12_dxbc_operand r;
         unsigned n = decode_operand(ctx, args, avail, &r);
         if (n && avail > n && r.index[0] < D12_DXBC_MAX_SIG) {
            if (opcode == D12_DXBC_OP_DCL_UAV_STRUCTURED)
               ctx->uav_stride[r.index[0]] = args[n];
            else
               ctx->srv_stride[r.index[0]] = args[n];

            if (opcode == D12_DXBC_OP_DCL_UAV_STRUCTURED)
               ctx->uav_dim[r.index[0]] =
                  D12_DXBC_RESOURCE_DIMENSION_BUFFER;
            else
               ctx->srv_dim[r.index[0]] =
                  D12_DXBC_RESOURCE_DIMENSION_BUFFER;

            if (r.num_indices == 3 && !r.index_is_relative[0] &&
                !r.index_is_relative[1] && !r.index_is_relative[2] &&
                n + 2 <= avail) {
               struct d12_dxbc_binding_range *range =
                  opcode == D12_DXBC_OP_DCL_UAV_STRUCTURED
                     ? &ctx->uav_range[r.index[0]]
                     : &ctx->srv_range[r.index[0]];
               *range =
                  (struct d12_dxbc_binding_range) {
                     .lower_bound = r.index[1],
                     .upper_bound = r.index[2],
                     .space = args[n + 1],
                     .valid = true,
                  };
            }
         }
         ctx->tok += len;
         continue;
      }

      case D12_DXBC_OP_DCL_THREAD_GROUP:
         if (avail >= 3) {
            b->shader->info.workgroup_size[0] = args[0];
            b->shader->info.workgroup_size[1] = args[1];
            b->shader->info.workgroup_size[2] = args[2];
         }
         ctx->tok += len;
         continue;

      default:
         break;
      }

      /* Texture instructions have a fixed operand count. fxc can include
       * zero padding inside the encoded instruction length (notably
       * sample_l_s). Decoding that padding as a TEMP operand made the last
       * source become r0 instead of the explicit LOD. Match the public
       * formats audited against DXVK's dxbc_defs.cpp, and reject unknown
       * nonzero payload rather than inventing another operand. */
      unsigned operand_limit = D12_DXBC_MAX_OPERANDS;
      switch (opcode) {
      case D12_DXBC_OP_LD:
      case D12_DXBC_OP_LD_RAW:
      case D12_DXBC_OP_LD_UAV_TYPED: operand_limit = 3; break;
      case D12_DXBC_OP_SAMPLE:
      case D12_DXBC_OP_GATHER4:
      case D12_DXBC_OP_LD_MS:
      case D12_DXBC_OP_LD_STRUCTURED:
      case D12_DXBC_OP_LD_S:
      case D12_DXBC_OP_LD_RAW_S:
      case D12_DXBC_OP_LD_UAV_TYPED_S: operand_limit = 4; break;
      case D12_DXBC_OP_SAMPLE_C:
      case D12_DXBC_OP_SAMPLE_C_LZ:
      case D12_DXBC_OP_SAMPLE_L:
      case D12_DXBC_OP_SAMPLE_B:
      case D12_DXBC_OP_GATHER4_C:
      case D12_DXBC_OP_GATHER4_PO:
      case D12_DXBC_OP_GATHER4_S:
      case D12_DXBC_OP_LD_MS_S:
      case D12_DXBC_OP_LD_STRUCTURED_S: operand_limit = 5; break;
      case D12_DXBC_OP_SAMPLE_D:
      case D12_DXBC_OP_GATHER4_PO_C:
      case D12_DXBC_OP_GATHER4_C_S:
      case D12_DXBC_OP_GATHER4_PO_S:
      case D12_DXBC_OP_SAMPLE_L_S:
      case D12_DXBC_OP_SAMPLE_C_LZ_S:
      case D12_DXBC_OP_SAMPLE_CLAMP_S: operand_limit = 6; break;
      case D12_DXBC_OP_GATHER4_PO_C_S:
      case D12_DXBC_OP_SAMPLE_B_CLAMP_S:
      case D12_DXBC_OP_SAMPLE_C_CLAMP_S: operand_limit = 7; break;
      default: break;
      }
      while (avail && nops < operand_limit) {
         unsigned n = decode_operand(ctx, args, avail, &o[nops]);
         if (!n) {
            fail(ctx, "malformed operand");
            return;
         }
         args += n;
         avail -= n;
         nops++;
      }
      if (operand_limit != D12_DXBC_MAX_OPERANDS) {
         while (avail) {
            if (*args++) {
               fail(ctx, "nonzero payload after texture operands");
               return;
            }
            avail--;
         }
      }

#define SRC(i) read_src(ctx, &o[i])
#define ISRC(i) read_src_int(ctx, &o[i])
#define DST(v) write_dst(ctx, &o[0], apply_dst_saturate(b, op_tok, (v)))
#define NEED(n)                                                              \
   do {                                                                      \
      if (nops < (n)) {                                                      \
         snprintf(ctx->errbuf, sizeof(ctx->errbuf),                          \
                  "DXBC opcode %u has %u operands, needs %u", opcode,       \
                  nops, (unsigned)(n));                                      \
         fail(ctx, ctx->errbuf);                                             \
         return;                                                             \
      }                                                                      \
   } while (0)

      /* SM5 sparse instructions insert a residency destination after the
       * data destination, and the clamp forms append an optional minimum
       * LOD. Normalize their operand shape before sharing ordinary texture
       * translation. This follows DXVK's zlib-licensed instruction formats
       * (v2.7.1 dxbc_defs.cpp); the NIR residency path is shared with DXIL.
       */
      unsigned ordinary_opcode = opcode;
      bool instruction_clamp = false;
      switch (opcode) {
      case D12_DXBC_OP_GATHER4_S: ordinary_opcode = D12_DXBC_OP_GATHER4; break;
      case D12_DXBC_OP_GATHER4_C_S: ordinary_opcode = D12_DXBC_OP_GATHER4_C; break;
      case D12_DXBC_OP_GATHER4_PO_S: ordinary_opcode = D12_DXBC_OP_GATHER4_PO; break;
      case D12_DXBC_OP_GATHER4_PO_C_S: ordinary_opcode = D12_DXBC_OP_GATHER4_PO_C; break;
      case D12_DXBC_OP_LD_S: ordinary_opcode = D12_DXBC_OP_LD; break;
      case D12_DXBC_OP_LD_MS_S: ordinary_opcode = D12_DXBC_OP_LD_MS; break;
      case D12_DXBC_OP_LD_UAV_TYPED_S: ordinary_opcode = D12_DXBC_OP_LD_UAV_TYPED; break;
      case D12_DXBC_OP_LD_RAW_S: ordinary_opcode = D12_DXBC_OP_LD_RAW; break;
      case D12_DXBC_OP_LD_STRUCTURED_S: ordinary_opcode = D12_DXBC_OP_LD_STRUCTURED; break;
      case D12_DXBC_OP_SAMPLE_L_S: ordinary_opcode = D12_DXBC_OP_SAMPLE_L; break;
      case D12_DXBC_OP_SAMPLE_C_LZ_S: ordinary_opcode = D12_DXBC_OP_SAMPLE_C_LZ; break;
      case D12_DXBC_OP_SAMPLE_CLAMP_S:
         ordinary_opcode = D12_DXBC_OP_SAMPLE; instruction_clamp = true; break;
      case D12_DXBC_OP_SAMPLE_B_CLAMP_S:
         ordinary_opcode = D12_DXBC_OP_SAMPLE_B; instruction_clamp = true; break;
      case D12_DXBC_OP_SAMPLE_D_CLAMP_S:
         ordinary_opcode = D12_DXBC_OP_SAMPLE_D; instruction_clamp = true; break;
      case D12_DXBC_OP_SAMPLE_C_CLAMP_S:
         ordinary_opcode = D12_DXBC_OP_SAMPLE_C; instruction_clamp = true; break;
      default: break;
      }
      bool sparse = ordinary_opcode != opcode;
      struct d12_dxbc_operand feedback_dst = {0};
      nir_def *instruction_min_lod = NULL;
      if (sparse) {
         NEED(2);
         feedback_dst = o[1];
         memmove(&o[1], &o[2], (nops - 2) * sizeof(o[0]));
         nops--;
         opcode = ordinary_opcode;
         if (instruction_clamp) {
            NEED(5);
            if (o[nops - 1].file != D12_DXBC_FILE_NULL)
               instruction_min_lod = nir_channel(b, SRC(nops - 1), 0);
            nops--;
         }
         /* A null feedback destination needs the same sampling/clamp
          * semantics, but no hardware residency result or extra lane. */
         sparse = feedback_dst.file != D12_DXBC_FILE_NULL;
      }

      switch (opcode) {
      case D12_DXBC_OP_CHECK_ACCESS_FULLY_MAPPED: {
         NEED(2);
         nir_def *resident = nir_is_sparse_texels_resident(
            b, 1, nir_channel(b, SRC(1), 0));
         DST(nir_replicate(b, nir_bcsel(b, resident, nir_imm_int(b, -1),
                                       nir_imm_int(b, 0)), 4));
         break;
      }
      case D12_DXBC_OP_MOV:      NEED(2); DST(SRC(1)); break;
      case D12_DXBC_OP_ADD:      NEED(3); DST(nir_fadd(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_MUL:      NEED(3); DST(nir_fmul(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_DIV:      NEED(3); DST(nir_fdiv(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_MAD:      NEED(4); DST(nir_ffma(b, SRC(1), SRC(2), SRC(3))); break;
      case D12_DXBC_OP_MIN:      NEED(3); DST(nir_fmin(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_MAX:      NEED(3); DST(nir_fmax(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_FRC:      NEED(2); DST(nir_ffract(b, SRC(1))); break;
      case D12_DXBC_OP_EXP:      NEED(2); DST(nir_fexp2(b, SRC(1))); break;
      case D12_DXBC_OP_LOG:      NEED(2); DST(nir_flog2(b, SRC(1))); break;
      case D12_DXBC_OP_SQRT:     NEED(2); DST(nir_fsqrt(b, SRC(1))); break;
      case D12_DXBC_OP_RSQ:      NEED(2); DST(nir_frsq(b, SRC(1))); break;
      case D12_DXBC_OP_ROUND_NE: NEED(2); DST(nir_fround_even(b, SRC(1))); break;
      case D12_DXBC_OP_ROUND_NI: NEED(2); DST(nir_ffloor(b, SRC(1))); break;
      case D12_DXBC_OP_ROUND_PI: NEED(2); DST(nir_fceil(b, SRC(1))); break;
      case D12_DXBC_OP_ROUND_Z:  NEED(2); DST(nir_ftrunc(b, SRC(1))); break;
      case D12_DXBC_OP_SINCOS: {
         /* Two independently masked destinations: sine first, cosine
          * second. Either destination may be the public null register. */
         NEED(3);
         nir_def *source = SRC(2);
         write_dst(ctx, &o[0],
                   apply_dst_saturate(b, op_tok, nir_fsin(b, source)));
         write_dst(ctx, &o[1],
                   apply_dst_saturate(b, op_tok, nir_fcos(b, source)));
         break;
      }

      case D12_DXBC_OP_DP2:
      case D12_DXBC_OP_DP3:
      case D12_DXBC_OP_DP4: {
         NEED(3);
         unsigned n = opcode == D12_DXBC_OP_DP2 ? 2
                    : opcode == D12_DXBC_OP_DP3 ? 3 : 4;
         nir_def *d = nir_fdot(b, nir_trim_vector(b, SRC(1), n),
                               nir_trim_vector(b, SRC(2), n));
         /* A dot product replicates its scalar across the write mask. */
         DST(nir_replicate(b, d, 4));
         break;
      }

      /* Integer opcodes operate on the same registers, reinterpreted. */
      case D12_DXBC_OP_IADD: NEED(3); DST(nir_iadd(b, ISRC(1), ISRC(2))); break;
      case D12_DXBC_OP_IMUL: {
         NEED(4);
         nir_def *a = SRC(2), *c = SRC(3);
         /* SM5 spells this as imul destHI, destLO, src0, src1. The common
          * address-calculation form deliberately sends destHI to null. */
         write_dst(ctx, &o[0], nir_imul_high(b, a, c));
         write_dst(ctx, &o[1], nir_imul(b, a, c));
         break;
      }
      case D12_DXBC_OP_IMAD: NEED(4); DST(nir_iadd(b, nir_imul(b, ISRC(1), ISRC(2)), ISRC(3))); break;
      case D12_DXBC_OP_IMAX: NEED(3); DST(nir_imax(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_IMIN: NEED(3); DST(nir_imin(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_UMAX: NEED(3); DST(nir_umax(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_UMIN: NEED(3); DST(nir_umin(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_INEG: NEED(2); DST(nir_ineg(b, SRC(1))); break;
      case D12_DXBC_OP_AND:  NEED(3); DST(nir_iand(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_OR:   NEED(3); DST(nir_ior(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_XOR:  NEED(3); DST(nir_ixor(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_NOT:  NEED(2); DST(nir_inot(b, SRC(1))); break;
      case D12_DXBC_OP_ISHL: NEED(3); DST(nir_ishl(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_ISHR: NEED(3); DST(nir_ishr(b, SRC(1), SRC(2))); break;
      case D12_DXBC_OP_USHR: NEED(3); DST(nir_ushr(b, SRC(1), SRC(2))); break;

      case D12_DXBC_OP_ITOF: NEED(2); DST(nir_i2f32(b, SRC(1))); break;
      case D12_DXBC_OP_UTOF: NEED(2); DST(nir_u2f32(b, SRC(1))); break;
      case D12_DXBC_OP_FTOI: NEED(2); DST(nir_f2i32(b, SRC(1))); break;
      case D12_DXBC_OP_FTOU: NEED(2); DST(nir_f2u32(b, SRC(1))); break;
      case D12_DXBC_OP_F32TOF16:
      case D12_DXBC_OP_F16TOF32: {
         NEED(2);
         nir_def *source = SRC(1);
         nir_def *component[4];
         for (unsigned c = 0; c < 4; c++) {
            nir_def *scalar = nir_channel(b, source, c);
            if (opcode == D12_DXBC_OP_F32TOF16) {
               /* SM5 f32tof16 clamps finite overflow to the largest finite
                * half while preserving infinities and NaNs.  AGX's native
                * f32-to-f16 conversion otherwise rounds finite overflow to
                * infinity. */
               nir_def *exponent = nir_iand_imm(b, scalar, 0x7f800000);
               nir_def *finite = nir_ine_imm(b, exponent, 0x7f800000);
               scalar = nir_bcsel(
                  b, finite,
                  nir_fclamp(b, scalar, nir_imm_float(b, -65504.0f),
                             nir_imm_float(b, 65504.0f)),
                  scalar);

               /* The half bit pattern occupies the low 16 bits of each
                * 32-bit destination lane. */
               component[c] = nir_pack_half_2x16_split(
                  b, scalar, nir_imm_float(b, 0.0f));
            } else {
               component[c] = nir_channel(
                  b, nir_unpack_half_2x16(b, scalar), 0);
            }
         }
         DST(nir_vec(b, component, ARRAY_SIZE(component)));
         break;
      }

      /* Comparisons produce an all-ones or all-zero mask, not NIR's 1/0
       * integer conversion of a boolean. Negating the 1/0 value preserves
       * DXBC bitwise use of comparison results as well as control flow. */
#define DXBC_MASK(cmp) nir_ineg(b, nir_b2i32(b, (cmp)))
      case D12_DXBC_OP_EQ:  NEED(3); DST(DXBC_MASK(nir_feq(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_NE:  NEED(3); DST(DXBC_MASK(nir_fneu(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_LT:  NEED(3); DST(DXBC_MASK(nir_flt(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_GE:  NEED(3); DST(DXBC_MASK(nir_fge(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_IEQ: NEED(3); DST(DXBC_MASK(nir_ieq(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_INE: NEED(3); DST(DXBC_MASK(nir_ine(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_ILT: NEED(3); DST(DXBC_MASK(nir_ilt(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_IGE: NEED(3); DST(DXBC_MASK(nir_ige(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_ULT: NEED(3); DST(DXBC_MASK(nir_ult(b, SRC(1), SRC(2)))); break;
      case D12_DXBC_OP_UGE: NEED(3); DST(DXBC_MASK(nir_uge(b, SRC(1), SRC(2)))); break;
#undef DXBC_MASK

      case D12_DXBC_OP_MOVC:
         NEED(4);
         DST(nir_bcsel(b, nir_ine_imm(b, SRC(1), 0), SRC(2), SRC(3)));
         break;

      case D12_DXBC_OP_SWAPC: {
         /* The sources must all be captured before either destination is
          * written because DXBC permits every source/destination aliasing
          * combination.  A true condition swaps the two source lanes. */
         NEED(5);
         nir_def *cond = nir_ine_imm(b, SRC(2), 0);
         nir_def *first = SRC(3);
         nir_def *second = SRC(4);
         nir_def *dst0 = nir_bcsel(b, cond, second, first);
         nir_def *dst1 = nir_bcsel(b, cond, first, second);
         write_dst(ctx, &o[0], apply_dst_saturate(b, op_tok, dst0));
         write_dst(ctx, &o[1], apply_dst_saturate(b, op_tok, dst1));
         break;
      }

      /*
       * D3D's bit-field ops are defined with explicit 5-bit masking of
       * the width and offset. GLSL's bitfieldInsert leaves the
       * out-of-range case undefined, so the formula is spelled out
       * rather than mapped onto it -- a shader that relies on the
       * documented wrap must not depend on a lowering's opinion.
       *
       *   width  = src0 & 31,  offset = src1 & 31
       *   mask   = ((1 << width) - 1) << offset
       *   dest   = ((src2 << offset) & mask) | (src3 & ~mask)
       */
      case D12_DXBC_OP_BFI: {
         NEED(5);
         nir_def *width = nir_iand_imm(b, SRC(1), 31);
         nir_def *offset = nir_iand_imm(b, SRC(2), 31);
         nir_def *ones = nir_imm_int(b, 1);
         nir_def *mask =
            nir_ishl(b, nir_isub(b, nir_ishl(b, ones, width), ones), offset);
         nir_def *ins = nir_iand(b, nir_ishl(b, SRC(3), offset), mask);
         nir_def *base = nir_iand(b, SRC(4), nir_inot(b, mask));
         DST(nir_ior(b, ins, base));
         break;
      }

      case D12_DXBC_OP_UBFE:
      case D12_DXBC_OP_IBFE: {
         NEED(4);
         nir_def *width = nir_iand_imm(b, SRC(1), 31);
         nir_def *offset = nir_iand_imm(b, SRC(2), 31);
         nir_def *bits = nir_umin(b, width,
                                  nir_isub(b, nir_imm_int(b, 32), offset));
         nir_def *shift = nir_isub(b, nir_imm_int(b, 32), bits);
         nir_def *value = nir_ushr(b, SRC(3), offset);
         nir_def *result = opcode == D12_DXBC_OP_UBFE
                              ? nir_ushr(b, nir_ishl(b, value, shift), shift)
                              : nir_ishr(b, nir_ishl(b, value, shift), shift);
         DST(nir_bcsel(b, nir_ieq_imm(b, bits, 0), nir_imm_int(b, 0),
                       result));
         break;
      }

      case D12_DXBC_OP_RCP:       NEED(2); DST(nir_frcp(b, SRC(1))); break;
      case D12_DXBC_OP_DERIV_RTX:
      case D12_DXBC_OP_DERIV_RTY:
      case D12_DXBC_OP_DERIV_RTX_COARSE:
      case D12_DXBC_OP_DERIV_RTX_FINE:
      case D12_DXBC_OP_DERIV_RTY_COARSE:
      case D12_DXBC_OP_DERIV_RTY_FINE:
         NEED(2); DST(emit_derivative(ctx, opcode, &o[1])); break;
      case D12_DXBC_OP_COUNTBITS: NEED(2); DST(nir_bit_count(b, SRC(1))); break;
      case D12_DXBC_OP_FIRSTBIT_LO:  NEED(2); DST(nir_find_lsb(b, SRC(1))); break;
      case D12_DXBC_OP_FIRSTBIT_HI:
      case D12_DXBC_OP_FIRSTBIT_SHI: {
         NEED(2);
         nir_def *value = ISRC(1);

         /* AGX ffs reports the bit position from the MSB, whereas DXBC
          * firstbit_hi reports it from the LSB.  Signed firstbit_hi searches
          * for the first bit different from the sign bit. */
         if (opcode == D12_DXBC_OP_FIRSTBIT_SHI) {
            nir_def *sign = nir_ishr_imm(b, value, 31);
            value = nir_ixor(b, value, sign);
         }

         nir_def *from_msb = nir_ufind_msb(b, value);
         nir_def *from_lsb = nir_isub(b, nir_imm_int(b, 31), from_msb);
         DST(nir_bcsel(b, nir_ieq_imm(b, value, 0),
                       nir_imm_int(b, -1), from_lsb));
         break;
      }
      case D12_DXBC_OP_BFREV:     NEED(2); DST(nir_bitfield_reverse(b, SRC(1))); break;
      case D12_DXBC_OP_UMAD:      NEED(4); DST(nir_iadd(b, nir_imul(b, SRC(1), SRC(2)), SRC(3))); break;
      case D12_DXBC_OP_UDIV: {
         /* DXBC udiv has two destinations: quotient, then remainder.  The
          * latter is observable even when the quotient mask is empty (HLSL
          * `%` commonly emits exactly that form). */
         NEED(4);
         nir_def *numerator = SRC(2);
         nir_def *denominator = SRC(3);
         nir_def *zero = nir_ieq_imm(b, denominator, 0);
         nir_def *undefined = nir_imm_int(b, -1);
         write_dst(ctx, &o[0],
                   nir_bcsel(b, zero, undefined,
                             nir_udiv(b, numerator, denominator)));
         write_dst(ctx, &o[1],
                   nir_bcsel(b, zero, undefined,
                             nir_umod(b, numerator, denominator)));
         break;
      }
      case D12_DXBC_OP_UMUL: {
         NEED(4);
         nir_def *a = SRC(2), *c = SRC(3);
         write_dst(ctx, &o[0], nir_umul_high(b, a, c));
         write_dst(ctx, &o[1], nir_imul(b, a, c));
         break;
      }
      case D12_DXBC_OP_MSAD: {
         NEED(4);
         nir_def *ref = SRC(1);
         nir_def *src = SRC(2);
         nir_def *acc = SRC(3);

         /* Per D3D11.1, zero reference bytes are masked out. Expand the
          * operation because AGX has no native MSAD instruction. */
         for (unsigned i = 0; i < 4; i++) {
            nir_def *rb =
               nir_iand_imm(b, nir_ushr_imm(b, ref, i * 8), 0xff);
            nir_def *sb =
               nir_iand_imm(b, nir_ushr_imm(b, src, i * 8), 0xff);
            nir_def *diff = nir_iabs(b, nir_isub(b, rb, sb));
            acc = nir_iadd(
               b, acc,
               nir_bcsel(b, nir_ine_imm(b, rb, 0), diff,
                          nir_imm_int(b, 0)));
         }
         DST(acc);
         break;
      }
      case D12_DXBC_OP_EVAL_SAMPLE_INDEX: {
         NEED(3);
         nir_def *sample = nir_channel(b, SRC(2), 0);
         nir_def *bary = nir_load_barycentric_at_sample(b, 32, sample);
         DST(read_interpolated_src(ctx, &o[1], bary));
         break;
      }
      case D12_DXBC_OP_EVAL_CENTROID: {
         NEED(2);
         nir_def *bary = nir_load_barycentric_centroid(b, 32);
         DST(read_interpolated_src(ctx, &o[1], bary));
         break;
      }
      case D12_DXBC_OP_NOP:       break;

      /*
       * SM5 BUFINFO exposes the logical view size, not the backing BO size.
       * Descriptor upload has already packed the exact element/byte count in
       * the low word and a structured stride in the high word.  Restrict this
       * first implementation to statically indexed SM5.0 t#/u# operands;
       * SM5.1 range indexing needs a separate dynamic slot calculation.
       */
      case D12_DXBC_OP_BUFINFO: {
         NEED(2);

         const struct d12_dxbc_operand *res = &o[1];
         bool uav = res->file == D12_DXBC_FILE_UAV;
         if ((!uav && res->file != D12_DXBC_FILE_RESOURCE) ||
             (res->num_indices != 1 && res->num_indices != 2) ||
             (res->num_indices == 1 && res->index_is_relative[0])) {
            fail(ctx, "unsupported BUFINFO resource operand");
            return;
         }

         unsigned reg = res->index[0];
         if (reg >= D12_DXBC_MAX_SIG ||
             (uav ? ctx->uav_dim[reg] : ctx->srv_dim[reg]) !=
                D12_DXBC_RESOURCE_DIMENSION_BUFFER) {
            fail(ctx, "BUFINFO resource is not a declared buffer");
            return;
         }

         nir_def *count, *stride;
         buffer_view_info(ctx, res, &count, &stride);
         if (ctx->error)
            return;
         bool structured =
            (uav ? ctx->uav_stride[reg] : ctx->srv_stride[reg]) != 0;
         bool raw = uav ? ctx->uav_raw[reg] : ctx->srv_raw[reg];
         nir_def *dimension =
            raw ? nir_bcsel(b, nir_ine_imm(b, stride, 0),
                            nir_imul(b, count, stride), count)
                : count;
         nir_def *value =
            structured
               ? nir_vec4(b, count, stride, nir_imm_int(b, 0),
                          nir_imm_int(b, 1))
               : nir_replicate(b, dimension, 4);

         unsigned sw[4] = {res->swizzle[0], res->swizzle[1],
                           res->swizzle[2], res->swizzle[3]};
         DST(nir_swizzle(b, value, sw, 4));
         break;
      }

      /* RESINFO lowers to the native descriptor query for every texture
       * dimensionality whose AGX descriptor path is implemented below.
       */
      case D12_DXBC_OP_RESINFO: {
         NEED(3);

         const struct d12_dxbc_operand *res = &o[2];
         bool uav = res->file == D12_DXBC_FILE_UAV;
         unsigned return_type =
            (op_tok & D12_DXBC_RESINFO_RETURN_TYPE_MASK) >>
            D12_DXBC_RESINFO_RETURN_TYPE_SHIFT;
         if ((return_type != D12_DXBC_RESINFO_RETURN_UINT &&
              return_type != D12_DXBC_RESINFO_RETURN_FLOAT) ||
             (!uav && res->file != D12_DXBC_FILE_RESOURCE) ||
             (res->num_indices != 1 && res->num_indices != 2) ||
             (res->num_indices == 1 && res->index_is_relative[0])) {
            fail(ctx, "unsupported RESINFO resource operand");
            return;
         }

         unsigned reg = res->index[0];
         if (reg >= D12_DXBC_MAX_SIG) {
            fail(ctx, "RESINFO resource register out of range");
            return;
         }

         unsigned dim = uav ? ctx->uav_dim[reg] : ctx->srv_dim[reg];
         bool is_1d = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D ||
                      dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY;
         bool is_array =
            dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY ||
            dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY ||
            dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY ||
            dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY;
         bool is_ms = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS ||
                      dim ==
                         D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY;
         bool is_3d = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_3D;
         enum glsl_sampler_dim sampler_dim =
            is_3d   ? GLSL_SAMPLER_DIM_3D
            : is_1d ? GLSL_SAMPLER_DIM_1D
            : dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE ||
                  dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY
               ? GLSL_SAMPLER_DIM_CUBE
            : is_ms ? GLSL_SAMPLER_DIM_MS
                    : GLSL_SAMPLER_DIM_2D;
         if (dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE &&
             dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY && !is_3d) {
            fail(ctx, "unsupported RESINFO resource dimension");
            return;
         }

         nir_def *descriptor = uav ? uav_rooted_descriptor(ctx, res)
                                   : srv_rooted_descriptor(ctx, res);
         nir_def *handle;
         if (descriptor) {
            nir_def *index = rooted_native_index(ctx, descriptor);
            handle = nir_bindless_image_agx(
               b, nir_imul_imm(b, index,
                               ctx->native_layout.texture_stride_B),
               .desc_set = D12_UNIFORM_HEAP_TEXTURES);
         } else {
            nir_def *offset = uav ? uav_texture_offset(ctx, res)
                                  : srv_texture_offset(ctx, res);
            if (!offset) {
               if (d12_debug()->validate_bindings) {
                  const struct d12_dxbc_binding_range *range =
                     uav ? &ctx->uav_range[reg] : &ctx->srv_range[reg];
                  mesa_logi("DXBC RESINFO operand: uav=%u reg=%u "
                            "indices=%u idx0=%u idx1=%u rel0=%u rel1=%u "
                            "range-valid=%u lower=%u upper=%u space=%u",
                            uav, reg, res->num_indices, res->index[0],
                            res->index[1], res->index_is_relative[0],
                            res->index_is_relative[1], range->valid,
                            range->lower_bound, range->upper_bound,
                            range->space);
               }
               fail(ctx,
                    "RESINFO resource is not bound by the root signature");
               return;
            }
            handle = nir_bindless_image_agx(b, offset,
               .desc_set = res->num_indices == 2
                  ? D12_UNIFORM_HEAP_TEXTURES : D12_UNIFORM_TEXTURES);
         }
         nir_tex_instr *txs = nir_tex_instr_create(b->shader, is_ms ? 1 : 2);
         txs->op = nir_texop_txs;
         txs->sampler_dim = sampler_dim;
         txs->is_array = is_array;
         txs->dest_type = nir_type_uint32;
         txs->coord_components = is_3d   ? 3
                                 : is_1d ? (is_array ? 2 : 1)
                                 : sampler_dim == GLSL_SAMPLER_DIM_CUBE
                                    ? (is_array ? 4 : 3)
                                 : is_array ? 3
                                            : 2;
         txs->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle,
                                            handle);
         nir_def *lod = NULL;
         if (!is_ms)
            txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                              lod = nir_channel(b, SRC(1), 0));
         unsigned txs_comps = is_1d                 ? (is_array ? 2 : 1)
                              : (is_array || is_3d) ? 3
                                                    : 2;
         nir_def_init(&txs->instr, &txs->def, txs_comps, 32);
         nir_builder_instr_insert(b, &txs->instr);

         nir_tex_instr *query = nir_tex_instr_create(b->shader, 1);
         query->op = is_ms ? nir_texop_texture_samples
                           : nir_texop_query_levels;
         query->sampler_dim = sampler_dim;
         query->is_array = is_array;
         query->dest_type = nir_type_uint32;
         query->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle,
                                              handle);
         nir_def_init(&query->instr, &query->def, 1, 32);
         nir_builder_instr_insert(b, &query->instr);

         nir_def *width = nir_channel(b, &txs->def, 0);
         nir_def *height = is_1d ? (is_array ? nir_channel(b, &txs->def, 1)
                                             : nir_imm_int(b, 0))
                                 : nir_channel(b, &txs->def, 1);
         nir_def *depth_or_layers = (!is_1d && (is_array || is_3d))
                                       ? nir_channel(b, &txs->def, 2)
                                    : is_ms ? &query->def
                                            : nir_imm_int(b, 0);
         if (!is_ms) {
            /* AGX clamps an out-of-range TCS LOD to the last level. D3D12
             * RESINFO instead returns zero for every queried dimension while
             * retaining the resource's level count.
             */
            nir_def *in_range = nir_ult(b, lod, &query->def);
            width = nir_bcsel(b, in_range, width, nir_imm_int(b, 0));
            height = nir_bcsel(b, in_range, height, nir_imm_int(b, 0));
            if (is_array || is_3d)
               depth_or_layers = nir_bcsel(b, in_range, depth_or_layers,
                                           nir_imm_int(b, 0));
         }

         nir_def *value = nir_vec4(
            b, width, height, depth_or_layers,
            is_ms && is_array ? &query->def
                              : is_ms ? nir_imm_int(b, 0) : &query->def);
         if (return_type == D12_DXBC_RESINFO_RETURN_FLOAT)
            value = nir_u2f32(b, value);
         unsigned sw[4] = {res->swizzle[0], res->swizzle[1],
                           res->swizzle[2], res->swizzle[3]};
         DST(nir_swizzle(b, value, sw, 4));
         break;
      }

      /*
       * STRUCTURED CONTROL FLOW. The opcode's control bits say whether
       * the test is "non-zero" or "zero"; bit 18 carries it.
       */
      case D12_DXBC_OP_IF: {
         NEED(1);
         if (ctx->cf_depth >= ARRAY_SIZE(ctx->cf)) {
            fail(ctx, "control flow nested too deeply");
            return;
         }
         bool test_zero = !((op_tok >> 18) & 1);
         nir_def *c = nir_channel(b, SRC(0), 0);
         nir_if *nif = nir_push_if(
            b, test_zero ? nir_ieq_imm(b, c, 0) : nir_ine_imm(b, c, 0));
         ctx->cf[ctx->cf_depth] = (struct d12_dxbc_cf) {
            .kind = D12_DXBC_CF_IF,
            .nif = nif,
         };
         ctx->cf_depth++;
         break;
      }

      case D12_DXBC_OP_ELSE:
         if (!ctx->cf_depth ||
             ctx->cf[ctx->cf_depth - 1].kind != D12_DXBC_CF_IF) {
            fail(ctx, "else outside an if");
            return;
         }
         nir_push_else(b, NULL);
         ctx->cf[ctx->cf_depth - 1].in_else = true;
         break;

      case D12_DXBC_OP_ENDIF:
         if (!ctx->cf_depth ||
             ctx->cf[ctx->cf_depth - 1].kind != D12_DXBC_CF_IF) {
            fail(ctx, "endif outside an if");
            return;
         }
         nir_pop_if(b, NULL);
         ctx->cf_depth--;
         break;

      case D12_DXBC_OP_LOOP:
         if (ctx->cf_depth >= ARRAY_SIZE(ctx->cf)) {
            fail(ctx, "control flow nested too deeply");
            return;
         }
         nir_loop *loop = nir_push_loop(b);
         /* DXBC loops may contain CONTINUE/CONTINUEC.  NIR requires an
          * explicit continue construct before a continue jump can target the
          * loop; lower the construct again once translation is complete so
          * AGX sees ordinary structured control flow. */
         nir_loop_add_continue_construct(loop);
         ctx->cf[ctx->cf_depth].kind = D12_DXBC_CF_LOOP;
         ctx->cf[ctx->cf_depth].in_else = false;
         ctx->cf_depth++;
         break;

      case D12_DXBC_OP_ENDLOOP:
         if (!ctx->cf_depth ||
             ctx->cf[ctx->cf_depth - 1].kind != D12_DXBC_CF_LOOP) {
            fail(ctx, "endloop outside a loop");
            return;
         }
         nir_pop_loop(b, NULL);
         ctx->cf_depth--;
         break;

      case D12_DXBC_OP_SWITCH: {
         NEED(1);
         if (ctx->cf_depth >= ARRAY_SIZE(ctx->cf)) {
            fail(ctx, "control flow nested too deeply");
            return;
         }

         nir_def *value = nir_channel(b, SRC(0), 0);
         nir_def *any_match =
            d12_dxbc_switch_any_match(ctx, ctx->tok + len, value);
         if (ctx->error)
            return;

         nir_variable *matched = nir_local_variable_create(
            b->impl, glsl_bool_type(), "switch_matched");
         nir_store_var(b, matched, nir_imm_false(b), 1);
         nir_push_loop(b);

         ctx->cf[ctx->cf_depth] = (struct d12_dxbc_cf) {
            .kind = D12_DXBC_CF_SWITCH,
            .switch_value = value,
            .switch_any_match = any_match,
            .switch_matched = matched,
         };
         ctx->cf_depth++;
         break;
      }

      case D12_DXBC_OP_CASE:
      case D12_DXBC_OP_DEFAULT: {
         if (!ctx->cf_depth ||
             ctx->cf[ctx->cf_depth - 1].kind != D12_DXBC_CF_SWITCH) {
            fail(ctx, opcode == D12_DXBC_OP_CASE ? "case outside a switch"
                                                : "default outside a switch");
            return;
         }

         struct d12_dxbc_cf *sw = &ctx->cf[ctx->cf_depth - 1];
         if (sw->case_open)
            nir_pop_if(b, NULL);

         nir_def *matched = nir_load_var(b, sw->switch_matched);
         nir_def *enter;
         if (opcode == D12_DXBC_OP_CASE) {
            NEED(1);
            enter = nir_ior(
               b, matched,
               nir_ieq(b, sw->switch_value, nir_channel(b, SRC(0), 0)));
         } else {
            enter = nir_ior(b, matched, nir_inot(b, sw->switch_any_match));
         }

         nir_push_if(b, enter);
         nir_store_var(b, sw->switch_matched, nir_imm_true(b), 1);
         sw->case_open = true;
         break;
      }

      case D12_DXBC_OP_ENDSWITCH: {
         if (!ctx->cf_depth ||
             ctx->cf[ctx->cf_depth - 1].kind != D12_DXBC_CF_SWITCH) {
            fail(ctx, "endswitch outside a switch");
            return;
         }

         struct d12_dxbc_cf *sw = &ctx->cf[ctx->cf_depth - 1];
         if (sw->case_open)
            nir_pop_if(b, NULL);
         nir_jump(b, nir_jump_break);
         nir_pop_loop(b, NULL);
         ctx->cf_depth--;
         break;
      }

      case D12_DXBC_OP_BREAK:
         nir_jump(b, nir_jump_break);
         break;

      case D12_DXBC_OP_CONTINUE:
         for (unsigned i = ctx->cf_depth; i > 0; i--) {
            if (ctx->cf[i - 1].kind == D12_DXBC_CF_SWITCH) {
               fail(ctx, "continue inside switch unsupported");
               return;
            }
            if (ctx->cf[i - 1].kind == D12_DXBC_CF_LOOP)
               break;
         }
         nir_jump(b, nir_jump_continue);
         break;

      case D12_DXBC_OP_BREAKC:
      case D12_DXBC_OP_CONTINUEC: {
         NEED(1);
         if (opcode == D12_DXBC_OP_CONTINUEC) {
            for (unsigned i = ctx->cf_depth; i > 0; i--) {
               if (ctx->cf[i - 1].kind == D12_DXBC_CF_SWITCH) {
                  fail(ctx, "conditional continue inside switch unsupported");
                  return;
               }
               if (ctx->cf[i - 1].kind == D12_DXBC_CF_LOOP)
                  break;
            }
         }
         bool test_zero = !((op_tok >> 18) & 1);
         nir_def *c = nir_channel(b, SRC(0), 0);
         nir_push_if(b, test_zero ? nir_ieq_imm(b, c, 0) : nir_ine_imm(b, c, 0));
         nir_jump(b, opcode == D12_DXBC_OP_BREAKC ? nir_jump_break
                                                  : nir_jump_continue);
         nir_pop_if(b, NULL);
         break;
      }

      case D12_DXBC_OP_DISCARD: {
         NEED(1);
         bool test_zero = !((op_tok >> 18) & 1);
         nir_def *c = nir_channel(b, SRC(0), 0);
         nir_discard_if(b, test_zero ? nir_ieq_imm(b, c, 0)
                                     : nir_ine_imm(b, c, 0));
         b->shader->info.fs.uses_discard = true;
         break;
      }

      /*
       * STRUCTURED BUFFERS. The address is one indirection through the
       * root table -- the same shape as a constant buffer -- then
       * index * stride + offset. The stride comes from the declaration;
       * assuming 4 is right only for a buffer of 32-bit scalars, and a
       * 12-byte struct indexed at element 3 would read byte 12 instead
       * of byte 36.
       */
      case D12_DXBC_OP_LD_STRUCTURED: {
         NEED(4);
         /* Find the resource by FILE, not by position: the two forms
          * differ in operand count and the resource is not always last. */
         unsigned ri = nops - 1;
         for (unsigned k = 1; k < nops; k++) {
            if (o[k].file == D12_DXBC_FILE_UAV ||
                o[k].file == D12_DXBC_FILE_RESOURCE ||
                o[k].file == D12_DXBC_FILE_TGSM) {
               ri = k;
               break;
            }
         }
         uint32_t reg = o[ri].index[0];
         bool uav = o[ri].file == D12_DXBC_FILE_UAV;
         bool tgsm = o[ri].file == D12_DXBC_FILE_TGSM;
         if (sparse && tgsm) {
            fail(ctx, "buffer feedback requires an SRV or UAV operand");
            return;
         }
         unsigned stride =
            tgsm ? (reg < D12_DXBC_MAX_SIG ? ctx->tgsm[reg].stride : 0)
                 : (uav ? ctx->uav_stride : ctx->srv_stride)
                      [reg < D12_DXBC_MAX_SIG ? reg : 0];
         nir_def *addr = tgsm
                            ? tgsm_byte_offset(ctx, &o[ri], SRC(1), SRC(2))
                         : uav
                            ? uav_buffer_address(ctx, &o[ri], SRC(1), SRC(2),
                                                 stride)
                            : srv_buffer_address(ctx, &o[ri], SRC(1), SRC(2),
                                                 stride);
         if (!addr)
            return;
         if (tgsm) {
            nir_def *val =
               tgsm_load_vec4(ctx, &o[ri], addr, o[0].mask);
            unsigned sw[4] = {o[ri].swizzle[0], o[ri].swizzle[1],
                              o[ri].swizzle[2], o[ri].swizzle[3]};
            DST(nir_swizzle(b, val, sw, 4));
            break;
         }
         nir_def *byte = nir_iadd(
            b,
            nir_imul_imm(b, nir_u2u64(b, nir_channel(b, SRC(1), 0)),
                         stride ? stride : 4),
            nir_u2u64(b, nir_channel(b, SRC(2), 0)));
         nir_def *size = buffer_view_size_bytes(ctx, &o[ri]);
         if (ctx->error)
            return;
         nir_def *code = sparse
                            ? buffer_vector_feedback(ctx, &o[ri], &o[0], byte, &addr)
                            : NULL;
         if (ctx->error)
            return;
         nir_def *val = robust_load_buffer_vec4(ctx, addr, byte, size);
         unsigned sw[4] = {o[ri].swizzle[0], o[ri].swizzle[1],
                           o[ri].swizzle[2], o[ri].swizzle[3]};
         DST(nir_swizzle(b, val, sw, 4));
         if (code)
            write_dst(ctx, &feedback_dst, nir_replicate(b, code, 4));
         break;
      }

      case D12_DXBC_OP_LD_RAW: {
         NEED(3);

         unsigned ri = nops - 1;
         for (unsigned k = 1; k < nops; k++) {
            if (o[k].file == D12_DXBC_FILE_UAV ||
                o[k].file == D12_DXBC_FILE_RESOURCE ||
                o[k].file == D12_DXBC_FILE_TGSM) {
               ri = k;
               break;
            }
         }
         if (o[ri].file != D12_DXBC_FILE_UAV &&
             o[ri].file != D12_DXBC_FILE_RESOURCE &&
             o[ri].file != D12_DXBC_FILE_TGSM) {
            fail(ctx, "raw load with no buffer resource operand");
            return;
         }

         nir_def *zero = nir_imm_int(b, 0);
         bool tgsm = o[ri].file == D12_DXBC_FILE_TGSM;
         if (sparse && tgsm) {
            fail(ctx, "buffer feedback requires an SRV or UAV operand");
            return;
         }
         nir_def *addr = tgsm
                            ? tgsm_byte_offset(ctx, &o[ri], zero, SRC(1))
                         : o[ri].file == D12_DXBC_FILE_UAV
                            ? uav_buffer_address(ctx, &o[ri], zero, SRC(1), 0)
                            : srv_buffer_address(ctx, &o[ri], zero, SRC(1), 0);
         if (!addr)
            return;

         if (tgsm) {
            nir_def *val =
               tgsm_load_vec4(ctx, &o[ri], addr, o[0].mask);
            unsigned sw[4] = {o[ri].swizzle[0], o[ri].swizzle[1],
                              o[ri].swizzle[2], o[ri].swizzle[3]};
            DST(nir_swizzle(b, val, sw, 4));
            break;
         }

         nir_def *size = buffer_view_size_bytes(ctx, &o[ri]);
         if (ctx->error)
            return;
         nir_def *code = sparse
                            ? buffer_vector_feedback(ctx, &o[ri], &o[0], SRC(1), &addr)
                            : NULL;
         if (ctx->error)
            return;
         nir_def *val = robust_load_buffer_vec4(ctx, addr, SRC(1), size);
         unsigned sw[4] = {o[ri].swizzle[0], o[ri].swizzle[1],
                           o[ri].swizzle[2], o[ri].swizzle[3]};
         DST(nir_swizzle(b, val, sw, 4));
         if (code)
            write_dst(ctx, &feedback_dst, nir_replicate(b, code, 4));
         break;
      }

      case D12_DXBC_OP_STORE_STRUCTURED: {
         NEED(4);
         uint32_t reg = o[0].index[0];
         if (o[0].file == D12_DXBC_FILE_TGSM) {
            nir_def *byte = tgsm_byte_offset(ctx, &o[0], SRC(1), SRC(2));
            if (!byte)
               return;
            nir_def *val = SRC(3);
            unsigned mask = o[0].mask ? o[0].mask : 0xf;
            nir_store_shared(b, val, byte, .write_mask = mask,
                             .align_mul = 4);
            break;
         }

         unsigned stride =
            ctx->uav_stride[reg < D12_DXBC_MAX_SIG ? reg : 0];
         nir_def *addr =
            uav_buffer_address(ctx, &o[0], SRC(1), SRC(2), stride);
         if (!addr)
            return;
         nir_def *val = SRC(3);
         unsigned mask = o[0].mask ? o[0].mask : 0xf;
         nir_def *byte = nir_iadd(
            b,
            nir_imul_imm(b, nir_u2u64(b, nir_channel(b, SRC(1), 0)),
                         stride ? stride : 4),
            nir_u2u64(b, nir_channel(b, SRC(2), 0)));
         nir_def *size = buffer_view_size_bytes(ctx, &o[0]);
         if (ctx->error)
            return;

         robust_store_buffer_masked(ctx, addr, byte, size, val, mask);
         break;
      }

      /*
       * Loading from a UAV. Mirrors the typed store: element addressing,
       * one indirection through the root table. The resource is found by
       * REGISTER FILE rather than by position -- the operand count varies
       * between the load forms and finding it by file survives that.
       */
      case D12_DXBC_OP_LD_UAV_TYPED: {
         NEED(3);

         unsigned reg = 0;
         bool found = false, uav = true;
         const struct d12_dxbc_operand *resource = NULL;
         for (unsigned k = 1; k < nops; k++) {
            if (o[k].file == D12_DXBC_FILE_UAV ||
                o[k].file == D12_DXBC_FILE_RESOURCE) {
               reg = o[k].index[0];
               uav = o[k].file == D12_DXBC_FILE_UAV;
               resource = &o[k];
               found = true;
               break;
            }
         }
         if (!found) {
            fail(ctx, "UAV load with no resource operand");
            return;
         }
         if (sparse && !uav) {
            fail(ctx, "typed UAV feedback requires a UAV operand");
            return;
         }

         if (uav && reg < D12_DXBC_MAX_SIG &&
             ctx->uav_dim[reg] != D12_DXBC_RESOURCE_DIMENSION_BUFFER) {
            unsigned dim = ctx->uav_dim[reg];
            bool is_1d = dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D ||
                         dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY;
            bool is_array =
               dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY ||
               dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY;
            unsigned return_type = ctx->uav_return_type[reg];
            if ((dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D &&
                 dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D && !is_array) ||
                (return_type != D12_DXBC_RESOURCE_RETURN_UNORM &&
                 return_type != D12_DXBC_RESOURCE_RETURN_SNORM &&
                 return_type != D12_DXBC_RESOURCE_RETURN_FLOAT &&
                 return_type != D12_DXBC_RESOURCE_RETURN_UINT &&
                 return_type != D12_DXBC_RESOURCE_RETURN_SINT) ||
                (resource->num_indices != 1 && resource->num_indices != 2)) {
               fail(ctx, "unsupported texture UAV load");
               return;
            }

            /* Oversized/rooted ranges publish a GPU address into the
             * application's 40-byte descriptor heap in the root table.  It
             * is not a byte offset into the command-local native heap.  The
             * old path truncated that address to 32 bits and added it to
             * D12_UNIFORM_TEXTURES; test_null_descriptor_mismatch_type then
             * faulted at exactly compact_heap + low32(rooted_heap).
             *
             * Resolve the global descriptor index and select the parallel
             * shader-visible AGX texture heap, matching rooted SRV and typed
             * buffer loads.  Ordinary finite tables retain their compact
             * command-local heap.
             */
            nir_def *descriptor = uav_rooted_descriptor(ctx, resource);
            nir_def *handle;
            if (descriptor) {
               nir_def *native_slot =
                  rooted_native_index(ctx, descriptor);
               handle = nir_bindless_image_agx(
                  b, nir_imul_imm(b, native_slot,
                                  ctx->native_layout.texture_stride_B),
                  .desc_set = D12_UNIFORM_HEAP_TEXTURES);
            } else {
               nir_def *offset = uav_texture_offset(ctx, resource);
               if (!offset)
                  return;

               handle = nir_bindless_image_agx(
                  b, offset, .desc_set = resource->num_indices == 2
                     ? D12_UNIFORM_HEAP_TEXTURES : D12_UNIFORM_TEXTURES);
            }
            nir_def *coord = SRC(1);
            nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
            tex->op = nir_texop_txf;
            /* Texture1D resources use height-one 2D AGX descriptors.  Fetch
             * them through the physical dimension so the backend keeps the
             * array layer rather than interpreting it as a 1D coordinate. */
            tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
            tex->is_array = is_array;
            tex->is_sparse = sparse;
            tex->dest_type = return_type == D12_DXBC_RESOURCE_RETURN_UINT
                                ? nir_type_uint32
                             : return_type == D12_DXBC_RESOURCE_RETURN_SINT
                                ? nir_type_int32
                                : nir_type_float32;
            tex->coord_components = is_array ? 3 : 2;
            tex->src[0] = nir_tex_src_for_ssa(
               nir_tex_src_texture_handle, handle);
            nir_def *physical_coord =
               is_1d ? (is_array ? nir_vec3(b, nir_channel(b, coord, 0),
                                            nir_imm_int(b, 0),
                                            nir_channel(b, coord, 1))
                                 : nir_vec2(b, nir_channel(b, coord, 0),
                                            nir_imm_int(b, 0)))
                     : nir_trim_vector(b, coord, tex->coord_components);
            tex->src[1] =
               nir_tex_src_for_ssa(nir_tex_src_coord, physical_coord);
            nir_def_init(&tex->instr, &tex->def, sparse ? 5 : 4, 32);
            nir_builder_instr_insert(b, &tex->instr);

            unsigned rs[4] = {resource->swizzle[0], resource->swizzle[1],
                              resource->swizzle[2], resource->swizzle[3]};
            DST(nir_swizzle(b, &tex->def, rs, 4));
            if (sparse)
               write_dst(ctx, &feedback_dst,
                         nir_replicate(b, nir_channel(b, &tex->def, 4), 4));
            break;
         }

         if (uav) {
            nir_def *coord = nir_channel(b, SRC(1), 0);
            nir_def *descriptor = uav_rooted_descriptor(ctx, resource);
            nir_def *native_slot;
            unsigned desc_set;
            if (descriptor) {
               native_slot = rooted_native_index(ctx, descriptor);
               desc_set = D12_UNIFORM_HEAP_TEXTURES;
            } else {
               native_slot = uav_binding_slot(ctx, resource);
               if (!native_slot)
                  return;
               desc_set = D12_UNIFORM_TEXTURES;
            }

            nir_def *handle = nir_bindless_image_agx(
               b, nir_imul_imm(b, native_slot,
                               ctx->native_layout.texture_stride_B),
               .desc_set = desc_set);
            nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
            tex->op = nir_texop_txf;
            tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
            tex->dest_type =
               ctx->uav_return_type[reg] == D12_DXBC_RESOURCE_RETURN_SINT
                  ? nir_type_int32
                  : ctx->uav_return_type[reg] ==
                       D12_DXBC_RESOURCE_RETURN_FLOAT
                     ? nir_type_float32
                     : nir_type_uint32;
            tex->coord_components = 1;
            tex->src[0] = nir_tex_src_for_ssa(
               nir_tex_src_texture_handle, handle);
            tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
            nir_def_init(&tex->instr, &tex->def, 4, 32);
            nir_builder_instr_insert(b, &tex->instr);

            unsigned rs[4] = {resource->swizzle[0], resource->swizzle[1],
                              resource->swizzle[2], resource->swizzle[3]};
            DST(nir_swizzle(b, &tex->def, rs, 4));
            if (sparse) {
               nir_def *count, *stride;
               buffer_view_info(ctx, resource, &count, &stride);
               nir_def *code = buffer_feedback_code(
                  ctx, resource,
                  nir_imul(b, nir_u2u64(b, coord), nir_u2u64(b, stride)),
                  stride, NULL);
               if (ctx->error || !code)
                  return;
               write_dst(ctx, &feedback_dst, nir_replicate(b, code, 4));
            }
            break;
         }

         nir_def *off = nir_imul_imm(b, nir_channel(b, SRC(1), 0), 4);
         nir_def *addr =
            uav ? uav_buffer_address(ctx, resource, nir_imm_int(b, 0), off, 0)
                : srv_buffer_address(ctx, resource, nir_imm_int(b, 0), off, 0);
         if (!addr)
            return;
         nir_def *val = nir_load_global(b, 4, 32, addr, .align_mul = 4);

         /* The resource operand's swizzle reorders the result. */
         unsigned rs[4] = {0, 1, 2, 3};
         for (unsigned k = 1; k < nops; k++) {
            if (o[k].file == D12_DXBC_FILE_UAV ||
                o[k].file == D12_DXBC_FILE_RESOURCE) {
               for (unsigned c = 0; c < 4; c++)
                  rs[c] = o[k].swizzle[c];
               break;
            }
         }
         DST(nir_swizzle(b, val, rs, 4));
         break;
      }

      case D12_DXBC_OP_STORE_RAW:
      case D12_DXBC_OP_STORE_UAV_TYPED: {
         NEED(3);
         bool typed = opcode == D12_DXBC_OP_STORE_UAV_TYPED;
         uint32_t reg = o[0].index[0];
         bool tgsm = o[0].file == D12_DXBC_FILE_TGSM;

         if (tgsm) {
            if (typed) {
               fail(ctx, "typed thread-group shared-memory store");
               return;
            }
            nir_def *byte = tgsm_byte_offset(
               ctx, &o[0], nir_imm_int(b, 0), SRC(1));
            if (!byte)
               return;
            nir_def *val = SRC(2);
            unsigned mask = o[0].mask ? o[0].mask : 0xf;
            nir_store_shared(b, val, byte, .write_mask = mask,
                             .align_mul = 4);
            break;
         }

         if (typed && reg < D12_DXBC_MAX_SIG &&
             ctx->uav_dim[reg] != D12_DXBC_RESOURCE_DIMENSION_BUFFER) {
            bool is_array;
            enum glsl_sampler_dim image_dim;
            nir_def *handle =
               uav_image_handle(ctx, &o[0], &image_dim, &is_array);
            if (!handle)
               return;

            unsigned mask = o[0].mask ? o[0].mask : 0xf;
            if (mask != 0xf) {
               fail(ctx, "sparse texture UAV store mask unsupported");
               return;
            }

            nir_bindless_image_store(
               b, handle,
               uav_image_coord(ctx, SRC(1), image_dim, is_array),
               nir_imm_int(b, 0), SRC(2), nir_imm_int(b, 0),
               .image_dim = image_dim,
               .image_array = is_array, .format = PIPE_FORMAT_NONE,
               .access = ACCESS_NON_READABLE);
            break;
         }

         if (typed && reg >= D12_DXBC_MAX_SIG) {
            fail(ctx, "texture UAV register out of range");
            return;
         }

         nir_def *element_or_byte = nir_channel(b, SRC(1), 0);
         if (typed) {
            if (!d12_native_pbe_software_addressing_supported(
                   ctx->descriptor_generation)) {
               fail(ctx, "typed-buffer stores require unproven AGX3 PBE "
                         "software addressing");
               return;
            }

            nir_def *count = buffer_view_count_or_bytes(ctx, &o[0]);
            if (ctx->error)
               return;

            nir_def *descriptor = uav_rooted_descriptor(ctx, &o[0]);
            nir_def *handle;
            if (descriptor) {
               nir_def *native_slot = rooted_native_index(ctx, descriptor);
               handle = nir_bindless_image_agx(
                  b, nir_imul_imm(b, native_slot,
                                  ctx->native_layout.pbe_stride_B),
                  .desc_set = D12_UNIFORM_HEAP_PBES);
            } else {
               nir_def *slot = uav_binding_slot(ctx, &o[0]);
               if (!slot)
                  return;
               handle = nir_bindless_image_agx(
                  b, nir_imul_imm(b, slot,
                                  ctx->native_layout.pbe_stride_B),
                  .desc_set = D12_UNIFORM_PBES);
            }

            nir_def *zero = nir_imm_int(b, 0);
            nir_push_if(b, nir_ult(b, element_or_byte, count));
            nir_bindless_image_store(
               b, handle,
               nir_vec4(b, element_or_byte, zero, zero, zero), zero, SRC(2),
               zero, .image_dim = GLSL_SAMPLER_DIM_BUF,
               .image_array = false, .format = PIPE_FORMAT_NONE,
               .access = ACCESS_NON_READABLE);
            nir_pop_if(b, NULL);
            break;
         }

         nir_def *addr = uav_buffer_address(
            ctx, &o[0], nir_imm_int(b, 0), element_or_byte, 0);
         if (!addr)
            return;

         nir_def *size = buffer_view_size_bytes(ctx, &o[0]);
         if (ctx->error)
            return;
         robust_store_buffer_masked(
            ctx, addr, element_or_byte, size, SRC(2),
            o[0].mask ? o[0].mask : 0xf);
         break;
      }

      case D12_DXBC_OP_ATOMIC_AND:
      case D12_DXBC_OP_ATOMIC_OR:
      case D12_DXBC_OP_ATOMIC_XOR:
      case D12_DXBC_OP_ATOMIC_CMP_STORE:
      case D12_DXBC_OP_ATOMIC_IADD:
      case D12_DXBC_OP_ATOMIC_IMAX:
      case D12_DXBC_OP_ATOMIC_IMIN:
      case D12_DXBC_OP_ATOMIC_UMAX:
      case D12_DXBC_OP_ATOMIC_UMIN: {
         bool compare = opcode == D12_DXBC_OP_ATOMIC_CMP_STORE;
         NEED(compare ? 4 : 3);

         nir_atomic_op op;
         switch (opcode) {
         case D12_DXBC_OP_ATOMIC_AND:  op = nir_atomic_op_iand; break;
         case D12_DXBC_OP_ATOMIC_OR:   op = nir_atomic_op_ior;  break;
         case D12_DXBC_OP_ATOMIC_XOR:  op = nir_atomic_op_ixor; break;
         case D12_DXBC_OP_ATOMIC_IADD: op = nir_atomic_op_iadd; break;
         case D12_DXBC_OP_ATOMIC_IMAX: op = nir_atomic_op_imax; break;
         case D12_DXBC_OP_ATOMIC_IMIN: op = nir_atomic_op_imin; break;
         case D12_DXBC_OP_ATOMIC_UMAX: op = nir_atomic_op_umax; break;
         case D12_DXBC_OP_ATOMIC_UMIN: op = nir_atomic_op_umin; break;
         default:                      op = nir_atomic_op_cmpxchg; break;
         }

         unsigned reg = o[0].index[0];
         nir_def *typed_descriptor =
            o[0].file == D12_DXBC_FILE_UAV &&
                  reg < D12_DXBC_MAX_SIG &&
                  ctx->uav_dim[reg] == D12_DXBC_RESOURCE_DIMENSION_BUFFER &&
                  !ctx->uav_raw[reg] && ctx->uav_stride[reg] == 0
               ? uav_rooted_descriptor(ctx, &o[0])
               : NULL;
         bool typed_buffer = typed_descriptor != NULL;
         bool image = typed_buffer ||
                      (o[0].file == D12_DXBC_FILE_UAV &&
                       reg < D12_DXBC_MAX_SIG &&
                       ctx->uav_dim[reg] !=
                          D12_DXBC_RESOURCE_DIMENSION_BUFFER);
         if (image) {
            if (!d12_native_pbe_software_addressing_supported(
                   ctx->descriptor_generation)) {
               fail(ctx, "image atomics require unproven AGX3 PBE software "
                         "addressing");
               return;
            }

            bool is_array;
            enum glsl_sampler_dim image_dim;
            nir_def *handle;
            if (typed_buffer) {
               image_dim = GLSL_SAMPLER_DIM_BUF;
               is_array = false;
               handle = uav_rooted_typed_atomic_handle(
                  ctx, typed_descriptor, SRC(1));
            } else {
               handle =
                  uav_image_handle(ctx, &o[0], &image_dim, &is_array);
            }
            if (!handle)
               return;

            nir_def *coord = typed_buffer
                                ? nir_vec4(b, nir_channel(b, SRC(1), 0),
                                           nir_imm_int(b, 0),
                                           nir_imm_int(b, 0),
                                           nir_imm_int(b, 0))
                                : uav_image_coord(ctx, SRC(1), image_dim,
                                                  is_array);
            enum pipe_format format =
               op == nir_atomic_op_imin || op == nir_atomic_op_imax
                  ? PIPE_FORMAT_R32_SINT
                  : PIPE_FORMAT_R32_UINT;
            if (compare) {
               nir_bindless_image_atomic_swap(
                  b, 32, handle, coord, nir_imm_int(b, 0),
                  nir_channel(b, SRC(2), 0), nir_channel(b, SRC(3), 0),
                  .atomic_op = op, .image_dim = image_dim,
                  .image_array = is_array, .format = format, .access = 0);
            } else {
               nir_bindless_image_atomic(
                  b, 32, handle, coord, nir_imm_int(b, 0),
                  nir_channel(b, SRC(2), 0), .atomic_op = op,
                  .image_dim = image_dim,
                  .image_array = is_array, .format = format, .access = 0);
            }
            break;
         }

         bool tgsm = o[0].file == D12_DXBC_FILE_TGSM;
         nir_def *addr = tgsm ? tgsm_atomic_offset(ctx, &o[0], SRC(1))
                              : atomic_buffer_address(ctx, &o[0], SRC(1));
         if (!addr)
            return;

         if (tgsm && compare) {
            nir_shared_atomic_swap(b, 32, addr,
                                   nir_channel(b, SRC(2), 0),
                                   nir_channel(b, SRC(3), 0),
                                   .atomic_op = op);
         } else if (tgsm) {
            nir_shared_atomic(b, 32, addr, nir_channel(b, SRC(2), 0),
                              .atomic_op = op);
         } else if (compare) {
            nir_global_atomic_swap(b, 32, addr,
                                   nir_channel(b, SRC(2), 0),
                                   nir_channel(b, SRC(3), 0),
                                   .atomic_op = op);
         } else {
            nir_global_atomic(b, 32, addr, nir_channel(b, SRC(2), 0),
                              .atomic_op = op);
         }
         break;
      }

      case D12_DXBC_OP_IMM_ATOMIC_IADD:
      case D12_DXBC_OP_IMM_ATOMIC_AND:
      case D12_DXBC_OP_IMM_ATOMIC_OR:
      case D12_DXBC_OP_IMM_ATOMIC_XOR:
      case D12_DXBC_OP_IMM_ATOMIC_EXCH:
      case D12_DXBC_OP_IMM_ATOMIC_CMP_EXCH:
      case D12_DXBC_OP_IMM_ATOMIC_IMAX:
      case D12_DXBC_OP_IMM_ATOMIC_IMIN:
      case D12_DXBC_OP_IMM_ATOMIC_UMAX:
      case D12_DXBC_OP_IMM_ATOMIC_UMIN: {
         bool compare = opcode == D12_DXBC_OP_IMM_ATOMIC_CMP_EXCH;
         NEED(compare ? 5 : 4);

         nir_atomic_op op;
         switch (opcode) {
         case D12_DXBC_OP_IMM_ATOMIC_IADD: op = nir_atomic_op_iadd; break;
         case D12_DXBC_OP_IMM_ATOMIC_AND:  op = nir_atomic_op_iand; break;
         case D12_DXBC_OP_IMM_ATOMIC_OR:   op = nir_atomic_op_ior;  break;
         case D12_DXBC_OP_IMM_ATOMIC_XOR:  op = nir_atomic_op_ixor; break;
         case D12_DXBC_OP_IMM_ATOMIC_EXCH: op = nir_atomic_op_xchg; break;
         case D12_DXBC_OP_IMM_ATOMIC_IMAX: op = nir_atomic_op_imax; break;
         case D12_DXBC_OP_IMM_ATOMIC_IMIN: op = nir_atomic_op_imin; break;
         case D12_DXBC_OP_IMM_ATOMIC_UMAX: op = nir_atomic_op_umax; break;
         case D12_DXBC_OP_IMM_ATOMIC_UMIN: op = nir_atomic_op_umin; break;
         default:                          op = nir_atomic_op_cmpxchg; break;
         }

         nir_def *previous;
         unsigned reg = o[1].index[0];
         nir_def *typed_descriptor =
            o[1].file == D12_DXBC_FILE_UAV &&
                  reg < D12_DXBC_MAX_SIG &&
                  ctx->uav_dim[reg] == D12_DXBC_RESOURCE_DIMENSION_BUFFER &&
                  !ctx->uav_raw[reg] && ctx->uav_stride[reg] == 0
               ? uav_rooted_descriptor(ctx, &o[1])
               : NULL;
         bool typed_buffer = typed_descriptor != NULL;
         bool image = typed_buffer ||
                      (o[1].file == D12_DXBC_FILE_UAV &&
                       reg < D12_DXBC_MAX_SIG &&
                       ctx->uav_dim[reg] !=
                          D12_DXBC_RESOURCE_DIMENSION_BUFFER);
         if (image) {
            if (!d12_native_pbe_software_addressing_supported(
                   ctx->descriptor_generation)) {
               fail(ctx, "image atomics require unproven AGX3 PBE software "
                         "addressing");
               return;
            }

            bool is_array;
            enum glsl_sampler_dim image_dim;
            nir_def *handle;
            if (typed_buffer) {
               image_dim = GLSL_SAMPLER_DIM_BUF;
               is_array = false;
               handle = uav_rooted_typed_atomic_handle(
                  ctx, typed_descriptor, SRC(2));
            } else {
               handle =
                  uav_image_handle(ctx, &o[1], &image_dim, &is_array);
            }
            if (!handle)
               return;

            nir_def *coord = typed_buffer
                                ? nir_vec4(b, nir_channel(b, SRC(2), 0),
                                           nir_imm_int(b, 0),
                                           nir_imm_int(b, 0),
                                           nir_imm_int(b, 0))
                                : uav_image_coord(ctx, SRC(2), image_dim,
                                                  is_array);
            enum pipe_format format =
               op == nir_atomic_op_imin || op == nir_atomic_op_imax
                  ? PIPE_FORMAT_R32_SINT
                  : PIPE_FORMAT_R32_UINT;
            if (compare) {
               previous = nir_bindless_image_atomic_swap(
                  b, 32, handle, coord, nir_imm_int(b, 0),
                  nir_channel(b, SRC(3), 0), nir_channel(b, SRC(4), 0),
                  .atomic_op = op, .image_dim = image_dim,
                  .image_array = is_array, .format = format, .access = 0);
            } else {
               previous = nir_bindless_image_atomic(
                  b, 32, handle, coord, nir_imm_int(b, 0),
                  nir_channel(b, SRC(3), 0), .atomic_op = op,
                  .image_dim = image_dim,
                  .image_array = is_array, .format = format, .access = 0);
            }
         } else {
            bool tgsm = o[1].file == D12_DXBC_FILE_TGSM;
            nir_def *addr = tgsm
                               ? tgsm_atomic_offset(ctx, &o[1], SRC(2))
                               : atomic_buffer_address(ctx, &o[1], SRC(2));
            if (!addr)
               return;

            if (tgsm && compare) {
               previous = nir_shared_atomic_swap(
                  b, 32, addr, nir_channel(b, SRC(3), 0),
                  nir_channel(b, SRC(4), 0), .atomic_op = op);
            } else if (tgsm) {
               previous = nir_shared_atomic(
                  b, 32, addr, nir_channel(b, SRC(3), 0), .atomic_op = op);
            } else if (compare) {
               previous = nir_global_atomic_swap(
                  b, 32, addr, nir_channel(b, SRC(3), 0),
                  nir_channel(b, SRC(4), 0), .atomic_op = op);
            } else {
               previous = nir_global_atomic(b, 32, addr,
                                            nir_channel(b, SRC(3), 0),
                                            .atomic_op = op);
            }
         }

         /* DXBC temporaries are vec4 even when the destination mask names
          * only .x. Preserve the scalar atomic result in that component
          * without constructing an ill-typed scalar store to a vec4 var.
          */
         DST(nir_replicate(b, previous, 4));
         break;
      }

      case D12_DXBC_OP_IMM_ATOMIC_ALLOC:
      case D12_DXBC_OP_IMM_ATOMIC_CONSUME: {
         /* Both instructions are {dest, UAV}.  The hardware atomic returns
          * the pre-update value.  D3D allocation exposes that value, while
          * consumption exposes the value after subtracting one.
          */
         NEED(2);

         nir_def *addr = uav_counter_address(ctx, &o[1]);
         if (!addr)
            return;

         bool consume = opcode == D12_DXBC_OP_IMM_ATOMIC_CONSUME;
         nir_def *previous =
            nir_global_atomic(b, 32, addr,
                              nir_imm_int(b, consume ? -1 : 1),
                              .atomic_op = nir_atomic_op_iadd);
         nir_def *result = consume ? nir_iadd_imm(b, previous, -1) : previous;
         DST(nir_replicate(b, result, 4));
         break;
      }

      case D12_DXBC_OP_SYNC: {
         /* Public opcode-token bits 11..14 independently request group
          * execution rendezvous, TGSM ordering, group-scoped UAV ordering,
          * and device-scoped UAV ordering. Preserve both scope axes instead
          * of treating every memory barrier as a full device barrier. */
         unsigned flags = op_tok & 0x7800;
         bool threads = flags & 0x0800;
         bool shared = flags & 0x1000;
         bool uav_group = flags & 0x2000;
         bool uav_global = flags & 0x4000;
         nir_variable_mode modes = 0;
         if (shared)
            modes |= nir_var_mem_shared;
         if (uav_group || uav_global)
            modes |= nir_var_mem_global | nir_var_image;
         if (!threads && !modes) {
            fail(ctx, "sync instruction has no synchronization flags");
            return;
         }

         nir_barrier(
            b, .execution_scope = threads ? SCOPE_WORKGROUP : SCOPE_NONE,
            .memory_scope = modes ? (uav_global ? SCOPE_DEVICE
                                                : SCOPE_WORKGROUP)
                                  : SCOPE_NONE,
            .memory_semantics = modes ? NIR_MEMORY_ACQ_REL : 0,
            .memory_modes = modes);
         break;
      }

      case D12_DXBC_OP_SAMPLE_INFO: {
         /* sample_info_{float,uint} dst, rasterizer.  For the rasterizer
          * pseudo-resource, x is the active render-target sample count and
          * the source swizzle replicates it for GetRenderTargetSampleCount.
          * The hardware publishes log2(samples), so recover 1/2/4 here. */
         NEED(2);
         nir_def *count;
         if (o[1].file == D12_DXBC_FILE_RASTERIZER) {
            count = nir_ishl(b, nir_imm_int(b, 1),
                             nir_u2u32(b, nir_load_samples_log2_agx(b)));
         } else if (o[1].file == D12_DXBC_FILE_RESOURCE) {
            unsigned reg = o[1].index[0];
            unsigned dim = reg < D12_DXBC_MAX_SIG ? ctx->srv_dim[reg] : 0;
            bool array =
               dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY;
            if (dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS &&
                !array) {
               fail(ctx, "sample_info requires a multisample texture");
               return;
            }

            nir_def *descriptor = srv_rooted_descriptor(ctx, &o[1]);
            nir_def *handle;
            if (descriptor) {
               nir_def *index = rooted_native_index(ctx, descriptor);
               handle = nir_bindless_image_agx(
                  b, nir_imul_imm(b, index,
                                  ctx->native_layout.texture_stride_B),
                  .desc_set = D12_UNIFORM_HEAP_TEXTURES);
            } else {
               unsigned slot =
                  ctx->map ? d12_binding_map_find(ctx->map, D12_RES_SRV, 0,
                                                  reg)
                           : reg;
               if (slot == ~0u) {
                  fail(ctx,
                       "sample_info resource is not in the root signature");
                  return;
               }
               handle = nir_bindless_image_agx(
                  b, nir_imm_int(b,
                                 slot * ctx->native_layout.texture_stride_B),
                  .desc_set = D12_UNIFORM_TEXTURES);
            }

            nir_tex_instr *query = nir_tex_instr_create(b->shader, 1);
            query->op = nir_texop_texture_samples;
            query->sampler_dim = GLSL_SAMPLER_DIM_MS;
            query->is_array = array;
            query->dest_type = nir_type_uint32;
            query->src[0] = nir_tex_src_for_ssa(
               nir_tex_src_texture_handle, handle);
            nir_def_init(&query->instr, &query->def, 1, 32);
            nir_builder_instr_insert(b, &query->instr);
            count = &query->def;
         } else {
            fail(ctx, "sample_info on unsupported operand");
            return;
         }

         const bool return_uint = (op_tok >> 11) & 1;
         nir_def *value = return_uint ? count : nir_u2f32(b, count);
         DST(nir_replicate(b, value, 4));
         break;
      }

      case D12_DXBC_OP_SAMPLE_POS: {
         /* sample_pos dst.xy, Texture2DMS, sample_index. D3D12AGX exposes
          * only the hardware's standard 1/2/4-sample pattern, whose packed
          * 4x value is 0xeaa26e26 in render commands. Decode those nibbles
          * to D3D's centre-relative coordinates. */
         NEED(3);
         if (o[1].file != D12_DXBC_FILE_RESOURCE) {
            fail(ctx, "sample_pos on non-resource operand");
            return;
         }

         unsigned tslot =
            ctx->map ? d12_binding_map_find(ctx->map, D12_RES_SRV, 0,
                                            o[1].index[0])
                     : o[1].index[0];
         if (tslot == ~0u) {
            fail(ctx, "sample_pos resource is not in the root signature");
            return;
         }

         nir_def *tex_handle = nir_bindless_image_agx(
            b, nir_imm_int(b,
                           tslot * ctx->native_layout.texture_stride_B),
            .desc_set = D12_UNIFORM_TEXTURES);
         nir_tex_instr *query = nir_tex_instr_create(b->shader, 1);
         query->op = nir_texop_texture_samples;
         query->sampler_dim = GLSL_SAMPLER_DIM_2D;
         query->dest_type = nir_type_uint32;
         query->src[0] =
            nir_tex_src_for_ssa(nir_tex_src_texture_handle, tex_handle);
         nir_def_init(&query->instr, &query->def, 1, 32);
         nir_builder_instr_insert(b, &query->instr);

         nir_def *count = &query->def;
         nir_def *index = nir_channel(b, SRC(2), 0);
         nir_def *x = nir_imm_float(b, -0.125f);
         nir_def *y = nir_imm_float(b, -0.375f);
         x = nir_bcsel(b, nir_ieq_imm(b, index, 1),
                       nir_imm_float(b, 0.375f), x);
         y = nir_bcsel(b, nir_ieq_imm(b, index, 1),
                       nir_imm_float(b, -0.125f), y);
         x = nir_bcsel(b, nir_ieq_imm(b, index, 2),
                       nir_imm_float(b, -0.375f), x);
         y = nir_bcsel(b, nir_ieq_imm(b, index, 2),
                       nir_imm_float(b, 0.125f), y);
         x = nir_bcsel(b, nir_ieq_imm(b, index, 3),
                       nir_imm_float(b, 0.125f), x);
         y = nir_bcsel(b, nir_ieq_imm(b, index, 3),
                       nir_imm_float(b, 0.375f), y);
         nir_def *two_x = nir_bcsel(b, nir_ieq_imm(b, index, 1),
                                    nir_imm_float(b, -0.25f),
                                    nir_imm_float(b, 0.25f));
         nir_def *two_y = two_x;
         x = nir_bcsel(b, nir_ieq_imm(b, count, 2), two_x, x);
         y = nir_bcsel(b, nir_ieq_imm(b, count, 2), two_y, y);
         x = nir_bcsel(b, nir_ieq_imm(b, count, 1),
                       nir_imm_float(b, 0.0f), x);
         y = nir_bcsel(b, nir_ieq_imm(b, count, 1),
                       nir_imm_float(b, 0.0f), y);
         DST(nir_vec4(b, x, y, nir_imm_float(b, 0.0f),
                      nir_imm_float(b, 0.0f)));
         break;
      }

      /*
       * Geometry shader emission. The outputs a shader has stored so far
       * ARE the vertex, so emitting is a boundary rather than a value --
       * the same shape the DXIL front end gives emitStream.
       */
      case D12_DXBC_OP_EMIT:
      case D12_DXBC_OP_EMITTHENCUT:
      case D12_DXBC_OP_CUT:
      case D12_DXBC_OP_EMIT_STREAM:
      case D12_DXBC_OP_CUT_STREAM:
      case D12_DXBC_OP_EMITTHENCUT_STREAM: {
         unsigned stream = 0;
         bool streamed = opcode == D12_DXBC_OP_EMIT_STREAM ||
                         opcode == D12_DXBC_OP_CUT_STREAM ||
                         opcode == D12_DXBC_OP_EMITTHENCUT_STREAM;
         if (streamed && nops >= 1)
            stream = o[0].index[0];

         bool cut = opcode == D12_DXBC_OP_CUT ||
                    opcode == D12_DXBC_OP_CUT_STREAM ||
                    opcode == D12_DXBC_OP_EMITTHENCUT ||
                    opcode == D12_DXBC_OP_EMITTHENCUT_STREAM;
         bool emit = opcode != D12_DXBC_OP_CUT &&
                     opcode != D12_DXBC_OP_CUT_STREAM;

         if (emit)
            nir_emit_vertex(b, .stream_id = stream);
         if (cut)
            nir_end_primitive(b, .stream_id = stream);
         break;
      }

      /*
       * TEXTURE ACCESS. t# and s# register numbers map to binding-map
       * slots the same way every other resource here does, so the tex
       * instruction is built with those as texture_index/sampler_index
       * -- the shape the DXIL front end already uses for dx.op.sample.
       *
       * DXBC hands the coordinate as a float4 and lets the swizzle pick;
       * a 2D sample takes .xy, and ld's address carries the mip level in
       * .w rather than as a separate operand.
       */
      case D12_DXBC_OP_SAMPLE:
      case D12_DXBC_OP_SAMPLE_C:
      case D12_DXBC_OP_SAMPLE_C_LZ:
      case D12_DXBC_OP_SAMPLE_L:
      case D12_DXBC_OP_SAMPLE_D:
      case D12_DXBC_OP_SAMPLE_B:
      case D12_DXBC_OP_GATHER4:
      case D12_DXBC_OP_GATHER4_C:
      case D12_DXBC_OP_GATHER4_PO:
      case D12_DXBC_OP_GATHER4_PO_C:
      case D12_DXBC_OP_LD:
      case D12_DXBC_OP_LD_MS: {
         bool is_ld_ms = opcode == D12_DXBC_OP_LD_MS;
         bool is_ld = opcode == D12_DXBC_OP_LD || is_ld_ms;
         bool gather = opcode == D12_DXBC_OP_GATHER4 ||
                       opcode == D12_DXBC_OP_GATHER4_C ||
                       opcode == D12_DXBC_OP_GATHER4_PO ||
                       opcode == D12_DXBC_OP_GATHER4_PO_C;
         bool dynamic_offset = opcode == D12_DXBC_OP_GATHER4_PO ||
                               opcode == D12_DXBC_OP_GATHER4_PO_C;
         bool compare = opcode == D12_DXBC_OP_SAMPLE_C ||
                        opcode == D12_DXBC_OP_SAMPLE_C_LZ ||
                        opcode == D12_DXBC_OP_GATHER4_C ||
                        opcode == D12_DXBC_OP_GATHER4_PO_C;
         bool has_lod = opcode == D12_DXBC_OP_SAMPLE_L ||
                        opcode == D12_DXBC_OP_SAMPLE_C_LZ;
         bool has_grad = opcode == D12_DXBC_OP_SAMPLE_D;
         bool has_bias = opcode == D12_DXBC_OP_SAMPLE_B;
         NEED(is_ld_ms ? 4 : is_ld ? 3
                                   : 4 + dynamic_offset + compare +
                                        (has_grad ? 2 : has_bias));

         /* The resource is the operand in the RESOURCE file; the sampler
          * the one in the SAMPLER file. Finding them by file rather than
          * by position survives the operand-count differences between
          * these three. */
         unsigned tex_reg = 0;
         bool have_tex = false, have_smp = false;
         const struct d12_dxbc_operand *tex_operand = NULL;
         const struct d12_dxbc_operand *smp_operand = NULL;
         for (unsigned k = 1; k < nops; k++) {
            if (o[k].file == D12_DXBC_FILE_RESOURCE && !have_tex) {
               tex_reg = o[k].index[0];
               tex_operand = &o[k];
               have_tex = true;
            } else if (o[k].file == D12_DXBC_FILE_SAMPLER && !have_smp) {
               smp_operand = &o[k];
               have_smp = true;
            }
         }

         if (!have_tex) {
            fail(ctx, "texture op with no resource operand");
            return;
         }

         bool is_buffer =
            is_ld && !is_ld_ms && tex_reg < D12_DXBC_MAX_SIG &&
            ctx->srv_dim[tex_reg] == D12_DXBC_RESOURCE_DIMENSION_BUFFER;
         unsigned resource_dim = tex_reg < D12_DXBC_MAX_SIG
                                    ? ctx->srv_dim[tex_reg]
                                    : D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D;
         bool is_1d =
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D ||
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY;
         bool is_array =
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY ||
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY ||
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY ||
            resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY;
         enum glsl_sampler_dim sampler_dim =
            is_1d ? GLSL_SAMPLER_DIM_1D
            : resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE ||
                  resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_CUBE_ARRAY
               ? GLSL_SAMPLER_DIM_CUBE
            : resource_dim == D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS ||
                  resource_dim ==
                     D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY
               ? GLSL_SAMPLER_DIM_MS
               : GLSL_SAMPLER_DIM_2D;
         unsigned coord_components = is_1d ? (is_array ? 2 : 1)
                                     : sampler_dim == GLSL_SAMPLER_DIM_CUBE
                                        ? (is_array ? 4 : 3)
                                     : is_array ? 3
                                                : 2;
         if (is_ld_ms &&
             (tex_reg >= D12_DXBC_MAX_SIG ||
              (resource_dim != D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS &&
               resource_dim !=
                  D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_MS_ARRAY))) {
            fail(ctx, "LD_MS requires a Texture2DMS resource");
            return;
         }

         nir_def *rooted_texture =
            srv_rooted_descriptor(ctx, tex_operand);
         nir_def *texture_slot = NULL;
         if (!rooted_texture) {
            texture_slot = srv_binding_slot(ctx, tex_operand);
            if (!texture_slot) {
               if (d12_debug()->validate_bindings) {
                  mesa_logi("DXBC texture operand: reg=%u indices=%u "
                            "idx0=%u idx1=%u rel0=%u rel1=%u "
                            "range-valid=%u lower=%u upper=%u space=%u",
                            tex_reg, tex_operand->num_indices,
                            tex_operand->index[0], tex_operand->index[1],
                            tex_operand->index_is_relative[0],
                            tex_operand->index_is_relative[1],
                            tex_reg < D12_DXBC_MAX_SIG
                               ? ctx->srv_range[tex_reg].valid : 0,
                            tex_reg < D12_DXBC_MAX_SIG
                               ? ctx->srv_range[tex_reg].lower_bound : 0,
                            tex_reg < D12_DXBC_MAX_SIG
                               ? ctx->srv_range[tex_reg].upper_bound : 0,
                            tex_reg < D12_DXBC_MAX_SIG
                               ? ctx->srv_range[tex_reg].space : 0);
               }
               fail(ctx, "texture is not bound by the root signature");
               return;
            }
         }
         unsigned sslot = have_smp
                             ? finite_binding_slot(ctx, D12_RES_SAMPLER,
                                                   smp_operand,
                                                   ctx->sampler_range)
                             : ~0u;
         nir_def *rooted_sampler =
            have_smp ? sampler_rooted_index(ctx, smp_operand) : NULL;
         if (ctx->error)
            return;
         if (!is_ld && sslot == ~0u && !rooted_sampler) {
            fail(ctx, "sampler is not bound by the root signature");
            return;
         }

         nir_def *coord4 = SRC(1);
         nir_def *buffer_coord =
            is_buffer ? nir_channel(b, coord4, 0) : NULL;

         /*
          * AGX reaches a texture ONLY through a bindless handle -- a byte
          * offset into the texture heap. Setting texture_index and hoping
          * asserts in agx_nir_lower_texture ("must be bindless"), which
          * is what three tests turned into crashes before this. Samplers
          * are NOT the same mechanism: the backend's sampler operand is a
          * 16-bit index into sampler STATE REGISTERS, so a constant slot
          * is simply sampler_index. Same split the DXIL path makes.
          */
         nir_def *tex_handle;
         if (rooted_texture) {
            nir_def *index = rooted_native_index(ctx, rooted_texture);
            tex_handle = nir_bindless_image_agx(
               b, nir_imul_imm(b, index,
                               ctx->native_layout.texture_stride_B),
               .desc_set = D12_UNIFORM_HEAP_TEXTURES);
         } else {
            nir_def *native_slot = texture_slot;
            nir_def *native_offset =
               is_buffer ? nir_imul_imm(
                              b, native_slot,
                              ctx->native_layout.texture_stride_B)
                         : srv_texture_offset(ctx, tex_operand);
            if (!native_offset)
               return;
            tex_handle = nir_bindless_image_agx(
               b, native_offset, .desc_set = !is_buffer && tex_operand->num_indices == 2
                  ? D12_UNIFORM_HEAP_TEXTURES : D12_UNIFORM_TEXTURES);
         }

         /* Public Extended Opcode SAMPLE_CONTROLS packs signed 4-bit U/V/W
          * immediate texel offsets at bits 9, 13 and 17.  They apply to LD
          * as well as sampling opcodes.  The AGX texture lowering consumes
          * the corresponding NIR offset source directly. */
         int texel_offset[3] = {0};
         bool has_offset = false;
         for (unsigned k = 1; k < opcode_words; k++) {
            uint32_t ext = ctx->tok[k];
            if ((ext & 0x3f) != 1)
               continue;

            for (unsigned c = 0; c < 3; c++) {
               int v = (ext >> (9 + 4 * c)) & 0xf;
               texel_offset[c] = (v & 8) ? v - 16 : v;
               has_offset |= texel_offset[c] != 0;
            }
         }

         if (is_buffer && has_offset) {
            fail(ctx, "buffer texture load has a texel offset");
            return;
         }

         /* ResourceMinLODClamp belongs to the SRV, not the sampler. Finite
          * root-table views publish its float bits in the otherwise buffer-
          * only BUFINFO plane; rooted heap views retain it in the descriptor
          * metadata word at the buffer-stride offset. */
         nir_def *view_min_lod = NULL;
         nir_def *view_txf_min_lod = NULL;
         nir_def *view_min_lod_oob = NULL;
         if (!is_buffer && !is_ld_ms) {
            if (rooted_texture) {
               nir_def *packed = nir_load_global(
                  b, 1, 32,
                  nir_iadd_imm(b, rooted_texture,
                               D12_DESC_TEXTURE_METADATA0_OFFSET),
                  .align_mul = 4);
               view_min_lod = nir_load_global(
                  b, 1, 32,
                  nir_iadd_imm(b, rooted_texture,
                               D12_DESC_TEXTURE_METADATA1_OFFSET),
                  .align_mul = 4);
               view_txf_min_lod = nir_u2f32(
                  b, nir_iand_imm(b, nir_ushr_imm(b, packed, 22), 0x1f));
               view_min_lod_oob =
                  nir_ine_imm(b, nir_iand_imm(b, packed, 1u << 27), 0);
            } else {
               nir_def *bufinfo = root_plane(ctx, D12_ROOT_BUFINFO_BASE);
               nir_def *info = nir_load_global(
                  b, 2, 32,
                  nir_iadd(b, bufinfo,
                           nir_imul_imm(b, nir_u2u64(b, texture_slot),
                                        sizeof(uint64_t))),
                  .align_mul = 8);
               view_min_lod = nir_channel(b, info, 0);
               nir_def *packed = nir_channel(b, info, 1);
               view_txf_min_lod =
                  nir_u2f32(b, nir_iand_imm(b, packed, 0x1f));
               view_min_lod_oob =
                  nir_ine_imm(b, nir_iand_imm(b, packed, 0x80000000u), 0);
            }
         }

         if (instruction_min_lod)
            view_min_lod = view_min_lod
                              ? nir_fmax(b, view_min_lod, instruction_min_lod)
                              : instruction_min_lod;

         bool implicit_min_lod = view_min_lod && !is_ld && !has_lod &&
                                 !has_grad && !has_bias && !gather;

         unsigned nsrc = 2 + ((is_ld && !is_buffer && !is_ld_ms) ? 1 : 0) +
                         (is_ld_ms ? 1 : 0) +
                         (has_lod ? 1 : 0) + (compare ? 1 : 0) +
                         (has_grad ? 2 : 0) + (has_bias ? 1 : 0) +
                         (view_min_lod && has_lod && !compare ? 1 : 0) +
                         (view_min_lod && (has_grad || has_bias) ? 1 : 0) +
                         (dynamic_offset ? 1 : 0) +
                         (has_offset ? 1 : 0) +
                         (rooted_sampler ? 1 : 0) +
                         (implicit_min_lod
                             ? 3
                             : (view_min_lod && is_ld ? 1 : 0));

         nir_tex_instr *tex = nir_tex_instr_create(b->shader, nsrc);
         tex->op = is_ld_ms ? nir_texop_txf_ms
                         : is_ld ? nir_texop_txf
                         : gather ? nir_texop_tg4
                         : has_lod ? nir_texop_txl
                         : has_grad ? nir_texop_txd
                         : has_bias ? nir_texop_txb
                         : implicit_min_lod ? nir_texop_txd
                                            : nir_texop_tex;
         const bool physical_1d_load = is_ld && is_1d;
         tex->sampler_dim = is_buffer          ? GLSL_SAMPLER_DIM_BUF
                            : physical_1d_load ? GLSL_SAMPLER_DIM_2D
                                               : sampler_dim;
         tex->is_array = !is_buffer && is_array;
         tex->is_shadow = compare;
         tex->is_sparse = sparse && !is_buffer;
         tex->dest_type = nir_type_float32;
         tex->coord_components = is_buffer          ? 1
                                 : physical_1d_load ? (is_array ? 3 : 2)
                                                    : coord_components;
         tex->sampler_index = have_smp && !rooted_sampler ? sslot : 0;
         if (gather) {
            for (unsigned k = 1; k < nops; k++) {
               /* SM4/5 encodes GatherRed/Green/Blue/Alpha as a select-one
                * swizzle on the sampler operand. */
               if (o[k].file == D12_DXBC_FILE_SAMPLER) {
                  tex->component = o[k].swizzle[0];
                  break;
               }
            }
         }

         unsigned si = 0;
         tex->src[si++] =
            nir_tex_src_for_ssa(nir_tex_src_texture_handle, tex_handle);

         if (rooted_sampler) {
            nir_def *sampler_handle = nir_bindless_sampler_agx(
               b,
               nir_imul_imm(b, rooted_sampler, D12_SAMPLER_DESC_SIZE),
               nir_u2u16(b, nir_iadd_imm(b, rooted_sampler, 28)),
               .desc_set = D12_UNIFORM_HEAP_SAMPLERS);
            tex->src[si++] = nir_tex_src_for_ssa(
               nir_tex_src_sampler_handle, sampler_handle);
         }

         if (is_ld) {
            /* DXBC registers are untyped. LD coordinates are integer bits
             * for textures as well as buffers; numerically converting the
             * uint bit-pattern 3 as a float produced zero and fetched texel
             * (0, 0) for every nonzero location. */
            nir_def *icoord = coord4;
            nir_def *load_coord =
               is_buffer ? buffer_coord
               : physical_1d_load
                  ? (is_array
                        ? nir_vec3(b, nir_channel(b, icoord, 0),
                                   nir_imm_int(b, 0), nir_channel(b, icoord, 1))
                        : nir_vec2(b, nir_channel(b, icoord, 0),
                                   nir_imm_int(b, 0)))
                  : nir_trim_vector(b, icoord, coord_components);
            tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_coord, load_coord);

            if (has_offset) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_offset,
                  nir_vec2(b, nir_imm_int(b, texel_offset[0]),
                           nir_imm_int(b, texel_offset[1])));
            }

            if (is_ld_ms) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_ms_index, nir_channel(b, SRC(3), 0));
            /* Buffers and multisampled textures have no mip levels.
             * Ordinary texture LD carries mip in .w. */
            } else if (!is_buffer) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_lod, nir_channel(b, icoord, 3));
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_min_lod, view_txf_min_lod);
            }
         } else {
            tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                                 nir_trim_vector(
                                                    b, coord4,
                                                    coord_components));
            if (dynamic_offset) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_offset, nir_trim_vector(b, SRC(2), 2));
            }
            if (has_offset) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_offset,
                  nir_vec2(b, nir_imm_int(b, texel_offset[0]),
                           nir_imm_int(b, texel_offset[1])));
            }
            if (compare) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_comparator,
                  nir_channel(b, SRC(nops - 1), 0));
               if (has_lod)
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_lod,
                     view_min_lod
                        ? nir_fmax(b, nir_imm_float(b, 0.0f), view_min_lod)
                        : nir_imm_int(b, 0));
               else if (implicit_min_lod) {
                  nir_def *gradient_coord =
                     nir_trim_vector(b, coord4, coord_components);
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_ddx, nir_ddx(b, gradient_coord));
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_ddy, nir_ddy(b, gradient_coord));
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_min_lod, view_min_lod);
               }
            } else if (has_lod) {
               nir_def *lod = nir_channel(b, SRC(nops - 1), 0);
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_lod, lod);
               if (view_min_lod)
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_min_lod, view_min_lod);
            } else if (has_grad) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_ddx,
                  nir_trim_vector(b, SRC(nops - 2), coord_components));
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_ddy,
                  nir_trim_vector(b, SRC(nops - 1), coord_components));
               if (view_min_lod)
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_min_lod, view_min_lod);
            } else if (has_bias) {
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_bias, nir_channel(b, SRC(nops - 1), 0));
               if (view_min_lod)
                  tex->src[si++] = nir_tex_src_for_ssa(
                     nir_tex_src_min_lod, view_min_lod);
            } else if (implicit_min_lod) {
               /* AGX carries a minimum LOD with explicit gradients.  Rebuild
                * implicit LOD from the coordinate derivatives; a zero-bias
                * txb is canonicalized back to tex before AGX packs the clamp
                * and would leave an illegal standalone min_lod source. */
               nir_def *gradient_coord =
                  nir_trim_vector(b, coord4, coord_components);
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_ddx, nir_ddx(b, gradient_coord));
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_ddy, nir_ddy(b, gradient_coord));
               tex->src[si++] = nir_tex_src_for_ssa(
                  nir_tex_src_min_lod, view_min_lod);
            }
         }

         nir_def_init(&tex->instr, &tex->def, tex->is_sparse ? 5 : 4, 32);
         nir_builder_instr_insert(b, &tex->instr);

         /* The RESOURCE operand's swizzle reorders the result. */
         unsigned rs[4] = {0, 1, 2, 3};
         if (!gather) {
            for (unsigned k = 1; k < nops; k++) {
               if (o[k].file == D12_DXBC_FILE_RESOURCE) {
                  for (unsigned c = 0; c < 4; c++)
                     rs[c] = o[k].swizzle[c];
                  break;
               }
            }
         }
         nir_def *result = nir_swizzle(b, &tex->def, rs, 4);
         if (view_min_lod_oob)
            result = nir_bcsel(b, view_min_lod_oob,
                               nir_imm_vec4(b, 0, 0, 0, 0), result);
         DST(result);
         if (sparse) {
            nir_def *code;
            if (is_buffer) {
               nir_def *count, *stride;
               buffer_view_info(ctx, tex_operand, &count, &stride);
               code = buffer_feedback_code(
                  ctx, tex_operand,
                  nir_imul(b, nir_u2u64(b, buffer_coord), nir_u2u64(b, stride)),
                  stride, NULL);
               if (ctx->error || !code)
                  return;
            } else {
               code = nir_channel(b, &tex->def, 4);
            }
            write_dst(ctx, &feedback_dst, nir_replicate(b, code, 4));
         }
         break;
      }

      case D12_DXBC_OP_RET:
         /* RET is not necessarily the final token: fxc emits it inside
          * switch cases for source-level early returns.  Treating it as an
          * implicit end-of-shader no-op made the matched case fall through
          * every later label and overwrite its result. */
         if (ctx->hs_instance_phase_open) {
            nir_pop_if(b, NULL);
            ctx->hs_instance_phase_open = false;
         }
         if (ctx->cf_depth && !ctx->conditional_return_if &&
             ctx->cf[0].kind == D12_DXBC_CF_IF && ctx->cf[0].nif)
            ctx->conditional_return_if = ctx->cf[0].nif;
         if (ctx->tok + len < ctx->end) {
            const unsigned next_opcode = ctx->tok[len] & 0x7ff;
            const bool ends_hull_phase =
               b->shader->info.stage == MESA_SHADER_TESS_CTRL &&
               (next_opcode == D12_DXBC_OP_HS_CONTROL_POINT_PHASE ||
                next_opcode == D12_DXBC_OP_HS_FORK_PHASE ||
                next_opcode == D12_DXBC_OP_HS_JOIN_PHASE);
            if (ends_hull_phase)
               break;
            nir_jump(b, nir_jump_return);
         }
         break;

      case D12_DXBC_OP_RETC: {
         NEED(1);
         bool test_zero = !((op_tok >> 18) & 1);
         nir_def *c = nir_channel(b, SRC(0), 0);
         nir_if *nif = nir_push_if(b, test_zero ? nir_ieq_imm(b, c, 0)
                                                : nir_ine_imm(b, c, 0));
         if (!ctx->conditional_return_if)
            ctx->conditional_return_if = nif;
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
         break;
      }

      default:
         /* Name the opcode: the next person needs a work list, not a
          * verdict. */
         /* SURVEY MODE: keep walking and report EVERY missing opcode
          * rather than the first. Which shaders are one addition away
          * from translating is the only thing that predicts what a fix
          * is worth -- the score moves when a shader completes, not when
          * an opcode is added. */
         if (d12_debug()->dxbc_survey) {
            fprintf(stderr, "dxbc-missing: %u operands:", opcode);
            for (unsigned k = 0; k < nops; k++)
               fprintf(stderr, " file%u/idx%u", o[k].file, o[k].index[0]);
            fprintf(stderr, "\n");
            ctx->tok += len;
            continue;
         }
         snprintf(ctx->errbuf, sizeof(ctx->errbuf),
                  "unimplemented DXBC opcode %u", opcode);
         fail(ctx, ctx->errbuf);
         return;
      }

#undef SRC
#undef ISRC
#undef DST
#undef NEED

      ctx->tok += len;
   }

   if (ctx->hs_instance_phase_open) {
      nir_pop_if(b, NULL);
      ctx->hs_instance_phase_open = false;
   }

   /* fxc may omit the control-point phase entirely when it is a pure
    * passthrough. The public dxbc-spirv implementation synthesizes that
    * copy from the input and output signatures; without it the tessellator
    * gets valid factors but undefined control points and emits no pixels. */
   if (b->shader->info.stage == MESA_SHADER_TESS_CTRL &&
       !ctx->hs_has_control_point_phase &&
       b->shader->info.tess.tcs_vertices_out) {
      if (ctx->hs_input_control_points &&
          ctx->hs_input_control_points !=
             b->shader->info.tess.tcs_vertices_out) {
         fail(ctx, "implicit hull passthrough has mismatched point counts");
         return;
      }

      nir_def *vertex = nir_load_invocation_id(b);
      nir_push_if(b, nir_ult_imm(
                        b, vertex, b->shader->info.tess.tcs_vertices_out));
      for (unsigned i = 0; i < ctx->num_out_sig; i++) {
         const struct d12_dxbc_sig_elem *e = &ctx->out_sig[i];
         unsigned loc = output_location(ctx, e, e->reg);
         nir_def *value = nir_load_per_vertex_input(
            b, 4, 32, vertex, nir_imm_int(b, 0), .base = e->reg,
            .dest_type = nir_type_float32, .io_semantics.location = loc,
            .io_semantics.num_slots = 1);
         nir_store_per_vertex_output(
            b, value, vertex, nir_imm_int(b, 0), .base = e->reg,
            .write_mask = e->mask, .src_type = nir_type_float32,
            .io_semantics.location = loc, .io_semantics.num_slots = 1);
         b->shader->info.inputs_read |= BITFIELD64_BIT(loc);
         b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
      }
      nir_pop_if(b, NULL);
   }

   if (b->shader->info.stage == MESA_SHADER_TESS_CTRL)
      b->shader->info.workgroup_size[0] = ctx->hs_max_invocations;
}

/*
 * Translate a DXBC container to NIR.
 *
 * Unsupported opcode/operand combinations fail translation by name.
 * See docs/shader-gaps.md for the measured coverage and remaining gaps.
 */
nir_shader *
d12_dxbc_to_nir(const void *data, size_t size_B, mesa_shader_stage stage,
                const struct d12_binding_map *map,
                uint32_t descriptor_generation, uint8_t *texture_shapes,
                const char **error)
{
   struct d12_native_descriptor_layout native_layout;
   if (!d12_native_descriptor_layout(descriptor_generation, &native_layout)) {
      *error = "unsupported native descriptor generation";
      return NULL;
   }

   const uint8_t *chunk;
   size_t chunk_size;
   uint32_t version = 0;

   if (!d12_dxbc_find_shader(data, size_B, &chunk, &chunk_size, &version)) {
      *error = "no SHEX/SHDR chunk in DXBC container";
      return NULL;
   }

   struct d12_dxbc_ctx ctx;
   memset(&ctx, 0, sizeof(ctx));

   ctx.map = map;
   ctx.descriptor_generation = descriptor_generation;
   ctx.native_layout = native_layout;
   ctx.b = nir_builder_init_simple_shader(stage, &agx_nir_options, "dxbc");

   /* Geometry instance count defaults to one when DCL_GS_INSTANCE_COUNT is
    * absent. Stream 0 is likewise always active for the unstreamed EMIT/CUT
    * opcodes. Leaving either field zero makes libpoly optimize the complete
    * geometry body away even though the token stream contains valid emits. */
   if (stage == MESA_SHADER_GEOMETRY) {
      ctx.b.shader->info.gs.invocations = 1;
      ctx.b.shader->info.gs.active_stream_mask = 1;
   }

   /* Token 0 is the version, token 1 the length in tokens. */
   const uint32_t *tokens = (const uint32_t *)chunk;
   unsigned num_tokens = chunk_size / 4;
   if (num_tokens < 2) {
      *error = "truncated shader chunk";
      ralloc_free(ctx.b.shader);
      return NULL;
   }

   /* The signatures say what each register MEANS. Parse them before any
    * instruction, since the very first store_output needs its location. */
   const uint8_t *sig;
   size_t sig_size;
   bool extended = find_chunk(data, size_B, "ISG1", NULL, &sig, &sig_size);
   if (extended ||
       find_chunk(data, size_B, "ISGN", NULL, &sig, &sig_size)) {
      ctx.num_in_sig =
         parse_signature(sig, sig_size, ctx.in_sig, D12_DXBC_MAX_SIG,
                         extended, extended);
   }
   bool min_precision =
      find_chunk(data, size_B, "OSG1", NULL, &sig, &sig_size);
   bool stream = min_precision;
   if (!stream)
      stream = find_chunk(data, size_B, "OSG5", NULL, &sig, &sig_size);
   if (stream ||
       find_chunk(data, size_B, "OSGN", NULL, &sig, &sig_size)) {
      ctx.num_out_sig =
         parse_signature(sig, sig_size, ctx.out_sig, D12_DXBC_MAX_SIG,
                         stream, min_precision);
   }
   extended = find_chunk(data, size_B, "PSG1", NULL, &sig, &sig_size);
   if (extended ||
       find_chunk(data, size_B, "PCSG", NULL, &sig, &sig_size)) {
      ctx.num_patch_sig =
         parse_signature(sig, sig_size, ctx.patch_sig, D12_DXBC_MAX_SIG,
                         extended, extended);
   }

   ctx.tok = tokens + 2;
   ctx.end = tokens + MIN2(num_tokens, tokens[1]);

   translate_tokens(&ctx);

   if (ctx.error) {
      *error = ctx.error;
      ralloc_free(ctx.b.shader);
      return NULL;
   }

   if (texture_shapes && map) {
      for (unsigned klass = D12_RES_SRV; klass <= D12_RES_UAV; klass++) {
         const uint8_t *dims = klass == D12_RES_SRV ? ctx.srv_dim : ctx.uav_dim;
         const struct d12_dxbc_binding_range *ranges =
            klass == D12_RES_SRV ? ctx.srv_range : ctx.uav_range;

         for (unsigned id = 0; id < D12_DXBC_MAX_SIG; id++) {
            enum d12_texture_shape shape = D12_TEXTURE_SHAPE_NONE;
            switch (dims[id]) {
            case D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D:
               shape = D12_TEXTURE_SHAPE_1D;
               break;
            case D12_DXBC_RESOURCE_DIMENSION_TEXTURE_1D_ARRAY:
               shape = D12_TEXTURE_SHAPE_1D_ARRAY;
               break;
            case D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D:
               shape = D12_TEXTURE_SHAPE_2D;
               break;
            case D12_DXBC_RESOURCE_DIMENSION_TEXTURE_2D_ARRAY:
               shape = D12_TEXTURE_SHAPE_2D_ARRAY;
               break;
            default:
               continue;
            }

            unsigned lower = ranges[id].valid ? ranges[id].lower_bound : id;
            unsigned upper = ranges[id].valid ? ranges[id].upper_bound : id;
            unsigned space = ranges[id].valid ? ranges[id].space : 0;
            for (unsigned e = 0; e < map->count; e++) {
               if (map->entries[e].klass != klass ||
                   map->entries[e].space != space ||
                   map->entries[e].reg < lower || map->entries[e].reg > upper)
                  continue;
               unsigned slot = map->entries[e].slot;
               if (texture_shapes[slot] != D12_TEXTURE_SHAPE_NONE &&
                   texture_shapes[slot] != shape) {
                  *error = "one root slot has conflicting texture declarations";
                  ralloc_free(ctx.b.shader);
                  return NULL;
               }
               texture_shapes[slot] = shape;
            }
         }
      }
   }

   ctx.b.shader->info.shared_size = ctx.shared_size;

   /* AGX divergence analysis deliberately accepts neither continue
    * constructs nor return jumps.  fxc uses both for source-level structured
    * flow, so lower them while the NIR is still owned by this front end. */
   nir_lower_continue_constructs(ctx.b.shader);
   nir_lower_returns(ctx.b.shader);
   nir_validate_shader(ctx.b.shader, "after DXBC translation");
   return ctx.b.shader;
}
