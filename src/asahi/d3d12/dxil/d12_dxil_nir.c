/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DXIL -> NIR translation. This is the piece that makes the campaign's
 * "no SPIR-V" claim real: DXIL instructions and dx.op intrinsics are
 * lowered straight onto NIR with nir_builder, and NIR is what the AGX
 * backend already consumes.
 *
 *     DXIL bitcode --(here)--> NIR --> agx_compile_shader_nir --> AGX ISA
 *
 * DXIL is emitted by DXC as LLVM 3.7 IR whose shader semantics live in
 * calls to `dx.op.*` intrinsics, each taking its opcode as the first
 * argument. Opcode numbering and the LLVM binop/cast codes are taken from
 * Mesa's own MIT-licensed DXIL writer (src/microsoft/compiler), which
 * emits this exact dialect.
 */

#define MESA_LOG_TAG "d3d12agx"

#include "d12_debug.h"
#include "d12_dxil.h"
#include "../d12_native_descriptor_core.h"
#include "poly/nir/poly_nir.h"

#include <stdlib.h>
#include <string.h>

#include "agx_compile.h"
#include "nir.h"
#include "nir_builder.h"
#include "d12_nir_integer.h"
#include "d12_nir_sparse_buffer.h"
#include "d12_nir_min_lod.h"

#include "util/simple_mtx.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"
#include "util/log.h"

/* dx.op opcodes (subset), from Mesa's enum dxil_intr */
enum d12_dxop {
   D12_DXOP_LOAD_INPUT = 4,
   D12_DXOP_STORE_OUTPUT = 5,
   D12_DXOP_FABS = 6,
   D12_DXOP_SATURATE = 7,
   D12_DXOP_FCOS = 12,
   D12_DXOP_FSIN = 13,
   D12_DXOP_FEXP2 = 21,
   D12_DXOP_FRC = 22,
   D12_DXOP_FLOG2 = 23,
   D12_DXOP_SQRT = 24,
   D12_DXOP_RSQRT = 25,
   D12_DXOP_ROUND_NE = 26,
   D12_DXOP_ROUND_NI = 27,
   D12_DXOP_ROUND_PI = 28,
   D12_DXOP_ROUND_Z = 29,
   D12_DXOP_BFREV = 30,
   D12_DXOP_COUNTBITS = 31,
   D12_DXOP_FIRSTBIT_LO = 32,
   D12_DXOP_FIRSTBIT_HI = 33,
   D12_DXOP_FIRSTBIT_SHI = 34,
   D12_DXOP_FMAX = 35,
   D12_DXOP_FMIN = 36,
   D12_DXOP_IMAX = 37,
   D12_DXOP_IMIN = 38,
   D12_DXOP_UMAX = 39,
   D12_DXOP_UMIN = 40,
   D12_DXOP_FMA = 47,
   D12_DXOP_IBFE = 51,
   D12_DXOP_UBFE = 52,
   D12_DXOP_BFI = 53,
   D12_DXOP_CREATE_HANDLE = 57,
   D12_DXOP_TEXTURE_LOAD = 66,
   D12_DXOP_CBUFFER_LOAD = 58,
   D12_DXOP_CBUFFER_LOAD_LEGACY = 59,
   D12_DXOP_BUFFER_LOAD = 68,
   D12_DXOP_BUFFER_STORE = 69,
   D12_DXOP_RAW_BUFFER_LOAD = 139,
   D12_DXOP_RAW_BUFFER_STORE = 140,
   D12_DXOP_ATOMIC_BINOP = 78,
   D12_DXOP_ATOMIC_CMPXCHG = 79,
   D12_DXOP_EVAL_SAMPLE_INDEX = 88,
   D12_DXOP_EVAL_CENTROID = 89,
   /* SM 6.6 replaced createHandle with a binding-descriptor form. It is by
    * far the most common resource intrinsic in modern DXIL.
    */
   /* SM 6.x declares resources as globals; this converts a loaded global
    * into a handle. Like annotateHandle, it is a rename.
    */
   /* Wave (subgroup) operations. AGX has these natively and NIR names
    * them the same way, so the whole family is a mapping exercise.
    */
   D12_DXOP_WAVE_GET_LANE_INDEX = 111,
   D12_DXOP_WAVE_GET_LANE_COUNT = 112,
   D12_DXOP_WAVE_ANY_TRUE = 113,
   D12_DXOP_WAVE_ALL_TRUE = 114,
   D12_DXOP_WAVE_ACTIVE_ALL_EQUAL = 115,
   D12_DXOP_WAVE_READ_LANE_AT = 117,
   D12_DXOP_WAVE_READ_LANE_FIRST = 118,
   D12_DXOP_WAVE_ALL_OP = 119,
   D12_DXOP_WAVE_ACTIVE_BIT = 120,
   D12_DXOP_WAVE_PREFIX_OP = 121,
   D12_DXOP_WAVE_IS_FIRST_LANE = 110,
   D12_DXOP_CREATE_HANDLE_FOR_LIB = 160,
   D12_DXOP_ANNOTATE_HANDLE = 216,
   D12_DXOP_CREATE_HANDLE_FROM_BINDING = 217,
   D12_DXOP_CREATE_HANDLE_FROM_HEAP = 218,
   /* SM 6.6 8-bit packing. The numbering is easy to transpose and doing so
    * is not loud: unpack takes two arguments and pack takes five, so a
    * swap surfaces as "too few arguments" on one of them and as a silently
    * wrong result on the other. These values are the ones the corpus
    * actually carries -- dx.op.unpack4x8.* is 219, dx.op.pack4x8.* is 220.
    */
   D12_DXOP_UNPACK4X8 = 219,
   D12_DXOP_PACK4X8 = 220,
   D12_DXOP_DOT2 = 54,
   D12_DXOP_DOT3 = 55,
   D12_DXOP_DOT4 = 56,
   D12_DXOP_WAVE_ACTIVE_BALLOT = 116,
   /*
    * Inline raytracing (SM 6.5 ray queries). There is no ray-tracing
    * hardware and no acceleration-structure support, and the device
    * reports RaytracingTier 0, so no application can build a structure
    * to trace against. Traversal therefore visits nothing: Proceed is
    * false immediately and the committed status is COMMITTED_NOTHING.
    * That is the correct answer for every state reachable through this
    * driver -- not a placeholder for one.
    */
   D12_DXOP_ALLOCATE_RAY_QUERY = 178,
   D12_DXOP_RQ_TRACE_RAY_INLINE = 179,
   D12_DXOP_RQ_PROCEED = 180,
   D12_DXOP_RQ_ABORT = 181,
   D12_DXOP_RQ_COMMIT_NON_OPAQUE_TRIANGLE_HIT = 182,
   D12_DXOP_RQ_COMMIT_PROCEDURAL_HIT = 183,
   D12_DXOP_RQ_COMMITTED_STATUS = 184,
   D12_DXOP_RQ_CANDIDATE_TYPE = 185,
   D12_DXOP_RQ_MATRIX_FIRST = 186,   /* ObjectToWorld / WorldToObject 3x4 */
   D12_DXOP_RQ_MATRIX_LAST = 189,
   D12_DXOP_RQ_STATE_SCALAR_FIRST = 190, /* flags, T, indices, front face */
   D12_DXOP_RQ_STATE_SCALAR_LAST = 216,
   D12_DXOP_ALLOCATE_RAY_QUERY2 = 258,
   /* Work graphs (SM 6.8) */
   D12_DXOP_ALLOCATE_NODE_OUTPUT_RECORDS = 238,
   D12_DXOP_GET_NODE_RECORD_PTR = 239,
   D12_DXOP_INCREMENT_OUTPUT_COUNT = 240,
   D12_DXOP_OUTPUT_COMPLETE = 241,
   D12_DXOP_GET_INPUT_RECORD_COUNT = 242,
   D12_DXOP_FINISHED_CROSS_GROUP_SHARING = 243,
   D12_DXOP_BARRIER_BY_MEMORY_TYPE = 244,
   D12_DXOP_BARRIER_BY_MEMORY_HANDLE = 245,
   D12_DXOP_BARRIER_BY_NODE_RECORD_HANDLE = 246,
   D12_DXOP_CREATE_NODE_OUTPUT_HANDLE = 247,
   D12_DXOP_INDEX_NODE_HANDLE = 248,
   D12_DXOP_ANNOTATE_NODE_HANDLE = 249,
   D12_DXOP_CREATE_NODE_INPUT_RECORD_HANDLE = 250,
   D12_DXOP_ANNOTATE_NODE_RECORD_HANDLE = 251,
   D12_DXOP_NODE_OUTPUT_IS_VALID = 252,
   D12_DXOP_GET_REMAINING_RECURSION_LEVELS = 253,
   /* Mesh and amplification */
   D12_DXOP_SET_MESH_OUTPUT_COUNTS = 168,
   D12_DXOP_EMIT_INDICES = 169,
   D12_DXOP_GET_MESH_PAYLOAD = 170,
   D12_DXOP_STORE_VERTEX_OUTPUT = 171,
   D12_DXOP_STORE_PRIMITIVE_OUTPUT = 172,
   D12_DXOP_DISPATCH_MESH = 173,
   /* Tessellation and geometry */
   D12_DXOP_EMIT_STREAM = 97,
   D12_DXOP_CUT_STREAM = 98,
   D12_DXOP_EMIT_THEN_CUT_STREAM = 99,
   D12_DXOP_GS_INSTANCE_ID = 100,
   D12_DXOP_LOAD_OUTPUT_CONTROL_POINT = 103,
   D12_DXOP_LOAD_PATCH_CONSTANT = 104,
   D12_DXOP_DOMAIN_LOCATION = 105,
   D12_DXOP_STORE_PATCH_CONSTANT = 106,
   D12_DXOP_OUTPUT_CONTROL_POINT_ID = 107,
   D12_DXOP_WAVE_MULTI_PREFIX_OP = 166,
   D12_DXOP_WAVE_MULTI_PREFIX_BIT_COUNT = 167,
   D12_DXOP_DOT2_ADD_HALF = 162,
   D12_DXOP_DOT4_ADD_I8_PACKED = 163,
   D12_DXOP_DOT4_ADD_U8_PACKED = 164,
   D12_DXOP_SAMPLE_CMP_GRAD = 254,
   D12_DXOP_TEXTURE_STORE_SAMPLE = 225,
   D12_DXOP_START_VERTEX_LOCATION = 256,
   D12_DXOP_START_INSTANCE_LOCATION = 257,
   D12_DXOP_WAVE_MATCH = 165,
   D12_DXOP_MSAD = 50,
   D12_DXOP_SAMPLE = 60,
   D12_DXOP_SAMPLE_CMP = 64,
   D12_DXOP_SAMPLE_CMP_LEVEL_ZERO = 65,
   D12_DXOP_DERIV_COARSE_X = 83,
   D12_DXOP_DERIV_COARSE_Y = 84,
   D12_DXOP_DERIV_FINE_X = 85,
   D12_DXOP_DERIV_FINE_Y = 86,
   D12_DXOP_SAMPLE_BIAS = 61,
   D12_DXOP_SAMPLE_LEVEL = 62,
   D12_DXOP_SAMPLE_GRAD = 63,
   D12_DXOP_TEXTURE_STORE = 67,
   D12_DXOP_BUFFER_UPDATE_COUNTER = 70,
   D12_DXOP_CHECK_ACCESS_FULLY_MAPPED = 71,
   D12_DXOP_GET_DIMENSIONS = 72,
   D12_DXOP_TEXTURE_GATHER = 73,
   D12_DXOP_TEXTURE2DMS_GET_SAMPLE_POSITION = 75,
   D12_DXOP_RENDER_TARGET_GET_SAMPLE_COUNT = 77,
   D12_DXOP_DISCARD = 82,
   D12_DXOP_BARRIER = 80,
   D12_DXOP_SAMPLE_INDEX = 90,
   D12_DXOP_COVERAGE = 91,
   D12_DXOP_INNER_COVERAGE = 92,
   D12_DXOP_PRIMITIVE_ID = 108,
   D12_DXOP_VIEW_ID = 138,
   D12_DXOP_LEGACY_F32TOF16 = 130,
   D12_DXOP_LEGACY_F16TOF32 = 131,
   D12_DXOP_WAVE_ALL_BIT_COUNT = 135,
   D12_DXOP_WAVE_PREFIX_BIT_COUNT = 136,
   D12_DXOP_CALCULATE_LOD = 81,
   D12_DXOP_MAKE_DOUBLE = 101,
   D12_DXOP_SPLIT_DOUBLE = 102,
   D12_DXOP_IS_HELPER_LANE = 221,
   D12_DXOP_QUAD_VOTE = 222,
   D12_DXOP_TEXTURE_GATHER_RAW = 223,
   D12_DXOP_SAMPLE_CMP_LEVEL = 224,
   D12_DXOP_SAMPLE_CMP_BIAS = 255,
   D12_DXOP_QUAD_READ_LANE_AT = 122,
   D12_DXOP_QUAD_OP = 123,
   D12_DXOP_THREAD_ID = 93,
   D12_DXOP_GROUP_ID = 94,
   D12_DXOP_THREAD_ID_IN_GROUP = 95,
   D12_DXOP_FLATTENED_THREAD_ID_IN_GROUP = 96,
};

/* LLVM binop / cast codes, from Mesa's dxil_module.h */
enum {
   D12_BINOP_ADD = 0, D12_BINOP_SUB = 1, D12_BINOP_MUL = 2,
   D12_BINOP_UDIV = 3, D12_BINOP_SDIV = 4, D12_BINOP_UREM = 5,
   D12_BINOP_SREM = 6, D12_BINOP_SHL = 7, D12_BINOP_LSHR = 8,
   D12_BINOP_ASHR = 9, D12_BINOP_AND = 10, D12_BINOP_OR = 11,
   D12_BINOP_XOR = 12,
};

enum {
   D12_CAST_TRUNC = 0, D12_CAST_ZEXT = 1, D12_CAST_SEXT = 2,
   D12_CAST_FPTOUI = 3, D12_CAST_FPTOSI = 4, D12_CAST_UITOFP = 5,
   D12_CAST_SITOFP = 6, D12_CAST_FPTRUNC = 7, D12_CAST_FPEXT = 8,
   D12_CAST_BITCAST = 11,
};

/* pack4x8's PackMode and unpack4x8's UnpackMode operands. */
enum {
   D12_PACK_TRUNC = 0, D12_PACK_SINT = 1, D12_PACK_UINT = 2,
};

enum {
   D12_UNPACK_UNSIGNED = 0, D12_UNPACK_SIGNED = 1,
};

/*
 * DXIL writes outputs one component at a time (storeOutput carries a
 * column index), but the AGX lowering wants one vector store per output.
 * Collect components here and flush them at the end of translation.
 */
#define D12_MAX_OUTPUTS 128

struct output_slot {
   bool used;
   unsigned sig_id;
   unsigned row;
   unsigned location;
   unsigned base;
   nir_def *comps[4];
   nir_def *regs[4];
   unsigned write_mask;
};

struct trans_ctx {
   const struct d12_module *mod;
   const struct d12_function *func;
   nir_builder *b;

   /* value index -> NIR def (NULL when not yet defined / not a value) */
   nir_def **defs;
   unsigned num_defs;

   struct output_slot outputs[D12_MAX_OUTPUTS];

   /* Which slot each binding lives in. Assigned densely from the root
    * signature, so a register space costs nothing unless it is used.
    */
   const struct d12_binding_map *map;

   /* Native heap geometry is part of the compiled shader ABI.  Keep the
    * generation as well as both strides so unsupported software descriptor
    * decodes can fail independently of ordinary hardware image access. */
   uint32_t descriptor_generation;
   struct d12_native_descriptor_layout native_layout;

   /* Per binding slot, from dx.resources: the structured element stride
    * (zero for raw and typed buffers) and the resource shape, which is
    * what actually decides how an access is addressed.
    */
   uint32_t slot_stride[D12_MAX_RESOURCE_BINDINGS];
   uint32_t slot_kind[D12_MAX_RESOURCE_BINDINGS];

   /* One NIR block per DXIL basic block, joined by goto/goto_if and
    * structurized afterwards. Empty for a single-block function, which is
    * built structured as before.
    */
   nir_block **blocks;
   unsigned num_blocks;

   /* value index -> the register a phi reads, for values that cross a
    * block boundary.
    */
   nir_def **phi_reg;

   /*
    * Groupshared globals and allocas, modelled as byte offsets into two
    * flat memories. A pointer value is (space, constant base) plus an
    * optional dynamic part in defs[] -- a GEP whose indices are all
    * constant never materializes an instruction, and the constant part
    * rides in the load/store's immediate base where the backend can fold
    * it.
    */
   /* value index -> the base slot of the RANGE a dynamic handle indexes
    * into, for stride/kind lookup (~0 = not a handle). A handle's def is
    * its slot number; for a constant handle the def alone suffices.
    */
   uint32_t *handle_base;

   /* SM 6.6 heap handles: the def is the HEAP INDEX, not a root slot.
    * kind/stride come from annotateHandle's DxilResourceProperties
    * rather than from a declared range.
    */
   uint8_t *handle_heap;      /* 0 none, 1 resource heap, 2 sampler heap */
   uint32_t *handle_kind_v;   /* decoded resource kind, per value */
   uint32_t *handle_stride_v; /* structured stride, per value */
   /* Runtime validity for dynamically indexed classic handles. NULL means
    * the handle is statically valid. Invalid handles use a safe lookup base. */
   nir_def **handle_valid;

   uint8_t *ptr_space;      /* value index -> D12_PTR_*, 0 = not a pointer */
   uint32_t *ptr_base;      /* value index -> constant byte offset */
   uint32_t *ptr_pointee;   /* value index -> type the pointer points at */
   nir_def **ptr_global;    /* D12_PTR_GLOBAL only: the 64-bit base */
   uint32_t shared_size;    /* bytes of groupshared assigned so far */
   uint32_t scratch_size;   /* bytes of per-thread scratch assigned so far */

   /* Latest 32-bit values stored at constant offsets in a small, linear
    * single-block alloca. Dynamic loads can select these in SSA and avoid
    * AGX stack traffic entirely. */
   nir_def *scratch_shadow[256 / 4];

   const char *error;
};

#define D12_PTR_NONE    0
#define D12_PTR_SHARED  1
#define D12_PTR_SCRATCH 2
/* A 64-bit address in global memory: the amplification payload. The
 * base address lives in ptr_global[], because defs[] already carries the
 * pointer's dynamic 32-bit byte offset.
 */
#define D12_PTR_GLOBAL  3

static void trans_fail(struct trans_ctx *ctx, const char *msg);
static nir_def *get_def(struct trans_ctx *ctx, uint32_t value_index);
static bool const_uint(struct trans_ctx *ctx, uint32_t value_index,
                       uint64_t *out);

/* Shared with the DXBC translator: resolve a shader register to the rooted
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

/* Return the number of consecutive descriptors in the finite API root range
 * beginning at (class, space, reg).  A DXIL declaration may call that range
 * unbounded even when the root signature deliberately supplies a finite
 * table.  Slot arithmetic is valid only while register, slot, range identity,
 * and stage visibility all remain contiguous. */
static unsigned
finite_root_range_count(const struct d12_binding_map *map, unsigned klass,
                        unsigned space, unsigned reg, unsigned base_slot)
{
   unsigned entry = d12_binding_map_find_entry(map, klass, space, reg);
   if (entry == ~0u || map->entries[entry].unbounded ||
       map->entries[entry].slot != base_slot)
      return 0;

   unsigned root_range = map->entries[entry].root_range;
   if (root_range == ~0u)
      return 0;

   unsigned count = 0;
   while (count < D12_MAX_RESOURCE_BINDINGS &&
          count <= UINT32_MAX - reg && count <= UINT32_MAX - base_slot) {
      unsigned next = d12_binding_map_find_entry(
         map, klass, space, reg + count);
      if (next == ~0u || map->entries[next].unbounded ||
          map->entries[next].root_range != root_range ||
          map->entries[next].slot != base_slot + count)
         break;
      count++;
   }

   return count;
}

/* DXIL carries booleans as i1 already, but a value that reached us as a
 * wider integer has to be compared before NIR will take it as a condition.
 */
static nir_def *
to_bool(struct trans_ctx *ctx, nir_def *val)
{
   if (!val)
      return NULL;
   return val->bit_size == 1 ? val : nir_ine_imm(ctx->b, val, 0);
}

static unsigned type_bit_size(const struct d12_module *mod, uint32_t type);

/*
 * Write the values this block feeds into its successor's phis, then
 * branch. The stores have to happen in the PREDECESSOR, before the jump:
 * that is what makes a phi a phi -- which value arrives depends on where
 * control came from, and only the predecessor knows.
 */
static void
store_phis_for_edge(struct trans_ctx *ctx, const struct d12_function *func,
                    unsigned from_block, unsigned to_block)
{
   for (unsigned i = 0; i < func->num_insts; i++) {
      const struct d12_inst *phi = &func->insts[i];

      if (phi->code != D12_FUNC_INST_PHI || phi->block != to_block)
         continue;
      if (phi->result_value == ~0u || !ctx->phi_reg[phi->result_value])
         continue;

      for (unsigned s = 0; s < phi->num_phi_srcs; s++) {
         if (phi->phi_srcs[s].block != from_block)
            continue;

         nir_def *val = get_def(ctx, phi->phi_srcs[s].value);
         if (!val) {
            /* A back edge whose value is not defined yet. Undefined is
             * wrong but silent; refuse instead.
             */
            trans_fail(ctx, "phi source is not available at the branch");
            return;
         }

         nir_store_reg(ctx->b, val, ctx->phi_reg[phi->result_value]);
         break;
      }
   }
}

static void
translate_branch(struct trans_ctx *ctx, const struct d12_function *func,
                 const struct d12_inst *inst)
{
   nir_builder *b = ctx->b;

   if (ctx->num_blocks <= 1) {
      trans_fail(ctx, "branch in a single-block function");
      return;
   }

   const unsigned from = inst->block;

   if (inst->num_extra == 1) {
      /* Unconditional: [bb#] */
      unsigned target = (unsigned)inst->extra[0];
      if (target >= ctx->num_blocks) {
         trans_fail(ctx, "branch target out of range");
         return;
      }

      store_phis_for_edge(ctx, func, from, target);
      nir_goto(b, ctx->blocks[target]);
      return;
   }

   if (inst->num_extra >= 3 && inst->num_ops >= 1) {
      /* Conditional: [true bb#, false bb#, cond] */
      unsigned t = (unsigned)inst->extra[0];
      unsigned f = (unsigned)inst->extra[1];

      if (t >= ctx->num_blocks || f >= ctx->num_blocks) {
         trans_fail(ctx, "branch target out of range");
         return;
      }

      nir_def *cond = get_def(ctx, inst->ops[0]);
      if (!cond) {
         trans_fail(ctx, "branch condition is not available");
         return;
      }

      /* NIR wants a 1-bit condition. */
      if (cond->bit_size != 1)
         cond = nir_ine_imm(b, cond, 0);

      store_phis_for_edge(ctx, func, from, t);
      if (f != t)
         store_phis_for_edge(ctx, func, from, f);

      if (t == f) {
         nir_goto(b, ctx->blocks[t]);
      } else {
         nir_goto_if(b, ctx->blocks[t], cond, ctx->blocks[f]);
      }
      return;
   }

   trans_fail(ctx, "malformed branch");
}

/*
 * Decode a ResBind constant aggregate: { i32 rangeLowerBound,
 * i32 rangeUpperBound, i32 spaceID, i8 resourceClass }.
 */
static bool
decode_res_bind(struct trans_ctx *ctx, uint32_t value_id, unsigned *lower,
                unsigned *upper, unsigned *space, unsigned *klass)
{
   const struct d12_module *mod = ctx->mod;

   if (value_id >= mod->num_values) {
      trans_fail(ctx, "resource binding operand out of range");
      return false;
   }

   const struct d12_value *agg = &mod->values[value_id];

   if (agg->kind != D12_VALUE_CONSTANT) {
      trans_fail(ctx, "resource binding is not a constant ResBind struct");
      return false;
   }

   /* An all-zero ResBind -- register 0, space 0, class SRV -- is emitted
    * as a null constant of the struct type rather than as an aggregate of
    * four zeroes. That is the single most common binding in existence, so
    * treating "no elements" as malformed rejects a large fraction of real
    * shaders.
    */
   if (agg->is_null) {
      *lower = 0;
      *upper = 0;
      *space = 0;
      *klass = D12_RES_SRV;
      return true;
   }

   if (agg->num_agg_elems < 4) {
      trans_fail(ctx, "resource binding is not a constant ResBind struct");
      return false;
   }

   unsigned out[4];
   for (unsigned i = 0; i < 4; i++) {
      uint32_t elem_id = agg->agg_elems[i];
      if (elem_id >= mod->num_values) {
         trans_fail(ctx, "malformed ResBind struct");
         return false;
      }
      const struct d12_value *e = &mod->values[elem_id];
      if (e->kind != D12_VALUE_CONSTANT) {
         trans_fail(ctx, "non-constant field in a ResBind struct");
         return false;
      }
      out[i] = e->is_null ? 0 : (unsigned)e->const_int;
   }

   *lower = out[0];
   *upper = out[1];
   *space = out[2];
   *klass = out[3];
   return true;
}

/*
 * The binding slot of the resource a global variable stands for, or ~0u if
 * the global is not a resource.
 */
static unsigned
resource_slot_for_global(struct trans_ctx *ctx, uint32_t value_id)
{
   const struct d12_module *mod = ctx->mod;

   if (value_id >= mod->num_values ||
       mod->values[value_id].kind != D12_VALUE_GLOBAL)
      return ~0u;

   for (unsigned i = 0; i < mod->num_resources; i++) {
      const struct d12_resource_binding *rb = &mod->resources[i];
      if (rb->global_value != value_id)
         continue;

      return d12_binding_map_find(ctx->map, rb->klass, rb->space,
                                  rb->lower_bound);
   }

   /*
    * Library targets (work graphs, raytracing) leave the resource
    * metadata's global-variable operand undef, so the link above finds
    * nothing. DXC's mangled global name embeds the source name -- the
    * global for `RWBuf` is named "?RWBuf@@3V?$RWStructuredBuffer@I@@A"
    * -- so match on that instead. Anchored on the leading '?' and the
    * trailing "@@" so a name cannot match a longer one containing it.
    */
   const char *gname = mod->values[value_id].name;

   /* LLVM prefixes a name with \01 to suppress its own mangling, so the
    * byte before the '?' is invisible in any printout -- which is
    * exactly how long this took to see.
    */
   if (gname && gname[0] == '\x01')
      gname++;

   if (gname && gname[0] == '?') {
      for (unsigned i = 0; i < mod->num_resources; i++) {
         const struct d12_resource_binding *rb = &mod->resources[i];
         if (!rb->name || !rb->name[0])
            continue;

         size_t n = strlen(rb->name);
         if (!strncmp(gname + 1, rb->name, n) && gname[1 + n] == '@' &&
             gname[2 + n] == '@') {
            return d12_binding_map_find(ctx->map, rb->klass, rb->space,
                                        rb->lower_bound);
         }
      }
   }

   return ~0u;
}

/*
 * Resolve a DXIL resource reference to a root-table slot.
 *
 * `klass` and `range_id` name an entry in dx.resources; classic DXIL
 * createHandle carries the absolute shader register. DXC adds the resource
 * lower bound when constructing the call: element 2 of t4[8] carries index 6.
 * Bounds checking catches a resource reference that does not correspond to
 * anything declared, which would otherwise silently read slot garbage.
 */
static bool
resolve_binding_slot(struct trans_ctx *ctx, unsigned klass,
                     unsigned range_id, unsigned index, unsigned *out_slot)
{
   const struct d12_module *mod = ctx->mod;

   for (unsigned i = 0; i < mod->num_resources; i++) {
      const struct d12_resource_binding *rb = &mod->resources[i];
      if (rb->klass != klass || rb->id != range_id)
         continue;

      unsigned count = d12_range_count(rb);

      if (index < rb->lower_bound ||
          (count && index - rb->lower_bound >= count)) {
         trans_fail(ctx, "resource index is outside its declared range");
         return false;
      }

      unsigned reg = index;
      unsigned relative = index - rb->lower_bound;

      /* Match the dynamic shader-unbounded/finite-root path: arithmetic
       * must stay inside the same contiguous API descriptor range. */
      if (!count) {
         unsigned base = d12_binding_map_find(ctx->map, klass, rb->space,
                                              rb->lower_bound);
         unsigned finite_count = base == ~0u ? 0 :
            finite_root_range_count(ctx->map, klass, rb->space,
                                    rb->lower_bound, base);
         if (!finite_count || relative >= finite_count ||
             relative > UINT32_MAX - base) {
            trans_fail(ctx, "resource index is outside its API root range");
            return false;
         }
         *out_slot = base + relative;
         return true;
      }

      *out_slot = d12_binding_map_find(ctx->map, klass, rb->space, reg);
      if (*out_slot == ~0u) {
         trans_fail(ctx, "shader uses a resource the root signature does "
                         "not bind");
         return false;
      }

      if (d12_debug()->debug_bindings) {
         fprintf(stderr,
                 "binding: class %u rangeId %u index %u -> register %u "
                 "space %u, slot %u (declared %u..%u, stride %uB)\n",
                 klass, range_id, index, reg, rb->space, *out_slot,
                 rb->lower_bound,
                 rb->lower_bound + count - 1, rb->stride_B);
      }

      return true;
   }

   trans_fail(ctx, "resource handle names no declared resource");
   return false;
}

static void
trans_fail(struct trans_ctx *ctx, const char *msg)
{
   if (!ctx->error)
      ctx->error = msg;
}

static bool
type_is_float(const struct d12_module *mod, uint32_t type)
{
   if (type >= mod->num_types)
      return false;
   enum d12_type_code k = mod->types[type].kind;
   return k == D12_TYPE_FLOAT || k == D12_TYPE_DOUBLE || k == D12_TYPE_HALF;
}

static unsigned
type_bit_size(const struct d12_module *mod, uint32_t type)
{
   if (type >= mod->num_types)
      return 32;

   const struct d12_type *t = &mod->types[type];
   switch (t->kind) {
   case D12_TYPE_HALF: return 16;
   case D12_TYPE_FLOAT: return 32;
   case D12_TYPE_DOUBLE: return 64;
   case D12_TYPE_INTEGER: return t->int_bits ? t->int_bits : 32;
   default: return 32;
   }
}

/* Resolve a value index to a NIR def, materializing constants on demand. */
static nir_def *
get_def(struct trans_ctx *ctx, uint32_t value_index)
{
   if (value_index >= ctx->mod->num_values) {
      trans_fail(ctx, "operand references an out-of-range value");
      return nir_imm_int(ctx->b, 0);
   }

   const struct d12_value *v = &ctx->mod->values[value_index];

   /* A constant first materialized inside one CFG branch does not dominate
    * a later use in a sibling branch.  Caching that NIR SSA def globally
    * therefore created invalid SSA; structurization repaired the illegal
    * cross-branch references as undef, silently corrupting vector lanes.
    * Materialize constants at each use and let the normal optimization
    * passes fold duplicates within their valid dominance scopes. */
   if (v->kind != D12_VALUE_CONSTANT && ctx->defs[value_index])
      return ctx->defs[value_index];

   if (v->kind == D12_VALUE_CONSTANT) {
      unsigned bits = type_bit_size(ctx->mod, v->type);
      nir_def *def;

      if (v->is_undef) {
         def = nir_undef(ctx->b, 1, bits);
      } else if (v->is_null) {
         def = nir_imm_intN_t(ctx->b, 0, bits);
      } else if (type_is_float(ctx->mod, v->type)) {
         /* CST_FLOAT carries raw bits of the target float type. */
         if (bits == 32) {
            union { uint32_t u; float f; } bitcast = {
               .u = (uint32_t)v->const_float };
            def = nir_imm_float(ctx->b, bitcast.f);
         } else if (bits == 64) {
            union { uint64_t u; double d; } bitcast = { .u = v->const_float };
            def = nir_imm_double(ctx->b, bitcast.d);
         } else {
            def = nir_imm_intN_t(ctx->b, v->const_float, bits);
         }
      } else {
         def = nir_imm_intN_t(ctx->b, v->const_int, bits);
      }

      return def;
   }

   mesa_loge("dxil: undefined operand: value %u kind %d type %u name '%s'",
             value_index, (int)v->kind, v->type, v->name ? v->name : "");
   trans_fail(ctx, "operand is not a constant or a defined instruction");
   return nir_imm_int(ctx->b, 0);
}

/* Constant integer operand (for dx.op opcodes, signature indices, etc.) */
/*
 * Natural (LLVM datalayout) size and alignment of a type as laid out in
 * groupshared or scratch memory. Returns false for types that have no
 * memory layout here.
 */
static bool
type_size_align(const struct d12_module *mod, uint32_t type,
                uint32_t *out_size, uint32_t *out_align)
{
   if (type >= mod->num_types)
      return false;

   const struct d12_type *t = &mod->types[type];
   switch (t->kind) {
   case D12_TYPE_INTEGER:
      if (t->int_bits != 8 && t->int_bits != 16 && t->int_bits != 32 &&
          t->int_bits != 64)
         return false;
      *out_size = *out_align = t->int_bits / 8;
      return true;
   case D12_TYPE_HALF:
      *out_size = *out_align = 2;
      return true;
   case D12_TYPE_FLOAT:
      *out_size = *out_align = 4;
      return true;
   case D12_TYPE_DOUBLE:
      *out_size = *out_align = 8;
      return true;
   case D12_TYPE_ARRAY:
   case D12_TYPE_VECTOR: {
      uint32_t es, ea;
      if (!type_size_align(mod, t->seq.elem, &es, &ea))
         return false;
      *out_size = es * (uint32_t)t->seq.count;
      *out_align = ea;
      /* LLVM vectors align to their full size rounded up to a power of
       * two; arrays align to the element.
       */
      if (t->kind == D12_TYPE_VECTOR)
         *out_align = util_next_power_of_two(*out_size);
      return true;
   }
   case D12_TYPE_STRUCT_NAMED:
      /*
       * dx.types.Handle is DXIL's opaque handle. DXC emits globals of
       * this type as STORAGE -- a handle is stored into one and loaded
       * back -- so it does need a memory layout, and ours is exactly
       * what a handle is here: a 32-bit index.
       */
      if (t->name && !strcmp(t->name, "dx.types.Handle")) {
         *out_size = *out_align = 4;
         return true;
      }
      FALLTHROUGH;
   case D12_TYPE_STRUCT_ANON: {
      uint32_t off = 0, align = 1;
      for (unsigned i = 0; i < t->strct.num_members; i++) {
         uint32_t ms, ma;
         if (!type_size_align(mod, t->strct.members[i], &ms, &ma))
            return false;
         off = ALIGN_POT(off, ma) + ms;
         align = MAX2(align, ma);
      }
      *out_size = ALIGN_POT(off, align);
      *out_align = align;
      return true;
   }
   default:
      return false;
   }
}

/* Byte offset of struct member `index`, by the same layout rule. */
static bool
struct_member_offset(const struct d12_module *mod, const struct d12_type *t,
                     unsigned index, uint32_t *out_off, uint32_t *out_type)
{
   if (index >= t->strct.num_members)
      return false;

   uint32_t off = 0;
   for (unsigned i = 0; i <= index; i++) {
      uint32_t ms, ma;
      if (!type_size_align(mod, t->strct.members[i], &ms, &ma))
         return false;
      off = ALIGN_POT(off, ma);
      if (i == index) {
         *out_off = off;
         *out_type = t->strct.members[i];
         return true;
      }
      off += ms;
   }
   return false;
}

/*
 * Resolve a value as a pointer, assigning a groupshared global its home on
 * first use. Assignment order is first-use order, which is deterministic
 * for a given module. Only the space/base/pointee tables are written --
 * nothing is emitted, so this is safe under any control flow.
 */
static uint8_t
pointer_for_value(struct trans_ctx *ctx, uint32_t id)
{
   if (id >= ctx->mod->num_values)
      return D12_PTR_NONE;
   if (ctx->ptr_space[id] != D12_PTR_NONE)
      return ctx->ptr_space[id];

   const struct d12_value *v = &ctx->mod->values[id];

   /*
    * A pointer that arrives as a FUNCTION ARGUMENT. Work-graph node
    * entry points take their input record that way -- the record is the
    * parameter, not something the body loads -- so an argument of
    * pointer type is the node input record in global memory.
    */
   if (v->kind == D12_VALUE_ARGUMENT) {
      if (v->type >= ctx->mod->num_types ||
          ctx->mod->types[v->type].kind != D12_TYPE_POINTER)
         return D12_PTR_NONE;

      nir_builder *b = ctx->b;
      nir_cursor saved = b->cursor;
      b->cursor = nir_before_impl(b->impl);

      nir_def *params = nir_load_node_param_buffer_agx(b);
      nir_def *recs = nir_load_global(
         b, 1, 64, nir_iadd_imm(b, params, D12_NODE_OFF_INPUT),
         .align_mul = 8);
      b->cursor = saved;

      ctx->ptr_space[id] = D12_PTR_GLOBAL;
      ctx->ptr_base[id] = 0;
      ctx->ptr_global[id] = recs;
      ctx->ptr_pointee[id] = ctx->mod->types[v->type].ptr.pointee;
      return D12_PTR_GLOBAL;
   }

   /* A constant-expression GEP: the address of a static array element,
    * computed at constant time. Resolve the base pointer, then walk the
    * indices exactly as the GEP instruction does -- every index here is
    * a constant by construction.
    */
   if (v->kind == D12_VALUE_CONSTANT && v->is_ce_gep) {
      if (!v->num_agg_elems)
         return D12_PTR_NONE;

      uint8_t space = pointer_for_value(ctx, v->agg_elems[0]);
      if (space == D12_PTR_NONE)
         return D12_PTR_NONE;

      uint32_t base = ctx->ptr_base[v->agg_elems[0]];
      uint32_t cur = ctx->ptr_pointee[v->agg_elems[0]];

      for (unsigned i = 1; i < v->num_agg_elems; i++) {
         uint64_t idx = 0;
         if (!const_uint(ctx, v->agg_elems[i], &idx))
            return D12_PTR_NONE;

         uint32_t stride, walk_align, next = cur;
         if (i == 1) {
            if (!type_size_align(ctx->mod, cur, &stride, &walk_align))
               return D12_PTR_NONE;
         } else {
            const struct d12_type *t = &ctx->mod->types[cur];
            if (t->kind == D12_TYPE_ARRAY || t->kind == D12_TYPE_VECTOR) {
               next = t->seq.elem;
               if (!type_size_align(ctx->mod, next, &stride, &walk_align))
                  return D12_PTR_NONE;
            } else if (t->kind == D12_TYPE_STRUCT_ANON ||
                       t->kind == D12_TYPE_STRUCT_NAMED) {
               uint32_t moff;
               if (!struct_member_offset(ctx->mod, t, (unsigned)idx, &moff,
                                         &next))
                  return D12_PTR_NONE;
               base += moff;
               cur = next;
               continue;
            } else {
               return D12_PTR_NONE;
            }
         }

         base += (uint32_t)idx * stride;
         cur = next;
      }

      ctx->ptr_space[id] = space;
      ctx->ptr_base[id] = base;
      ctx->ptr_pointee[id] = cur;
      return space;
   }

   if (v->kind != D12_VALUE_GLOBAL)
      return D12_PTR_NONE;

   uint32_t size, align;
   if (!type_size_align(ctx->mod, v->type, &size, &align)) {
      mesa_loge("dxil: global type %u has no memory layout (kind %d, "
                "name '%s')",
                v->type,
                v->type < ctx->mod->num_types
                   ? (int)ctx->mod->types[v->type].kind
                   : -1,
                v->type < ctx->mod->num_types && ctx->mod->types[v->type].name
                   ? ctx->mod->types[v->type].name
                   : "");
      return D12_PTR_NONE;
   }

   if (v->addr_space == D12_ADDR_SPACE_GROUPSHARED) {
      ctx->shared_size = ALIGN_POT(ctx->shared_size, align);
      ctx->ptr_space[id] = D12_PTR_SHARED;
      ctx->ptr_base[id] = ctx->shared_size;
      /* A global's recorded type IS the pointee (struct d12_value). */
      ctx->ptr_pointee[id] = v->type;
      ctx->shared_size += size;
      return D12_PTR_SHARED;
   }

   /* An addrspace-0 global DXC did not optimize away is an HLSL static:
    * per-thread storage, same as an alloca. A missing initializer or a
    * zeroinitializer both qualify -- the latter gets its zeroes stored
    * at function entry, since scratch is not implicitly cleared. A
    * DATA-carrying initializer (a lookup table) does not: handing back
    * zeroed scratch for it would be silently wrong values.
    */
   if (v->addr_space == D12_ADDR_SPACE_DEFAULT) {
      /* An HLSL `static`: per-thread storage, same as an alloca. It may
       * carry an initializer -- zeroes, or a `static const` lookup table
       * (an "immediate constant buffer"). Either way the values are
       * materialized into scratch at function entry, because scratch is
       * not implicitly initialized.
       */
      const struct d12_value *init = NULL;
      if (v->global_init_id) {
         uint32_t iid = v->global_init_id - 1;
         if (iid >= ctx->mod->num_values ||
             ctx->mod->values[iid].kind != D12_VALUE_CONSTANT)
            return D12_PTR_NONE;
         init = &ctx->mod->values[iid];

         /* Only forms whose bytes we can actually reproduce. */
         if (!init->is_null && !init->is_undef && !init->num_data_elems)
            return D12_PTR_NONE;
      }

      if (size > 4096) {
         mesa_loge("dxil: global too large for scratch (%u bytes)", size);
         return D12_PTR_NONE;
      }

      ctx->scratch_size = ALIGN_POT(ctx->scratch_size, align);
      ctx->ptr_space[id] = D12_PTR_SCRATCH;
      ctx->ptr_base[id] = ctx->scratch_size;
      ctx->ptr_pointee[id] = v->type;
      uint32_t base = ctx->scratch_size;
      ctx->scratch_size += size;

      bool zeroed = init && init->is_null;
      bool has_data = init && init->num_data_elems > 0;

      if (zeroed || has_data) {
         /* Emit where every use is dominated: the top of the entry
          * block.
          */
         nir_builder *b = ctx->b;
         nir_cursor saved = b->cursor;
         b->cursor = nir_before_impl(b->impl);

         unsigned esz = 4, ebits = 32;
         if (has_data) {
            const struct d12_type *t = &ctx->mod->types[v->type];
            if (t->kind == D12_TYPE_ARRAY || t->kind == D12_TYPE_VECTOR) {
               unsigned eb = type_bit_size(ctx->mod, t->seq.elem);
               if (eb == 16 || eb == 32 || eb == 64) {
                  ebits = eb;
                  esz = eb / 8;
               }
            }
         }

         for (uint32_t off = 0, i = 0; off < size; off += esz, i++) {
            uint64_t val = 0;
            if (has_data)
               val = i < init->num_data_elems ? init->data_elems[i] : 0;

            nir_def *init_def = nir_imm_intN_t(b, val, ebits);
            nir_store_scratch(b, init_def,
                              nir_imm_int(b, base + off),
                              .write_mask = 1, .align_mul = esz);
            if (ebits == 32 && (base + off) / 4 <
                                  ARRAY_SIZE(ctx->scratch_shadow))
               ctx->scratch_shadow[(base + off) / 4] = init_def;
         }
         b->cursor = saved;
      }
      return D12_PTR_SCRATCH;
   }

   return D12_PTR_NONE;
}

/* NIR component shape for a loaded/stored type. */
static bool
memory_value_shape(const struct d12_module *mod, uint32_t type,
                   unsigned *out_comps, unsigned *out_bits)
{
   if (type >= mod->num_types)
      return false;

   const struct d12_type *t = &mod->types[type];
   if (t->kind == D12_TYPE_VECTOR) {
      unsigned bits = type_bit_size(mod, t->seq.elem);
      if (!bits)
         return false;
      *out_comps = (unsigned)t->seq.count;
      *out_bits = bits;
      return true;
   }

   unsigned bits = type_bit_size(mod, type);
   if (!bits || bits == 1)
      return false;
   *out_comps = 1;
   *out_bits = bits;
   return true;
}

static bool
const_uint(struct trans_ctx *ctx, uint32_t value_index, uint64_t *out)
{
   if (value_index >= ctx->mod->num_values)
      return false;

   const struct d12_value *v = &ctx->mod->values[value_index];
   if (v->kind != D12_VALUE_CONSTANT || v->is_undef)
      return false;

   *out = v->is_null ? 0 : v->const_int;
   return true;
}

/* DXC fills unused trailing operands (offsets, clamp) with undef. */
static bool
operand_is_undef(struct trans_ctx *ctx, uint32_t value_index)
{
   return value_index < ctx->mod->num_values &&
          ctx->mod->values[value_index].kind == D12_VALUE_CONSTANT &&
          ctx->mod->values[value_index].is_undef;
}

static void
def_result(struct trans_ctx *ctx, const struct d12_inst *inst, nir_def *def)
{
   if (inst->result_value < ctx->mod->num_values)
      ctx->defs[inst->result_value] = def;
}

/* AGX returns its hardware sparse residency code beside the texel. Convert
 * that target-specific code to DXIL's status convention: all bits set means
 * fully mapped and zero means at least one accessed texel was nonresident. */
static nir_def *
dxil_sparse_feedback_status(nir_builder *b, nir_def *code)
{
   nir_def *resident = nir_is_sparse_texels_resident(b, 1, code);
   return nir_bcsel(b, resident, nir_imm_int(b, ~0u), nir_imm_int(b, 0));
}

/*
 * A resolved resource handle. `slot` is the index def -- a root-table
 * slot for range handles, a raw heap index for SM 6.6 heap handles --
 * and kind/stride describe how an access through it is addressed: from
 * the declared range for the former, from annotateHandle's
 * DxilResourceProperties for the latter.
 */
struct handle_ref {
   nir_def *slot;
   nir_def *valid;
   bool heap;
   bool rooted_heap;
   bool native_segment;
   unsigned base_slot;
   uint32_t kind;
   uint32_t stride;
};

static bool
handle_info(struct trans_ctx *ctx, uint32_t value_id,
            struct handle_ref *out)
{
   nir_def *def = get_def(ctx, value_id);
   if (!def)
      return false;

   memset(out, 0, sizeof(*out));
   out->slot = def;
   if (value_id < ctx->mod->num_values)
      out->valid = ctx->handle_valid[value_id];

   if (value_id < ctx->mod->num_values && ctx->handle_heap[value_id]) {
      out->heap = true;
      out->rooted_heap = ctx->handle_heap[value_id] == 3;
      out->base_slot = ctx->handle_base[value_id];
      out->kind = ctx->handle_kind_v[value_id];
      out->stride = ctx->handle_stride_v[value_id];
      return true;
   }

   unsigned info_slot;
   if (nir_src_is_const(nir_src_for_ssa(def))) {
      info_slot = nir_src_as_uint(nir_src_for_ssa(def));
   } else if (value_id < ctx->mod->num_values &&
              ctx->handle_base[value_id] != ~0u) {
      info_slot = ctx->handle_base[value_id];
   } else {
      mesa_loge("dxil: dynamic handle from value %u (kind %d) names no "
                "range", value_id,
                value_id < ctx->mod->num_values
                   ? (int)ctx->mod->values[value_id].kind
                   : -1);
      trans_fail(ctx, "handle is dynamic but names no resource range");
      return false;
   }

   if (info_slot >= D12_MAX_RESOURCE_BINDINGS) {
      trans_fail(ctx, "resource binding index out of range");
      return false;
   }

   out->kind = ctx->slot_kind[info_slot];
   out->stride = ctx->slot_stride[info_slot];
   out->base_slot = info_slot;

   /* Dynamic finite texture ranges are not addressed through the dense root
    * slots.  The command list gives every shader declaration an independent
    * native descriptor segment and publishes its byte base in the first root
    * resource slot.  Constant handles retain the compact slot ABI. */
   out->native_segment =
      !nir_src_is_const(nir_src_for_ssa(def)) &&
      d12_texture_shape_from_kind(out->kind) != D12_TEXTURE_SHAPE_NONE;
   return true;
}

/*
 * Resource address arithmetic for the binding model described at
 * D12_DXOP_CREATE_HANDLE. The base address of a slot is read from the root
 * table, then the DXIL coordinates are applied:
 *
 *   structured : addr = base + coord0 * stride + coord1
 *   raw        : addr = base + coord0            (coord1 unused)
 *
 * Which of those applies is decided by the resource SHAPE from
 * dx.resources, not by which dx.op was called: SM 6.0 reaches a
 * ByteAddressBuffer through the same bufferLoad a structured buffer uses.
 * Treating every bufferLoad as structured with an assumed 4-byte stride
 * multiplied raw byte offsets by four -- addresses that were wrong, in
 * bounds, and completely silent.
 *
 * The stride, when one applies, is the one the shader declared. It was
 * also previously hardcoded to 4, which is right only for buffers of
 * 32-bit scalars: a StructuredBuffer of a 12-byte struct indexed at
 * element 3 read from byte 12 instead of byte 36.
 */
enum d12_root_plane {
   D12_ROOT_PLANE_RESOURCE,
   D12_ROOT_PLANE_COUNTER,
   D12_ROOT_PLANE_BUFINFO,
   D12_ROOT_PLANE_SAMPLER,
   D12_ROOT_PLANE_ZERO,
   D12_ROOT_PLANE_DISCARD,
   D12_ROOT_PLANE_RESIDENCY,
   D12_ROOT_PLANE_HEAP_RESIDENCY,
   D12_ROOT_PLANE_SPARSE_RO_OFFSET,
   D12_ROOT_PLANE_HEAP_PBE_NULL,
};

static nir_def *
root_plane(struct trans_ctx *ctx, enum d12_root_plane plane)
{
   nir_builder *b = ctx->b;
   nir_def *root = nir_load_root_agx(b);
   nir_def *header = nir_load_global(b, 4, 64, root, .align_mul = 8);

   if (plane == D12_ROOT_PLANE_RESOURCE)
      return nir_channel(b, header, 0);
   if (plane == D12_ROOT_PLANE_SAMPLER)
      return nir_channel(b, header, 2);
   if (plane == D12_ROOT_PLANE_HEAP_PBE_NULL)
      return nir_channel(b, header, 3);

   nir_def *metadata = nir_channel(b, header, 1);
   if (plane == D12_ROOT_PLANE_RESIDENCY ||
       plane == D12_ROOT_PLANE_HEAP_RESIDENCY ||
       plane == D12_ROOT_PLANE_SPARSE_RO_OFFSET) {
      const unsigned offset =
         plane == D12_ROOT_PLANE_RESIDENCY      ? 32
         : plane == D12_ROOT_PLANE_HEAP_RESIDENCY ? 40
                                                  : 48;
      return nir_load_global(b, 1, 64,
                             nir_iadd_imm(b, metadata, offset),
                             .align_mul = 8);
   }
   nir_def *metadata_header =
      nir_load_global(b, 4, 64, metadata, .align_mul = 8);
   unsigned channel = plane == D12_ROOT_PLANE_COUNTER ? 0
                    : plane == D12_ROOT_PLANE_BUFINFO ? 1
                    : plane == D12_ROOT_PLANE_ZERO    ? 2
                                                     : 3;
   return nir_channel(b, metadata_header, channel);
}

/* Address of a GPU-visible d12_descriptor.  SM 6.6 heap handles use the
 * globally bound resource heap.  A heap-indexed root range instead stores
 * its descriptor-array base in one ordinary root resource slot, which
 * permits multiple independently based tables without another uniform. */
static nir_def *
descriptor_address(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   nir_def *heap;

   if (h->rooted_heap) {
      nir_def *resource_table = root_plane(ctx, D12_ROOT_PLANE_RESOURCE);
      heap = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, resource_table, h->base_slot * sizeof(uint64_t)),
         .align_mul = 8);
   } else {
      heap = nir_load_descriptor_heap_agx(b);
   }

   return nir_iadd(b, heap,
                   nir_imul_imm(b, nir_u2u64(b, h->slot),
                                D12_HEAP_DESC_SIZE));
}

/* Convert a heap-backed handle to the global D3D descriptor-heap index.
 * SM 6.6 handles already carry that index. A root-range handle is relative
 * to the descriptor-array base stored in its root slot, so recover the
 * absolute index from the address and the globally bound 40-byte heap base.
 * The parallel AGX texture/PBE arrays use this same index at the selected
 * descriptor generation's native stride. */
static nir_def *
native_heap_index(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   if (!h->rooted_heap)
      return nir_u2u32(b, h->slot);

   nir_def *descriptor = descriptor_address(ctx, h);
   nir_def *heap = nir_load_descriptor_heap_agx(b);
   nir_def *offset = nir_isub(b, descriptor, heap);
   return nir_u2u32(b, nir_udiv_imm(b, offset, D12_HEAP_DESC_SIZE));
}

static nir_def *
native_image_handle(struct trans_ctx *ctx, const struct handle_ref *h,
                    bool writable)
{
   nir_builder *b = ctx->b;
   const unsigned stride = writable ? ctx->native_layout.pbe_stride_B
                                    : ctx->native_layout.texture_stride_B;

   /* A dynamically indexed texture range addresses the bound heap's own
    * native mirrors: the range's first resource slot carries the heap index
    * of its first descriptor, and the index may legally run past the API
    * range to the end of the heap. The mirrors are refreshed by descriptor
    * writes, so no per-command copy exists. */
   if (h->native_segment) {
      nir_def *resource_table = root_plane(ctx, D12_ROOT_PLANE_RESOURCE);
      nir_def *first = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, resource_table, h->base_slot * sizeof(uint64_t)),
         .align_mul = 8);
      nir_def *relative = nir_iadd_imm(b, h->slot, -(int)h->base_slot);
      nir_def *index = nir_iadd(b, nir_u2u32(b, first), relative);
      return nir_bindless_image_agx(
         b, nir_imul_imm(b, index, stride),
         .desc_set = writable ? D12_UNIFORM_HEAP_PBES
                              : D12_UNIFORM_HEAP_TEXTURES);
   }

   nir_def *index = h->heap ? native_heap_index(ctx, h) : h->slot;
   unsigned desc_set = h->heap
                          ? writable ? D12_UNIFORM_HEAP_PBES
                                     : D12_UNIFORM_HEAP_TEXTURES
                          : writable ? D12_UNIFORM_PBES
                                     : D12_UNIFORM_TEXTURES;
   return nir_bindless_image_agx(b, nir_imul_imm(b, index, stride),
                                 .desc_set = desc_set);
}

static nir_def *
resource_view_size_bytes(struct trans_ctx *ctx, const struct handle_ref *h);

/* AGX texture-buffer loads honour the packed descriptor extent, but typed
 * buffer image atomics do not. Route an out-of-view atomic to the extra null
 * PBE slot at the end of the shader-visible heap. Keeping the selection in
 * the byte offset preserves the bindless-image intrinsic as the handle's
 * direct producer, as required by the AGX backend. */
static nir_def *
native_sampler_index(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   if (!h->rooted_heap)
      return nir_u2u32(b, h->slot);

   nir_def *sampler_table = root_plane(ctx, D12_ROOT_PLANE_SAMPLER);
   nir_def *base = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, sampler_table, h->base_slot * sizeof(uint64_t)),
      .align_mul = 8);
   return nir_iadd(b, nir_u2u32(b, base), nir_u2u32(b, h->slot));
}

static nir_def *
native_sampler_handle(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   nir_def *index = native_sampler_index(ctx, h);
   return nir_bindless_sampler_agx(
      b, nir_imul_imm(b, index, D12_SAMPLER_DESC_SIZE),
      nir_u2u16(b, nir_iadd_imm(b, index, 28)),
      .desc_set = D12_UNIFORM_HEAP_SAMPLERS);
}

static nir_def *
resource_base_address(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;

   /* One indirection to the resource's base address. Range handles go
    * through the root table (8-byte entries); SM 6.6 heap handles index
    * the shader-visible heap's descriptors directly, whose first field
    * is the address. Both bases are placeholders lowered to reserved
    * uniforms after preprocessing; the slot may be a runtime value, and
    * constant folding collapses the multiply for the constant case.
    */
   if (h->heap) {
      return nir_load_global(b, 1, 64, descriptor_address(ctx, h),
                             .align_mul = 8);
   }

   nir_def *resource_table = root_plane(ctx, D12_ROOT_PLANE_RESOURCE);
   return nir_load_global(
      b, 1, 64,
      nir_iadd(b, resource_table,
               nir_imul_imm(b, nir_u2u64(b, h->slot), 8)),
      .align_mul = 8);
}

static nir_def *
resource_address(struct trans_ctx *ctx, const struct handle_ref *h,
                 nir_def *coord0, nir_def *coord1)
{
   nir_builder *b = ctx->b;
   nir_def *base = resource_base_address(ctx, h);

   const uint32_t kind = h->kind;
   const uint32_t stride = h->stride;

   /* DXC uses rawBufferLoad/Store for structured resources in modern DXIL.
    * Their two coordinates retain the structured meaning (element, byte
    * offset); the declared resource shape, not the intrinsic spelling,
    * therefore decides whether the first coordinate needs the stride. */
   bool structured = kind == D12_RES_KIND_STRUCTURED_BUFFER;

   if (kind == D12_RES_KIND_TYPED_BUFFER) {
      trans_fail(ctx, "typed buffer access (needs format conversion)");
      return nir_imm_int64(b, 0);
   }

   nir_def *offset;
   if (structured) {
      if (!stride) {
         trans_fail(ctx, "structured buffer with no declared element "
                         "stride");
         return nir_imm_int64(b, 0);
      }
      offset = nir_iadd(
         b, nir_imul_imm(b, nir_u2u64(b, coord0), stride),
         nir_u2u64(b, coord1));
   } else {
      offset = nir_u2u64(b, coord0);
   }

   return nir_iadd(b, base, offset);
}

/* The shader declaration decides address arithmetic, but robustness is
 * defined by the descriptor VIEW actually bound at execution.  A raw shader
 * may legally alias a structured descriptor (and vice versa), so looking at
 * h->kind here silently turns a small view into its whole backing BO.
 *
 * Buffer descriptors and finite root tables publish {NumElements, stride}.
 * A zero stride is reserved for bare root descriptors, whose first word is
 * already a byte extent. */
static void
resource_view_info(struct trans_ctx *ctx, const struct handle_ref *h,
                   nir_def **count_or_bytes, nir_def **stride_B)
{
   nir_builder *b = ctx->b;

   if (h->heap) {
      nir_def *descriptor = descriptor_address(ctx, h);
      nir_def *info = nir_load_global(
         b, 2, 32,
         nir_iadd_imm(b, descriptor, D12_DESC_BUFFER_COUNT_OFFSET),
         .align_mul = 8);
      *count_or_bytes = nir_channel(b, info, 0);
      *stride_B = nir_channel(b, info, 1);
   } else {
      nir_def *bufinfo_table = root_plane(ctx, D12_ROOT_PLANE_BUFINFO);
      nir_def *info = nir_load_global(
         b, 2, 32,
         nir_iadd(b, bufinfo_table,
                  nir_imul_imm(b, nir_u2u64(b, h->slot), 8)),
         .align_mul = 8);
      *count_or_bytes = nir_channel(b, info, 0);
      *stride_B =
         nir_iand_imm(b, nir_channel(b, info, 1), 0xffff);
   }

   /* The slot for an invalid finite-table index is deliberately clamped to
    * a mapped descriptor so merely reading its metadata is safe.  A zero
    * size then redirects every raw/structured load or store to the command
    * list's mapped zero/sink buffers. */
   if (h->valid)
      *count_or_bytes =
         nir_bcsel(b, h->valid, *count_or_bytes, nir_imm_int(b, 0));
}

static nir_def *
resource_view_count_or_bytes(struct trans_ctx *ctx,
                             const struct handle_ref *h)
{
   nir_def *count_or_bytes, *stride_B;
   resource_view_info(ctx, h, &count_or_bytes, &stride_B);
   return count_or_bytes;
}

static nir_def *
resource_view_size_bytes(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   nir_def *count_or_bytes, *stride_B;
   resource_view_info(ctx, h, &count_or_bytes, &stride_B);
   nir_def *count64 = nir_u2u64(b, count_or_bytes);

   return nir_bcsel(b, nir_ine_imm(b, stride_B, 0),
                    nir_imul(b, count64, nir_u2u64(b, stride_B)), count64);
}

static nir_def *
resource_view_stride_bytes(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_def *count_or_bytes, *stride_B;
   resource_view_info(ctx, h, &count_or_bytes, &stride_B);
   return stride_B;
}

/* Reserved buffers carry one coherent 32-bit residency word per D3D12
 * 64 KiB tile. The descriptor/root plane stores a pointer prebiased to the
 * tile containing the view's first byte. Dense resources publish zero and
 * avoid the per-tile residency-word reads; dead DXIL status results let NIR
 * remove the metadata path entirely for ordinary SRV loads. */
static nir_def *
buffer_residency_pointer(struct trans_ctx *ctx, const struct handle_ref *h)
{
   nir_builder *b = ctx->b;
   nir_def *table;
   nir_def *index;

   if (h->heap) {
      table = root_plane(ctx, D12_ROOT_PLANE_HEAP_RESIDENCY);
      index = native_heap_index(ctx, h);
   } else {
      table = root_plane(ctx, D12_ROOT_PLANE_RESIDENCY);
      index = nir_u2u32(b, h->slot);
   }

   nir_def *has_table = nir_ine_imm(b, table, 0);
   nir_def *safe_index =
      nir_bcsel(b, has_table, index, nir_imm_int(b, 0));
   nir_def *safe_table = nir_bcsel(
      b, has_table, table, root_plane(ctx, D12_ROOT_PLANE_ZERO));
   nir_def *pointer = nir_load_global(
      b, 1, 64,
      nir_iadd(b, safe_table,
               nir_imul_imm(b, nir_u2u64(b, safe_index), 8)),
      .align_mul = 8);
   return nir_bcsel(b, has_table, pointer, nir_imm_int64(b, 0));
}

static nir_def *
resource_base_address_read(struct trans_ctx *ctx, nir_def *base,
                           nir_def *map)
{
   nir_builder *b = ctx->b;
   nir_def *ro_offset = root_plane(ctx, D12_ROOT_PLANE_SPARSE_RO_OFFSET);
   return nir_bcsel(b, nir_ine_imm(b, map, 0), nir_ior(b, base, ro_offset),
                    base);
}

static nir_def *
buffer_residency_status(struct trans_ctx *ctx, const struct handle_ref *h,
                        nir_def *map, nir_def *base, nir_def *byte_offset,
                        nir_def *access_size_B, nir_def *element_valid)
{
   return d12_nir_buffer_residency_status(
      ctx->b, map, base, byte_offset, resource_view_size_bytes(ctx, h),
      access_size_B, element_valid, root_plane(ctx, D12_ROOT_PLANE_ZERO));
}

/* Select a real buffer address only when the complete atomic operand is
 * inside its view and, for a reserved buffer, every touched tile is mapped.
 * Invalid operations still execute against the command-list sink so LLVM
 * CFG translation stays structured; the caller selects their return to zero.
 */
static nir_def *
buffer_atomic_address(struct trans_ctx *ctx, const struct handle_ref *h,
                      nir_def *coord0, nir_def *coord1, unsigned bits,
                      nir_def **out_accessible)
{
   nir_builder *b = ctx->b;
   nir_def *base = resource_base_address(ctx, h);
   nir_def *size = resource_view_size_bytes(ctx, h);
   nir_def *element_valid = NULL;
   nir_def *offset;

   if (h->kind == D12_RES_KIND_TYPED_BUFFER) {
      nir_def *stride =
         nir_umax_imm(b, resource_view_stride_bytes(ctx, h), 1);
      offset = nir_imul(b, nir_u2u64(b, coord0), nir_u2u64(b, stride));
   } else if (h->kind == D12_RES_KIND_STRUCTURED_BUFFER) {
      if (!h->stride) {
         trans_fail(ctx, "structured buffer atomic has zero stride");
         *out_accessible = nir_imm_false(b);
         return root_plane(ctx, D12_ROOT_PLANE_DISCARD);
      }
      offset = nir_iadd(
         b, nir_imul_imm(b, nir_u2u64(b, coord0), h->stride),
         nir_u2u64(b, coord1));
      element_valid =
         nir_ult(b, coord0, resource_view_count_or_bytes(ctx, h));
   } else {
      offset = nir_u2u64(b, coord0);
   }

   nir_def *last = nir_uadd_sat(b, offset, nir_imm_int64(b, bits / 8 - 1));
   nir_def *in_bounds = nir_ult(b, last, nir_u2u64(b, size));
   if (element_valid)
      in_bounds = nir_iand(b, in_bounds, element_valid);

   nir_def *residency = buffer_residency_pointer(ctx, h);
   nir_def *status = buffer_residency_status(
      ctx, h, residency, base, offset, nir_imm_int(b, bits / 8),
      element_valid);
   nir_def *accessible =
      nir_iand(b, in_bounds, nir_ine_imm(b, status, 0));
   *out_accessible = accessible;
   return nir_bcsel(b, accessible, nir_iadd(b, base, offset),
                    root_plane(ctx, D12_ROOT_PLANE_DISCARD));
}

/* Raw D3D buffer robustness is component-granular: a vec4 load straddling the
 * end of a view preserves its in-range lanes and returns zero for the rest.
 * Structured accesses additionally pass element_valid, because their API
 * boundary is NumElements and wrapped byte arithmetic must not resurrect an
 * out-of-range structure.
 *
 * DXIL is still an unstructured LLVM CFG while intrinsics are translated,
 * so inserting nir_if here corrupts multi-block shaders. Select mapped
 * command-list-local zero/sink addresses instead. Invalid accesses remain
 * legal memory operations, but can neither escape the view nor contaminate
 * later invalid reads. */
static nir_def *
robust_load_buffer_vec4(struct trans_ctx *ctx, nir_def *address,
                        nir_def *byte_offset, nir_def *size_bytes,
                        nir_def *element_valid, unsigned bits)
{
   nir_builder *b = ctx->b;
   nir_def *values[4];
   unsigned bytes = bits / 8;
   /* ByteAddressBuffer addressing is 32-bit.  Each vector lane adds its
    * component offset in u32 and may therefore wrap independently.  Doing
    * this add after extending to u64 incorrectly turned a wrapped address
    * near UINT32_MAX into a robust OOB read. */
   nir_def *offset32 = nir_u2u32(b, byte_offset);
   nir_def *offset = element_valid ? nir_u2u64(b, byte_offset)
                                   : nir_u2u64(b, offset32);
   nir_def *base = nir_isub(b, address, offset);
   nir_def *size64 = nir_u2u64(b, size_bytes);
   nir_def *zero = root_plane(ctx, D12_ROOT_PLANE_ZERO);

   for (unsigned c = 0; c < 4; c++) {
      nir_def *component_offset =
         element_valid
            ? nir_iadd_imm(b, offset, c * bytes)
            : nir_u2u64(b, nir_iadd_imm(b, offset32, c * bytes));
      nir_def *component_last = nir_uadd_sat(
         b, component_offset, nir_imm_int64(b, bytes - 1));
      nir_def *in_bounds = nir_ult(b, component_last, size64);
      if (element_valid)
         in_bounds = nir_iand(b, in_bounds, element_valid);
      nir_def *safe = nir_bcsel(
         b, in_bounds,
         nir_iadd(b, base, component_offset), zero);
      values[c] = nir_load_global(b, 1, bits, safe, .align_mul = bytes);
   }

   return nir_vec(b, values, ARRAY_SIZE(values));
}

static void
robust_store_buffer_masked(struct trans_ctx *ctx, nir_def *address,
                           nir_def *byte_offset, nir_def *size_bytes,
                           nir_def *element_valid, nir_def **values,
                           unsigned components)
{
   nir_builder *b = ctx->b;
   unsigned bytes = values[0]->bit_size / 8;
   nir_def *offset32 = nir_u2u32(b, byte_offset);
   nir_def *offset = element_valid ? nir_u2u64(b, byte_offset)
                                   : nir_u2u64(b, offset32);
   nir_def *base = nir_isub(b, address, offset);
   nir_def *size64 = nir_u2u64(b, size_bytes);
   nir_def *discard = root_plane(ctx, D12_ROOT_PLANE_DISCARD);

   for (unsigned c = 0; c < components; c++) {
      nir_def *component_offset =
         element_valid
            ? nir_iadd_imm(b, offset, c * bytes)
            : nir_u2u64(b, nir_iadd_imm(b, offset32, c * bytes));
      nir_def *component_last = nir_uadd_sat(
         b, component_offset, nir_imm_int64(b, bytes - 1));
      nir_def *in_bounds = nir_ult(b, component_last, size64);
      if (element_valid)
         in_bounds = nir_iand(b, in_bounds, element_valid);
      nir_def *safe = nir_bcsel(
         b, in_bounds,
         nir_iadd(b, base, component_offset), discard);
      nir_store_global(b, values[c], safe,
                       .write_mask = 1, .align_mul = bytes);
   }
}

/*
 * Map a patch-constant semantic to its NIR varying slot. The tessellation
 * factors are named slots the tessellator reads directly; everything else
 * is a generic per-patch varying.
 */
static unsigned
dxil_sig_rows(const struct d12_sig_element *se)
{
   return se && se->rows ? se->rows : 1;
}

static unsigned
dxil_sig_start_row(const struct d12_sig_element *se, unsigned sig_id)
{
   return se ? se->start_row : sig_id;
}

static unsigned
dxil_generic_location(const struct d12_sig_element *se, unsigned sig_id)
{
   return VARYING_SLOT_VAR0 + dxil_sig_start_row(se, sig_id);
}

/*
 * The component carried by loadInput/storeOutput is relative to the
 * signature element.  Multiple elements may share one physical register,
 * in which case DXIL metadata's start column places each element inside that
 * register.  Vertex inputs are deliberately exempt: our VBO lowering gives
 * every input-layout element its own NIR attribute and therefore starts it at
 * component zero.  All inter-stage and patch traffic uses the packed physical
 * register and must apply this offset.
 */
static unsigned
dxil_sig_component(const struct d12_sig_element *se, unsigned col)
{
   return (se ? se->start_col : 0) + col;
}

static unsigned
patch_location(const char *sem, unsigned sig_id)
{
   if (sem) {
      if (!strcmp(sem, "SV_TessFactor"))
         return VARYING_SLOT_TESS_LEVEL_OUTER;
      if (!strcmp(sem, "SV_InsideTessFactor"))
         return VARYING_SLOT_TESS_LEVEL_INNER;
   }
   return VARYING_SLOT_PATCH0 + sig_id;
}

static unsigned
dxil_system_value_component_count(const struct d12_module *mod,
                                  const char *name)
{
   unsigned count = 0;
   for (unsigned i = 0; i < mod->num_output_sig; ++i) {
      const struct d12_sig_element *e = &mod->output_sig[i];
      if (e->name && !strcmp(e->name, name))
         count += e->cols;
   }
   return count;
}

static unsigned
dxil_system_value_compact_component(const struct d12_module *mod,
                                    const struct d12_sig_element *target,
                                    unsigned col)
{
   unsigned base = 0;
   for (unsigned i = 0; i < mod->num_output_sig; ++i) {
      const struct d12_sig_element *e = &mod->output_sig[i];
      if (!e->name || strcmp(e->name, target->name))
         continue;
      if (e == target)
         return base + col;
      base += e->cols;
   }
   return base + col;
}

/*
 * Emit the buffered per-component output stores as one vector store per
 * slot, then reset the buffer. Geometry shaders call this at every
 * emitVertex -- a vertex boundary flushes what has been written so far --
 * and every stage calls it once at the end.
 */
static void
flush_outputs(struct trans_ctx *ctx)
{
   nir_builder *b = ctx->b;

   for (unsigned i = 0; i < D12_MAX_OUTPUTS && !ctx->error; i++) {
      struct output_slot *slot = &ctx->outputs[i];
      if (!slot->used)
         continue;

      unsigned num_comps = util_last_bit(slot->write_mask);

      /* A colour store is trimmed BY THE TILEBUFFER LOWERING to the
       * render target's component count, and trimming can only narrow:
       * a shader that writes .xy to an RGBA target must still hand the
       * lowering a 4-vector. The padding lanes are undef, which is
       * exactly what D3D says unwritten components are.
       */
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT &&
          slot->location >= FRAG_RESULT_DATA0 &&
          slot->location < FRAG_RESULT_DATA0 + 8 && num_comps < 4) {
         num_comps = 4;
         slot->write_mask = 0xf;
      }

      /* Varyings leave the shader as 32-bit: that is both what D3D
       * semantics say (interpolants are float unless the target format
       * narrows them) and what the AGX varying store requires. An fp16
       * shader's outputs are widened HERE, explicitly -- stamping the
       * 16-bit value float32 without converting is invalid NIR, and the
       * backend asserts on a 16-bit st_vary source.
       */
      nir_def *comps[4];
      for (unsigned c = 0; c < num_comps; c++) {
         nir_def *v = slot->regs[c]
                         ? nir_load_reg(b, slot->regs[c])
                         : slot->comps[c];
         if (v && v->bit_size != 32)
            v = nir_f2f32(b, v);
         comps[c] = v ? v : nir_undef(b, 1, 32);
      }

      /* A HULL shader's ordinary outputs are PER CONTROL POINT, and the
       * control point is the invocation. poly_nir_lower_tcs addresses a
       * plain store_output with an undefined vertex index -- so all four
       * invocations of a four-point patch write the same slot and three
       * control points are simply never written. The tessellator still
       * runs on the (correct) patch constants, so this presents as a
       * fully tessellated patch whose vertices are all at the origin.
       *
       * Tess factors and generic patch constants arrive through
       * storePatchConstant instead, and are per-patch, so they keep the
       * plain store.
       */
      if (b->shader->info.stage == MESA_SHADER_TESS_CTRL) {
         nir_store_per_vertex_output(
            b, nir_vec(b, comps, num_comps), nir_load_invocation_id(b),
            nir_imm_int(b, 0), .base = slot->base,
            .write_mask = slot->write_mask, .src_type = nir_type_float32,
            .io_semantics.location = slot->location,
            .io_semantics.num_slots = 1);
      } else {
         nir_store_output(b, nir_vec(b, comps, num_comps), nir_imm_int(b, 0),
                          .base = slot->base,
                          .write_mask = slot->write_mask,
                          .src_type = nir_type_float32,
                          .io_semantics.location = slot->location,
                          .io_semantics.num_slots = 1);
      }
   }


   memset(ctx->outputs, 0, sizeof(ctx->outputs));
}

static struct output_slot *
find_output_slot(struct trans_ctx *ctx, unsigned sig_id, unsigned row)
{
   struct output_slot *free_slot = NULL;

   for (unsigned i = 0; i < D12_MAX_OUTPUTS; ++i) {
      struct output_slot *slot = &ctx->outputs[i];
      if (slot->used && slot->sig_id == sig_id && slot->row == row)
         return slot;
      if (!slot->used && !free_slot)
         free_slot = slot;
   }

   if (!free_slot) {
      trans_fail(ctx, "too many output signature rows");
      return NULL;
   }

   free_slot->used = true;
   free_slot->sig_id = sig_id;
   free_slot->row = row;
   return free_slot;
}

/* DXC is permitted to leave derivatives in non-uniform LLVM blocks and
 * relies on the target compiler to execute them for the whole quad.  NIR's
 * derivative intrinsics describe the operation, but they do not repair that
 * placement by themselves.  Move a derivative and its safe dependency chain
 * to the entry block while the DXIL CFG is still explicit.
 *
 * Ordinary global loads are intentionally not speculatable.  Constant-buffer
 * reads are the one exception here: D3D12 makes a CBV read-only to the shader,
 * and the root-table/descriptor loads needed to reach it are immutable for the
 * draw.  Recognize that provenance from the DXIL value graph instead of
 * treating arbitrary load_global instructions (notably UAV reads) as safe.
 */
static bool
dxil_value_is_cbuffer_data(const struct trans_ctx *ctx, uint32_t value)
{
   for (unsigned i = 0; i < ctx->func->num_insts; i++) {
      const struct d12_inst *producer = &ctx->func->insts[i];
      if (producer->result_value != value)
         continue;

      if (producer->code == D12_FUNC_INST_EXTRACTVAL && producer->num_ops)
         return dxil_value_is_cbuffer_data(ctx, producer->ops[0]);

      if (producer->code != D12_FUNC_INST_CALL || producer->num_ops < 1)
         return false;

      uint32_t callee_id = producer->ops[0];
      if (callee_id >= ctx->mod->num_values)
         return false;

      const struct d12_value *callee = &ctx->mod->values[callee_id];
      return callee->kind == D12_VALUE_FUNCTION && callee->name &&
             (!strncmp(callee->name, "dx.op.cbufferLoad.", 18) ||
              !strncmp(callee->name, "dx.op.cbufferLoadLegacy.", 24));
   }

   return false;
}

struct d12_derivative_hoist {
   nir_block *entry;
   nir_cursor cursor;
   bool allow_cbuffer_loads;
};

static bool
can_hoist_derivative_def(nir_def *def, struct d12_derivative_hoist *state);

static bool
can_hoist_derivative_src(nir_src *src, void *data)
{
   return can_hoist_derivative_def(src->ssa, data);
}

static bool
can_hoist_derivative_def(nir_def *def, struct d12_derivative_hoist *state)
{
   if (nir_def_block(def) == state->entry)
      return true;

   nir_instr *instr = nir_def_instr(def);
   bool safe = nir_instr_can_speculate(instr);

   if (!safe && state->allow_cbuffer_loads &&
       instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
      safe = op == nir_intrinsic_load_global ||
             op == nir_intrinsic_load_root_agx;
   }

   return safe && nir_foreach_src(instr, can_hoist_derivative_src, state);
}

static bool
hoist_derivative_src(nir_src *src, void *data);

static void
hoist_derivative_def(nir_def *def, struct d12_derivative_hoist *state)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->block == state->entry)
      return;

   nir_foreach_src(instr, hoist_derivative_src, state);
   nir_instr_move(state->cursor, instr);
   state->cursor = nir_after_instr(instr);
}

static bool
hoist_derivative_src(nir_src *src, void *data)
{
   hoist_derivative_def(src->ssa, data);
   return true;
}

static nir_def *
emit_dxil_derivative(struct trans_ctx *ctx, nir_def *src,
                     nir_intrinsic_op op, bool cbuffer)
{
   nir_builder *b = ctx->b;
   nir_cursor saved = b->cursor;
   nir_block *saved_block = nir_cursor_current_block(saved);
   nir_block *entry = nir_start_block(b->impl);
   struct d12_derivative_hoist state = {
      .entry = entry,
      .cursor = nir_after_block_before_jump(entry),
      .allow_cbuffer_loads = cbuffer,
   };

   bool hoist = ctx->num_blocks > 1 &&
                saved_block != entry &&
                can_hoist_derivative_def(src, &state);
   if (hoist) {
      hoist_derivative_def(src, &state);
      b->cursor = state.cursor;
   }

   nir_def *result;
   switch (op) {
   case nir_intrinsic_ddx_coarse: result = nir_ddx_coarse(b, src); break;
   case nir_intrinsic_ddy_coarse: result = nir_ddy_coarse(b, src); break;
   case nir_intrinsic_ddx_fine:   result = nir_ddx_fine(b, src); break;
   case nir_intrinsic_ddy_fine:   result = nir_ddy_fine(b, src); break;
   default: UNREACHABLE("not a derivative intrinsic");
   }

   if (hoist)
      /* The saved cursor may name the operand instruction we just moved.
       * Reconstruct it from the original block instead of following that
       * instruction into the entry block. */
      b->cursor = nir_after_block_before_jump(saved_block);
   return result;
}

/*
 * dx.op call translation. `args` are the call's operands: args[0] is the
 * callee, args[1] the opcode constant, args[2..] the real parameters.
 */
static bool
translate_dxop(struct trans_ctx *ctx, const struct d12_inst *inst,
               uint64_t opcode, const char *callee)
{
   nir_builder *b = ctx->b;

   if (d12_debug()->dump_insts) {
      fprintf(stderr, "  dxop %llu (%s)\n", (unsigned long long)opcode,
              callee ? callee : "?");
   }

   /* Parameters start after callee + opcode. */
   const uint32_t *p = inst->ops + 2;
   unsigned np = inst->num_ops > 2 ? inst->num_ops - 2 : 0;

#define NEED(n)                                                              \
   do {                                                                      \
      if (np < (n)) {                                                        \
         mesa_loge("dxil: op %u has %u params, needed %u",                   \
                   (unsigned)opcode, np, (unsigned)(n));                     \
         trans_fail(ctx, "dx.op call has too few arguments");                \
         return false;                                                       \
      }                                                                      \
   } while (0)

   switch (opcode) {
   case D12_DXOP_LOAD_INPUT: {
      /* (sigId, rowIndex, colIndex, gsVertexAxis) */
      NEED(3);
      uint64_t sig_id = 0, col = 0;
      const_uint(ctx, p[0], &sig_id);
      if (!const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic component index on loadInput");
         return false;
      }

      /* Resolve the semantic so system values become the right NIR
       * intrinsic rather than a generic varying read.
       */
      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->input_sig, ctx->mod->num_input_sig,
                           (uint32_t)sig_id);
      const char *sem = se ? se->name : NULL;
      nir_def *row = nir_imm_int(b, 0);
      if (dxil_sig_rows(se) > 1) {
         row = get_def(ctx, p[1]);
         if (!row) {
            trans_fail(ctx, "invalid row index on loadInput");
            return false;
         }
         row = nir_u2u32(b, row);
      }

      if (sem && !strcmp(sem, "SV_VertexID")) {
         def_result(ctx, inst, nir_load_vertex_id(b));
         return true;
      }

      if (sem && !strcmp(sem, "SV_InstanceID")) {
         def_result(ctx, inst, nir_load_instance_id(b));
         return true;
      }

      if (sem && !strcmp(sem, "SV_Position") &&
          b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         /* Fragment position is the interpolated frag coord, not a
          * user varying. NIR follows OpenGL/Vulkan and exposes reciprocal
          * clip-W in frag_coord.w, while D3D's SV_Position.w is clip-W.
          */
         nir_def *coord = nir_load_frag_coord(b);
         nir_def *component = nir_channel(b, coord, (unsigned)col);
         if (col == 3)
            component = nir_frcp(b, component);
         def_result(ctx, inst, component);
         b->shader->info.inputs_read |= BITFIELD64_BIT(VARYING_SLOT_POS);
         return true;
      }

      if (sem && !strcmp(sem, "SV_IsFrontFace")) {
         /* Direct3D represents true as all bits set.  A plain boolean-to-int
          * conversion produces 1, which breaks comparisons against ~0u.
          */
         def_result(ctx, inst,
                    nir_ineg(b, nir_b2i32(b, nir_load_front_face(b, 1))));
         return true;
      }

      if (sem && !strcmp(sem, "SV_SampleIndex")) {
         b->shader->info.fs.uses_sample_shading = true;
         def_result(ctx, inst, nir_load_sample_id(b));
         return true;
      }

      if (sem && !strcmp(sem, "SV_RenderTargetArrayIndex") &&
          b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         def_result(ctx, inst, nir_load_layer_id(b));
         b->shader->info.inputs_read |= BITFIELD64_BIT(VARYING_SLOT_LAYER);
         return true;
      }

      /* Signature element -> IO slot. DXIL indexes signature elements;
       * map element N to a generic slot, consistently with outputs.
       *
       * The load's width follows the CALL's declared result type: an
       * fp16 shader loads half-precision inputs, and producing 32-bit
       * defs for them makes every arithmetic use a bit-size mismatch.
       */
      unsigned bits = type_bit_size(ctx->mod, inst->result_type);
      if (bits != 16 && bits != 32)
         bits = 32;

      nir_def *val;

      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         /* Fragment varyings must arrive interpolated: the AGX backend
          * consumes load_interpolated_input with a barycentric source,
          * not a raw load_input.
          */
         unsigned location = dxil_generic_location(se, (unsigned)sig_id);
         unsigned base = dxil_sig_start_row(se, (unsigned)sig_id);
         unsigned rows = dxil_sig_rows(se);
         unsigned component = dxil_sig_component(se, (unsigned)col);
         if (component > 3) {
            trans_fail(ctx, "packed input component outside register");
            return false;
         }
         const unsigned mode = se ? se->interpolation : 0;
         if (mode == 1) {
            val = nir_load_input(
               b, 1, bits, row, .base = base,
               .component = component, .dest_type = nir_type_uint | bits,
               .io_semantics.location = location,
               .io_semantics.num_slots = rows);
            b->shader->info.inputs_read |= BITFIELD64_RANGE(location, rows);
            def_result(ctx, inst, val);
            return true;
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

         val = nir_load_interpolated_input(
            b, 1, bits, bary, row,
            .base = base, .component = component,
            .dest_type = nir_type_float | bits,
            .io_semantics.location = location,
            .io_semantics.num_slots = rows);

         b->shader->info.inputs_read |= BITFIELD64_RANGE(location, rows);
      } else if (b->shader->info.stage == MESA_SHADER_GEOMETRY ||
                 b->shader->info.stage == MESA_SHADER_TESS_CTRL ||
                 b->shader->info.stage == MESA_SHADER_TESS_EVAL) {
         /* These stages read a PRIMITIVE's worth of vertices, so
          * loadInput's fourth operand selects which one. Emitting a
          * plain load_input drops that index and hands the backend an
          * intrinsic no lowering claims.
          *
          * A DOMAIN shader reads the hull shader's CONTROL POINTS the
          * same way. Falling through to the vertex-attribute branch gave
          * it VERT_ATTRIB_GENERIC0 slots and no control-point index, so
          * every corner of the patch read the same undefined attribute
          * and the interpolated position was garbage.
          */
         bool generic = !(sem && (!strcmp(sem, "SV_Position") ||
                                  !strcmp(sem, "SV_ClipDistance")));
         unsigned location = sem && !strcmp(sem, "SV_Position")
                                ? VARYING_SLOT_POS
                                : sem && !strcmp(sem, "SV_ClipDistance")
                                   ? VARYING_SLOT_CLIP_DIST0 +
                                        (se ? se->semantic_index : 0)
                                : dxil_generic_location(se,
                                                        (unsigned)sig_id);
         unsigned base = generic
                            ? dxil_sig_start_row(se, (unsigned)sig_id)
                            : (unsigned)sig_id;
         unsigned rows = generic ? dxil_sig_rows(se) : 1;
         unsigned component =
            generic ? dxil_sig_component(se, (unsigned)col) : (unsigned)col;
         if (component > 3) {
            trans_fail(ctx, "packed per-vertex component outside register");
            return false;
         }
         nir_def *vertex = np >= 4 && !operand_is_undef(ctx, p[3])
                              ? nir_u2u32(b, get_def(ctx, p[3]))
                              : nir_imm_int(b, 0);

         val = nir_load_per_vertex_input(
            b, 1, bits, vertex, row,
            .base = base, .component = component,
            .dest_type = nir_type_float | bits,
            .io_semantics.location = location,
            .io_semantics.num_slots = rows);

         b->shader->info.inputs_read |= BITFIELD64_RANGE(location, rows);
      } else {
         unsigned location = VERT_ATTRIB_GENERIC0 + sig_id;

         val = nir_load_input(b, 1, bits, row,
                              .base = (unsigned)sig_id,
                              .component = (unsigned)col,
                              .dest_type = nir_type_float | bits,
                              .io_semantics.location = location,
                              .io_semantics.num_slots = dxil_sig_rows(se));

         b->shader->info.inputs_read |= BITFIELD64_BIT(sig_id);
      }

      def_result(ctx, inst, val);
      return true;
   }

   case D12_DXOP_EVAL_SAMPLE_INDEX:
   case D12_DXOP_EVAL_CENTROID: {
      /* evalSampleIndex(sigId, rowIndex, colIndex, sampleIndex) and
       * evalCentroid(sigId, rowIndex, colIndex). */
      NEED(opcode == D12_DXOP_EVAL_SAMPLE_INDEX ? 4 : 3);
      uint64_t sig_id = 0, row = 0, col = 0;
      if (!const_uint(ctx, p[0], &sig_id) ||
          !const_uint(ctx, p[1], &row) ||
          !const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic evaluated-input location");
         return false;
      }

      if (b->shader->info.stage != MESA_SHADER_FRAGMENT) {
         trans_fail(ctx, "evaluated input outside a fragment shader");
         return false;
      }

      unsigned bits = type_bit_size(ctx->mod, inst->result_type);
      if (bits != 16 && bits != 32)
         bits = 32;

      nir_def *bary = opcode == D12_DXOP_EVAL_SAMPLE_INDEX
                         ? nir_load_barycentric_at_sample(
                              b, 32, nir_u2u32(b, get_def(ctx, p[3])))
                         : nir_load_barycentric_centroid(b, 32);
      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->input_sig, ctx->mod->num_input_sig,
                           (uint32_t)sig_id);
      const unsigned location = dxil_generic_location(se, (unsigned)sig_id);
      const unsigned base = dxil_sig_start_row(se, (unsigned)sig_id);
      const unsigned rows = dxil_sig_rows(se);
      const unsigned component = dxil_sig_component(se, (unsigned)col);
      if (component > 3) {
         trans_fail(ctx, "packed evaluated-input component outside register");
         return false;
      }
      nir_def *val = nir_load_interpolated_input(
         b, 1, bits, bary, nir_imm_int(b, row), .base = base,
         .component = component, .dest_type = nir_type_float | bits,
         .io_semantics.location = location, .io_semantics.num_slots = rows);
      b->shader->info.inputs_read |= BITFIELD64_RANGE(location, rows);
      def_result(ctx, inst, val);
      return true;
   }

   case D12_DXOP_STORE_OUTPUT: {
      /* (sigId, rowIndex, colIndex, value) */
      NEED(4);
      uint64_t sig_id = 0, row = 0, col = 0;
      const_uint(ctx, p[0], &sig_id);
      if (!const_uint(ctx, p[1], &row)) {
         trans_fail(ctx, "dynamic row index on storeOutput");
         return false;
      }
      if (!const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic component index on storeOutput");
         return false;
      }

      nir_def *val = get_def(ctx, p[3]);

      /* Semantic-driven, not positional: SV_Position is the clip-space
       * position wherever it appears in the signature.
       */
      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->output_sig, ctx->mod->num_output_sig,
                           (uint32_t)sig_id);
      const char *sem = se ? se->name : NULL;
      if (row >= dxil_sig_rows(se)) {
         trans_fail(ctx, "storeOutput row outside signature element");
         return false;
      }

      /* Depth, stencil-ref and coverage are scalar side channels with
       * their own consumers (agx_nir_lower_discard_zs_emit and
       * agx_nir_lower_sample_intrinsics), not colour outputs: store them
       * now and skip the vec4 collection entirely.
       */
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT && sem &&
          (!strcmp(sem, "SV_Depth") || !strcmp(sem, "SV_StencilRef") ||
           !strcmp(sem, "SV_Coverage"))) {
         unsigned loc;
         nir_alu_type t;
         if (!strcmp(sem, "SV_Depth")) {
            loc = FRAG_RESULT_DEPTH;
            t = nir_type_float32;
         } else if (!strcmp(sem, "SV_StencilRef")) {
            loc = FRAG_RESULT_STENCIL;
            t = nir_type_uint32;
         } else {
            loc = FRAG_RESULT_SAMPLE_MASK;
            t = nir_type_uint32;
         }

         nir_store_output(b, val, nir_imm_int(b, 0), .base = 0,
                          .write_mask = 1, .src_type = t,
                          .io_semantics.location = loc,
                          .io_semantics.num_slots = 1);
         b->shader->info.outputs_written |= BITFIELD64_BIT(loc);
         return true;
      }

      unsigned location;
      unsigned output_col = dxil_sig_component(se, (unsigned)col);
      unsigned output_base = (unsigned)sig_id;
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         /* SV_Target[n]; the semantic index rides in start_row. */
         location = FRAG_RESULT_DATA0 + (se ? se->start_row : sig_id);
      } else if (sem && !strcmp(sem, "SV_Position")) {
         location = VARYING_SLOT_POS;
      } else if (sem && !strcmp(sem, "SV_RenderTargetArrayIndex")) {
         location = VARYING_SLOT_LAYER;
      } else if (sem && !strcmp(sem, "SV_ClipDistance")) {
         const unsigned clip_count = dxil_system_value_component_count(
            ctx->mod, "SV_ClipDistance");
         const unsigned compact = dxil_system_value_compact_component(
            ctx->mod, se, col);
         location = VARYING_SLOT_CLIP_DIST0 + (compact / 4);
         output_col = compact % 4;
         b->shader->info.clip_distance_array_size = clip_count;
      } else if (sem && !strcmp(sem, "SV_CullDistance")) {
         const unsigned clip_count = dxil_system_value_component_count(
            ctx->mod, "SV_ClipDistance");
         const unsigned cull_count = dxil_system_value_component_count(
            ctx->mod, "SV_CullDistance");
         const unsigned compact = dxil_system_value_compact_component(
            ctx->mod, se, col);
         const unsigned combined = clip_count + compact;
         location = VARYING_SLOT_CLIP_DIST0 + (combined / 4);
         output_col = combined % 4;
         b->shader->info.clip_distance_array_size = clip_count;
         b->shader->info.cull_distance_array_size = cull_count;
      } else {
         output_base = dxil_sig_start_row(se, (unsigned)sig_id) + row;
         location = VARYING_SLOT_VAR0 + output_base;
      }
      b->shader->info.outputs_written |= BITFIELD64_BIT(location);

      if (output_col > 3) {
         trans_fail(ctx, "output signature index or component out of range");
         return false;
      }

      /* Buffered; flushed as one vector store per output. */
      struct output_slot *slot =
         find_output_slot(ctx, (unsigned)sig_id, (unsigned)row);
      if (!slot)
         return false;
      slot->location = location;
      slot->base = output_base;

      /* In a multi-block shader, the latest storeOutput in translation
       * order is not necessarily the value produced at runtime. DXC
       * commonly writes a default before an if and overwrites it in one
       * arm. Keeping only the latter SSA def turns the other arm into
       * undef when outputs are flushed in the common epilogue.
       *
       * Model each component as a NIR register across control flow. The
       * existing reg-to-SSA pass then reconstructs the required phi, while
       * flush_outputs still presents AGX with one vector store per output.
       */
      if (ctx->num_blocks > 1) {
         if (!slot->regs[output_col]) {
            nir_cursor saved = b->cursor;
            b->cursor = nir_before_block(ctx->blocks[0]);
            slot->regs[output_col] = nir_decl_reg(b, 1, val->bit_size, 0);
            nir_store_reg(b, nir_undef(b, 1, val->bit_size),
                          slot->regs[output_col]);
            b->cursor = saved;
         }
         nir_store_reg(b, val, slot->regs[output_col]);
      } else {
         slot->comps[output_col] = val;
      }
      slot->write_mask |= 1u << output_col;
      return true;
   }

   /* Unary float */
   case D12_DXOP_FABS:     NEED(1); def_result(ctx, inst, nir_fabs(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_SATURATE: NEED(1); def_result(ctx, inst, nir_fsat(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FCOS:     NEED(1); def_result(ctx, inst, nir_fcos(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FSIN:     NEED(1); def_result(ctx, inst, nir_fsin(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FEXP2:    NEED(1); def_result(ctx, inst, nir_fexp2(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FLOG2:    NEED(1); def_result(ctx, inst, nir_flog2(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FRC:      NEED(1); def_result(ctx, inst, nir_ffract(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_SQRT:     NEED(1); def_result(ctx, inst, nir_fsqrt(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_RSQRT:    NEED(1); def_result(ctx, inst, nir_frsq(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_ROUND_NE: NEED(1); def_result(ctx, inst, nir_fround_even(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_ROUND_NI: NEED(1); def_result(ctx, inst, nir_ffloor(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_ROUND_PI: NEED(1); def_result(ctx, inst, nir_fceil(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_ROUND_Z:  NEED(1); def_result(ctx, inst, nir_ftrunc(b, get_def(ctx, p[0]))); return true;

   /* Unary integer */
   case D12_DXOP_BFREV:       NEED(1); def_result(ctx, inst, nir_bitfield_reverse(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_COUNTBITS:   NEED(1); def_result(ctx, inst, nir_bit_count(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FIRSTBIT_LO: NEED(1); def_result(ctx, inst, nir_find_lsb(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FIRSTBIT_HI: NEED(1); def_result(ctx, inst, nir_ufind_msb(b, get_def(ctx, p[0]))); return true;
   case D12_DXOP_FIRSTBIT_SHI:NEED(1); def_result(ctx, inst, nir_ifind_msb(b, get_def(ctx, p[0]))); return true;

   /* Binary */
   case D12_DXOP_FMAX: NEED(2); def_result(ctx, inst, nir_fmax(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;
   case D12_DXOP_FMIN: NEED(2); def_result(ctx, inst, nir_fmin(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;
   case D12_DXOP_IMAX: NEED(2); def_result(ctx, inst, nir_imax(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;
   case D12_DXOP_IMIN: NEED(2); def_result(ctx, inst, nir_imin(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;
   case D12_DXOP_UMAX: NEED(2); def_result(ctx, inst, nir_umax(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;
   case D12_DXOP_UMIN: NEED(2); def_result(ctx, inst, nir_umin(b, get_def(ctx, p[0]), get_def(ctx, p[1]))); return true;

   /* Ternary */
   case D12_DXOP_FMA:
      NEED(3);
      def_result(ctx, inst,
                 nir_ffma(b, get_def(ctx, p[0]), get_def(ctx, p[1]),
                          get_def(ctx, p[2])));
      return true;

   case D12_DXOP_IBFE:
      NEED(3);
      def_result(ctx, inst,
                 nir_ibitfield_extract(b, get_def(ctx, p[2]),
                                       get_def(ctx, p[1]),
                                       get_def(ctx, p[0])));
      return true;

   case D12_DXOP_UBFE:
      NEED(3);
      def_result(ctx, inst,
                 nir_ubitfield_extract(b, get_def(ctx, p[2]),
                                       get_def(ctx, p[1]),
                                       get_def(ctx, p[0])));
      return true;

   case D12_DXOP_BFI:
      /* (width, offset, value, replacedValue) */
      NEED(4);
      def_result(ctx, inst,
                 nir_bitfield_insert(b, get_def(ctx, p[3]),
                                     get_def(ctx, p[2]),
                                     get_def(ctx, p[1]),
                                     get_def(ctx, p[0])));
      return true;

   case D12_DXOP_PACK4X8: {
      /*
       * pack4x8(packMode, x, y, z, w) -> i32
       *
       * The mode decides whether each component is merely truncated or
       * first saturated, and a saturating mode is not expressible as a
       * mask: pack_clamp_s8(200) is 127, not 0xc8 & 0xff. Reading the mode
       * as a constant is therefore mandatory rather than an optimisation,
       * and a dynamic one has to be refused instead of assumed -- guessing
       * Trunc would turn every out-of-range component into wraparound
       * garbage that still looks like a plausible colour.
       */
      NEED(5);

      uint64_t mode = 0;
      if (!const_uint(ctx, p[0], &mode)) {
         trans_fail(ctx, "non-constant pack4x8 mode");
         return false;
      }
      if (mode > D12_PACK_UINT) {
         trans_fail(ctx, "unhandled pack4x8 mode");
         return false;
      }

      nir_def *packed = nir_imm_int(b, 0);

      for (unsigned i = 0; i < 4; i++) {
         nir_def *c = get_def(ctx, p[1 + i]);

         if (mode == D12_PACK_TRUNC) {
            /* Mask first, then widen: the i16 overload's high bits are
             * discarded either way, so the widening cannot be signed or
             * unsigned wrongly.
             */
            c = nir_u2u32(b, nir_iand_imm(b, c, 0xff));
         } else {
            /* Both saturating modes take a SIGNED source -- UInt clamps
             * negatives to 0 rather than reinterpreting them as large
             * positives -- so the i16 overload sign-extends for both.
             */
            c = nir_i2i32(b, c);
            c = mode == D12_PACK_SINT
                   ? nir_imin_imm(b, nir_imax_imm(b, c, -128), 127)
                   : nir_imin_imm(b, nir_imax_imm(b, c, 0), 255);
            c = nir_iand_imm(b, c, 0xff);
         }

         packed = nir_ior(b, packed, nir_ishl_imm(b, c, 8 * i));
      }

      def_result(ctx, inst, packed);
      return true;
   }

   case D12_DXOP_UNPACK4X8: {
      /*
       * unpack4x8(unpackMode, packedVal) -> %dx.types.fouri32
       *
       * Struct-returning dx.op results are represented as NIR vectors and
       * picked apart by extractvalue, so a 4-component vector IS the
       * struct here and needs nothing further.
       *
       * The component width comes from the returned struct rather than
       * being assumed 32: there is an i16 overload, and it is the more
       * common one in the corpus. Handing back 32-bit components for it
       * would mismatch the bit size of every consumer.
       */
      NEED(2);

      uint64_t mode = 0;
      if (!const_uint(ctx, p[0], &mode)) {
         trans_fail(ctx, "non-constant unpack4x8 mode");
         return false;
      }
      if (mode > D12_UNPACK_SIGNED) {
         trans_fail(ctx, "unhandled unpack4x8 mode");
         return false;
      }

      unsigned bits = 32;
      if (inst->result_type < ctx->mod->num_types) {
         const struct d12_type *t = &ctx->mod->types[inst->result_type];
         if ((t->kind == D12_TYPE_STRUCT_ANON ||
              t->kind == D12_TYPE_STRUCT_NAMED) && t->strct.num_members)
            bits = type_bit_size(ctx->mod, t->strct.members[0]);
      }

      nir_def *packed = get_def(ctx, p[1]);
      nir_def *comps[4];

      for (unsigned i = 0; i < 4; i++) {
         if (mode == D12_UNPACK_SIGNED) {
            /* Shift byte i up to the sign bit and arithmetic-shift back:
             * sign extension without a dedicated 8-bit type.
             */
            nir_def *byte =
               nir_ishr_imm(b, nir_ishl_imm(b, packed, 24 - 8 * i), 24);
            comps[i] = nir_i2iN(b, byte, bits);
         } else {
            nir_def *byte =
               nir_iand_imm(b, nir_ushr_imm(b, packed, 8 * i), 0xff);
            comps[i] = nir_u2uN(b, byte, bits);
         }
      }

      def_result(ctx, inst, nir_vec(b, comps, 4));
      return true;
   }

   /* Compute system values */
   case D12_DXOP_THREAD_ID:
   case D12_DXOP_GROUP_ID:
   case D12_DXOP_THREAD_ID_IN_GROUP: {
      NEED(1);
      uint64_t comp = 0;
      if (!const_uint(ctx, p[0], &comp)) {
         trans_fail(ctx, "dynamic component on a thread-id intrinsic");
         return false;
      }

      nir_def *vec = opcode == D12_DXOP_THREAD_ID
                        ? nir_load_global_invocation_id(b, 32)
                        : (opcode == D12_DXOP_GROUP_ID
                              ? nir_load_workgroup_id(b)
                              : nir_load_local_invocation_id(b));

      def_result(ctx, inst, nir_channel(b, vec, (unsigned)comp));
      return true;
   }

   case D12_DXOP_ALLOCATE_RAY_QUERY:
   case D12_DXOP_ALLOCATE_RAY_QUERY2:
      /* Returns an opaque query handle; ours carries the ray flags. */
      def_result(ctx, inst,
                 np >= 1 ? nir_u2u32(b, get_def(ctx, p[0]))
                         : nir_imm_int(b, 0));
      return true;

   case D12_DXOP_RQ_TRACE_RAY_INLINE:
   case D12_DXOP_RQ_ABORT:
   case D12_DXOP_RQ_COMMIT_NON_OPAQUE_TRIANGLE_HIT:
   case D12_DXOP_RQ_COMMIT_PROCEDURAL_HIT:
      /* Nothing to traverse and nothing to commit against. */
      return true;

   case D12_DXOP_RQ_PROCEED:
      /* Traversal is complete before it starts. */
      def_result(ctx, inst, nir_imm_false(b));
      return true;

   case D12_DXOP_RQ_COMMITTED_STATUS:
      /* COMMITTED_NOTHING */
      def_result(ctx, inst, nir_imm_int(b, 0));
      return true;

   case D12_DXOP_RQ_CANDIDATE_TYPE:
      /* CANDIDATE_NON_OPAQUE_TRIANGLE; unreachable, since Proceed never
       * reports a candidate.
       */
      def_result(ctx, inst, nir_imm_int(b, 0));
      return true;

   case D12_DXOP_CREATE_NODE_OUTPUT_HANDLE:
   case D12_DXOP_CREATE_NODE_INPUT_RECORD_HANDLE:
      /* (index) -> a handle naming one of the entry's declared node
       * inputs/outputs. Like a resource handle, the handle IS the index;
       * what it addresses is decided where it is used.
       */
      NEED(1);
      def_result(ctx, inst, nir_u2u32(b, get_def(ctx, p[0])));
      return true;

   case D12_DXOP_ANNOTATE_NODE_HANDLE:
   case D12_DXOP_ANNOTATE_NODE_RECORD_HANDLE:
   case D12_DXOP_INDEX_NODE_HANDLE:
      /* Attaches properties, or selects an element of a node array.
       * Either way the handle is unchanged for our purposes.
       */
      NEED(1);
      def_result(ctx, inst, get_def(ctx, p[0]));
      return true;

   case D12_DXOP_ALLOCATE_NODE_OUTPUT_RECORDS: {
      /* (handle, count, perThread) -> a record handle. Bump the output
       * cursor by the requested count; the returned handle carries the
       * base index the caller writes at.
       */
      NEED(2);
      nir_def *params = nir_load_node_param_buffer_agx(b);
      nir_def *counts = nir_load_global(
         b, 1, 64, nir_iadd_imm(b, params, D12_NODE_OFF_COUNTS),
         .align_mul = 8);
      nir_def *base = nir_global_atomic(
         b, 32, nir_iadd_imm(b, counts, 4),
         nir_u2u32(b, get_def(ctx, p[1])), .atomic_op = nir_atomic_op_iadd);
      def_result(ctx, inst, base);
      return true;
   }

   case D12_DXOP_GET_NODE_RECORD_PTR: {
      /* (handle, arrayIndex) -> a pointer into the record array, walked
       * by getelementptr like any other memory. Records live in global
       * memory, so this joins the pointer model there.
       */
      NEED(2);
      nir_def *params = nir_load_node_param_buffer_agx(b);
      nir_def *recs = nir_load_global(
         b, 1, 64, nir_iadd_imm(b, params, D12_NODE_OFF_INPUT),
         .align_mul = 8);
      nir_def *stride = nir_load_global(
         b, 1, 32, nir_iadd_imm(b, params, D12_NODE_OFF_STRIDE),
         .align_mul = 4);

      nir_def *idx = nir_u2u32(b, get_def(ctx, p[1]));
      nir_def *addr =
         nir_iadd(b, recs, nir_u2u64(b, nir_imul(b, idx, stride)));

      if (inst->result_value < ctx->mod->num_values) {
         ctx->ptr_space[inst->result_value] = D12_PTR_GLOBAL;
         ctx->ptr_base[inst->result_value] = 0;
         ctx->ptr_global[inst->result_value] = addr;
         if (inst->result_type < ctx->mod->num_types &&
             ctx->mod->types[inst->result_type].kind == D12_TYPE_POINTER) {
            ctx->ptr_pointee[inst->result_value] =
               ctx->mod->types[inst->result_type].ptr.pointee;
         }
      }
      return true;
   }

   case D12_DXOP_GET_INPUT_RECORD_COUNT: {
      NEED(1);
      nir_def *params = nir_load_node_param_buffer_agx(b);
      nir_def *counts = nir_load_global(
         b, 1, 64, nir_iadd_imm(b, params, D12_NODE_OFF_COUNTS),
         .align_mul = 8);
      def_result(ctx, inst,
                 nir_load_global(b, 1, 32, counts, .align_mul = 8));
      return true;
   }

   case D12_DXOP_INCREMENT_OUTPUT_COUNT: {
      /* (handle, count, perThread) */
      NEED(2);
      nir_def *params = nir_load_node_param_buffer_agx(b);
      nir_def *counts = nir_load_global(
         b, 1, 64, nir_iadd_imm(b, params, D12_NODE_OFF_COUNTS),
         .align_mul = 8);
      nir_global_atomic(b, 32, nir_iadd_imm(b, counts, 4),
                        nir_u2u32(b, get_def(ctx, p[1])),
                        .atomic_op = nir_atomic_op_iadd);
      return true;
   }

   case D12_DXOP_OUTPUT_COMPLETE:
   case D12_DXOP_FINISHED_CROSS_GROUP_SHARING:
      /* Signals to the scheduler that this node's records are done.
       * With no scheduler these carry no shader-visible effect; the
       * memory ordering they imply is covered by the barriers DXC emits
       * alongside them.
       */
      return true;

   case D12_DXOP_BARRIER_BY_MEMORY_TYPE:
   case D12_DXOP_BARRIER_BY_MEMORY_HANDLE:
   case D12_DXOP_BARRIER_BY_NODE_RECORD_HANDLE:
      /* SM 6.8 barriers name what they order rather than taking a mask.
       * Order everything: correct, and no weaker than what was asked.
       */
      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_DEVICE,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_global | nir_var_mem_shared |
                                  nir_var_image);
      return true;

   case D12_DXOP_NODE_OUTPUT_IS_VALID:
      /* Every output a shader can name is one the graph declared. */
      def_result(ctx, inst, nir_imm_true(b));
      return true;

   case D12_DXOP_GET_REMAINING_RECURSION_LEVELS:
      /* No recursion budget is tracked without a scheduler. */
      def_result(ctx, inst, nir_imm_int(b, 0));
      return true;

   case D12_DXOP_SET_MESH_OUTPUT_COUNTS:
   case D12_DXOP_EMIT_INDICES:
   case D12_DXOP_GET_MESH_PAYLOAD:
   case D12_DXOP_STORE_VERTEX_OUTPUT:
   case D12_DXOP_STORE_PRIMITIVE_OUTPUT:
   case D12_DXOP_DISPATCH_MESH: {
      /* Mesh and amplification run as plain compute: the shader writes
       * its outputs to driver buffers named by a parameter block (see
       * D12_UNIFORM_MESH_PARAMS). Every address below is one indirection
       * through that block, which the driver fills per dispatch.
       */
      nir_def *params = nir_load_mesh_param_buffer_agx(b);

#define MESH_PTR(off)                                                        \
   nir_load_global(b, 1, 64, nir_iadd_imm(b, params, (off)), .align_mul = 8)

      switch (opcode) {
      case D12_DXOP_SET_MESH_OUTPUT_COUNTS: {
         /* (numVertices, numPrimitives) */
         NEED(2);
         nir_def *counts = MESH_PTR(D12_MESH_OFF_COUNTS);
         nir_store_global(b, nir_u2u32(b, get_def(ctx, p[0])), counts,
                          .write_mask = 1, .align_mul = 8);
         nir_store_global(b, nir_u2u32(b, get_def(ctx, p[1])),
                          nir_iadd_imm(b, counts, 4), .write_mask = 1,
                          .align_mul = 4);
         return true;
      }

      case D12_DXOP_EMIT_INDICES: {
         /* (primitiveIndex, i0, i1, i2) -- always three, even for
          * lines and points; the unused lanes are ignored downstream.
          */
         NEED(4);
         nir_def *base = MESH_PTR(D12_MESH_OFF_INDICES);
         nir_def *off =
            nir_imul_imm(b, nir_u2u32(b, get_def(ctx, p[0])), 12);
         for (unsigned i = 0; i < 3; i++) {
            nir_store_global(
               b, nir_u2u32(b, get_def(ctx, p[1 + i])),
               nir_iadd(b, base, nir_u2u64(b, nir_iadd_imm(b, off, i * 4))),
               .write_mask = 1, .align_mul = 4);
         }
         return true;
      }

      case D12_DXOP_GET_MESH_PAYLOAD:
         /* Returns a POINTER into the amplification payload, which the
          * shader then walks with getelementptr. It joins the pointer
          * model as a global-space pointer rather than an SSA value.
          */
         if (inst->result_value < ctx->mod->num_values) {
            ctx->ptr_space[inst->result_value] = D12_PTR_GLOBAL;
            ctx->ptr_base[inst->result_value] = 0;
            ctx->ptr_global[inst->result_value] =
               MESH_PTR(D12_MESH_OFF_PAYLOAD);
            if (inst->result_type < ctx->mod->num_types &&
                ctx->mod->types[inst->result_type].kind == D12_TYPE_POINTER) {
               ctx->ptr_pointee[inst->result_value] =
                  ctx->mod->types[inst->result_type].ptr.pointee;
            }
         }
         return true;

      case D12_DXOP_STORE_VERTEX_OUTPUT:
      case D12_DXOP_STORE_PRIMITIVE_OUTPUT: {
         /* (sigId, row, col, value, vertexOrPrimIndex) */
         NEED(5);
         uint64_t sig_id = 0, row = 0, col = 0;
         const_uint(ctx, p[0], &sig_id);
         const_uint(ctx, p[1], &row);
         if (!const_uint(ctx, p[2], &col)) {
            trans_fail(ctx, "dynamic component index on mesh output");
            return false;
         }
         if (sig_id + row >= D12_MESH_MAX_SLOTS) {
            trans_fail(ctx, "mesh output slot out of range");
            return false;
         }

         nir_def *val = get_def(ctx, p[3]);
         if (!val)
            return false;
         if (val->bit_size != 32)
            val = nir_f2f32(b, val);

         bool prim = opcode == D12_DXOP_STORE_PRIMITIVE_OUTPUT;
         nir_def *base =
            MESH_PTR(prim ? D12_MESH_OFF_PRIMITIVES : D12_MESH_OFF_VERTICES);

         /* index * (slots * 16) + (slot * 16) + col * 4 */
         nir_def *idx = nir_u2u32(b, get_def(ctx, p[4]));
         nir_def *off = nir_imul_imm(
            b, idx, D12_MESH_MAX_SLOTS * D12_MESH_SLOT_STRIDE);
         off = nir_iadd_imm(b, off,
                            (sig_id + row) * D12_MESH_SLOT_STRIDE + col * 4);

         nir_store_global(b, val, nir_iadd(b, base, nir_u2u64(b, off)),
                          .write_mask = 1, .align_mul = 4);
         return true;
      }

      default: {
         /* dispatchMesh(x, y, z, payload): the amplification shader's
          * hand-off. Record the grid; the payload buffer itself is
          * written through the pointer operand by ordinary stores.
          */
         NEED(3);
         nir_def *d = MESH_PTR(D12_MESH_OFF_DISPATCH);
         for (unsigned i = 0; i < 3; i++) {
            nir_store_global(b, nir_u2u32(b, get_def(ctx, p[i])),
                             nir_iadd_imm(b, d, i * 4), .write_mask = 1,
                             .align_mul = 4);
         }
         return true;
      }
      }
#undef MESH_PTR
   }

   case D12_DXOP_DOMAIN_LOCATION: {
      /* (component) -> the barycentric/UV coordinate inside the patch. */
      NEED(1);
      uint64_t comp = 0;
      if (!const_uint(ctx, p[0], &comp)) {
         trans_fail(ctx, "dynamic domainLocation component");
         return false;
      }
      def_result(ctx, inst,
                 nir_channel(b, nir_load_tess_coord(b), (unsigned)comp & 3));
      return true;
   }

   case D12_DXOP_OUTPUT_CONTROL_POINT_ID:
      def_result(ctx, inst, nir_load_invocation_id(b));
      return true;

   case D12_DXOP_GS_INSTANCE_ID:
      def_result(ctx, inst, nir_load_invocation_id(b));
      return true;

   case D12_DXOP_EMIT_STREAM:
   case D12_DXOP_CUT_STREAM:
   case D12_DXOP_EMIT_THEN_CUT_STREAM: {
      /* (streamId). Outputs buffered by storeOutput are flushed as the
       * vertex, exactly as the end-of-shader flush does for other
       * stages -- emitting is a vertex boundary, not a new value.
       */
      NEED(1);
      uint64_t stream = 0;
      const_uint(ctx, p[0], &stream);

      if (opcode != D12_DXOP_CUT_STREAM) {
         flush_outputs(ctx);
         nir_emit_vertex(b, .stream_id = (unsigned)stream);
      }
      if (opcode != D12_DXOP_EMIT_STREAM)
         nir_end_primitive(b, .stream_id = (unsigned)stream);
      return true;
   }

   case D12_DXOP_LOAD_PATCH_CONSTANT: {
      /* (sigId, row, col) -> a per-patch input of the domain shader. */
      NEED(3);
      uint64_t sig_id = 0, row = 0, col = 0;
      const_uint(ctx, p[0], &sig_id);
      const_uint(ctx, p[1], &row);
      if (!const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic component index on loadPatchConstant");
         return false;
      }

      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->patch_sig, ctx->mod->num_patch_sig,
                           (uint32_t)sig_id);
      const char *sem = se ? se->name : NULL;
      unsigned location = patch_location(sem, (unsigned)sig_id);
      unsigned component = (unsigned)col;
      if (location != VARYING_SLOT_TESS_LEVEL_OUTER &&
          location != VARYING_SLOT_TESS_LEVEL_INNER)
         component = dxil_sig_component(se, component);
      if (component > 3) {
         trans_fail(ctx, "packed patch-input component outside register");
         return false;
      }

      def_result(ctx, inst,
                 nir_load_input(b, 1, 32, nir_imm_int(b, row),
                                .base = (unsigned)sig_id,
                                .component = component,
                                .dest_type = nir_type_float32,
                                .io_semantics.location = location,
                                .io_semantics.num_slots = 1));
      b->shader->info.patch_inputs_read |=
         BITFIELD64_BIT(location - VARYING_SLOT_PATCH0);
      return true;
   }

   case D12_DXOP_STORE_PATCH_CONSTANT: {
      /* (sigId, row, col, value) -> a per-patch output of the hull
       * shader, including the tessellation factors.
       */
      NEED(4);
      uint64_t sig_id = 0, row = 0, col = 0;
      const_uint(ctx, p[0], &sig_id);
      const_uint(ctx, p[1], &row);
      if (!const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic component index on storePatchConstant");
         return false;
      }

      nir_def *val = get_def(ctx, p[3]);
      if (!val)
         return false;
      if (val->bit_size != 32)
         val = nir_f2f32(b, val);

      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->patch_sig, ctx->mod->num_patch_sig,
                           (uint32_t)sig_id);
      const char *sem = se ? se->name : NULL;
      unsigned location = patch_location(sem, (unsigned)sig_id);

      /* Tess factors are indexed by ROW; generic patch constants by
       * their slot with the row selecting the slot.
       */
      unsigned slot = location;
      unsigned comp = (unsigned)col;
      if (location == VARYING_SLOT_TESS_LEVEL_OUTER ||
          location == VARYING_SLOT_TESS_LEVEL_INNER)
         comp = (unsigned)row;
      else {
         slot = location + (unsigned)row;
         comp = dxil_sig_component(se, comp);
      }
      if (comp > 3) {
         trans_fail(ctx, "packed patch-output component outside register");
         return false;
      }

      nir_store_output(b, val, nir_imm_int(b, 0), .base = (unsigned)sig_id,
                       .write_mask = 1, .component = comp,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
      /* patch_outputs_written is indexed FROM VARYING_SLOT_PATCH0, and
       * the tess-level slots sit below it -- the subtraction underflows
       * and the shift poisons the mask. util_last_bit then reports 28
       * patch constants where there are none, and
       * poly_tcs_out_offs_el pushes every per-vertex output out past a
       * 134-float stride. The tessellator still runs; the domain shader
       * just reads control points that were never written there.
       *
       * Tess levels live in outputs_written, which poly places at fixed
       * offsets 0..5 ahead of the generic constants.
       */
      if (slot >= VARYING_SLOT_PATCH0) {
         b->shader->info.patch_outputs_written |=
            BITFIELD64_BIT(slot - VARYING_SLOT_PATCH0);
      } else {
         b->shader->info.outputs_written |= BITFIELD64_BIT(slot);
      }
      return true;
   }

   case D12_DXOP_LOAD_OUTPUT_CONTROL_POINT: {
      /* (sigId, row, col, vertex) -> a per-vertex output of the hull
       * shader, read by the domain shader.
       */
      NEED(4);
      uint64_t sig_id = 0, row = 0, col = 0;
      const_uint(ctx, p[0], &sig_id);
      const_uint(ctx, p[1], &row);
      if (!const_uint(ctx, p[2], &col)) {
         trans_fail(ctx, "dynamic component index on "
                         "loadOutputControlPoint");
         return false;
      }
      nir_def *vertex = get_def(ctx, p[3]);
      if (!vertex)
         return false;

      const struct d12_sig_element *se =
         d12_dxil_find_sig(ctx->mod->output_sig, ctx->mod->num_output_sig,
                           (uint32_t)sig_id);
      unsigned location = se && se->name && !strcmp(se->name, "SV_Position")
                             ? VARYING_SLOT_POS
                             : dxil_generic_location(se, (unsigned)sig_id);
      unsigned component =
         location == VARYING_SLOT_POS
            ? (unsigned)col
            : dxil_sig_component(se, (unsigned)col);
      if (component > 3) {
         trans_fail(ctx,
                    "packed output-control-point component outside register");
         return false;
      }
      def_result(ctx, inst,
                 nir_load_per_vertex_output(
                    b, 1, 32, nir_u2u32(b, vertex), nir_imm_int(b, row),
                    .base = dxil_sig_start_row(se, (unsigned)sig_id),
                    .component = component,
                    .dest_type = nir_type_float32,
                    .io_semantics.location = location,
                    .io_semantics.num_slots = 1));
      b->shader->info.inputs_read |= BITFIELD64_BIT(location);
      return true;
   }

   case D12_DXOP_FLATTENED_THREAD_ID_IN_GROUP:
      def_result(ctx, inst, nir_load_local_invocation_index(b));
      return true;

   /*
    * Resources. The binding model (campaign OQ-2): each bound resource's
    * 64-bit base address lives in AGX uniform registers, four 16-bit
    * registers per resource, starting at D12_UNIFORM_RESOURCE_BASE. The
    * shader reads them with load_preamble; the driver publishes them with
    * agx_usc_uniform. createHandle therefore resolves to nothing more than
    * its binding index, and the address arithmetic happens at the access.
    */
   case D12_DXOP_CREATE_HANDLE: {
      /* (resourceClass, rangeId, index, nonUniformIndex) */
      NEED(3);
      uint64_t klass = 0, range_id = 0, index = 0;
      if (!const_uint(ctx, p[0], &klass) ||
          !const_uint(ctx, p[1], &range_id)) {
         trans_fail(ctx, "dynamic resource range index unsupported");
         return false;
      }
      if (!const_uint(ctx, p[2], &index)) {
         /* A resource array indexed by a computed value. The range's
          * registers map to consecutive slots, so the handle is
          * base_slot + (index - lower_bound); DXC's index operand is the
          * absolute shader register (it emits the add against the array
          * base itself).
          */
         const struct d12_resource_binding *rb = NULL;
         for (unsigned i = 0; i < ctx->mod->num_resources; i++) {
            const struct d12_resource_binding *cand =
               &ctx->mod->resources[i];
            if (cand->klass == klass && cand->id == range_id) {
               rb = cand;
               break;
            }
         }
         if (!rb) {
            trans_fail(ctx, "dynamic index into an undeclared range");
            return false;
         }

         unsigned count = d12_range_count(rb);

         unsigned rooted_reg = 0;
         unsigned base_slot = ~0u;
         bool rooted_heap = unbounded_root_for_register(
            ctx->map, (unsigned)klass, rb->space, rb->lower_bound,
            &base_slot, &rooted_reg);
         if (!rooted_heap)
            base_slot = d12_binding_map_find(
               ctx->map, (unsigned)klass, rb->space, rb->lower_bound);
         if (base_slot == ~0u) {
            mesa_loge("dxil: dynamic handle: class %u space %u register "
                      "%u (range %u) unmapped",
                      (unsigned)klass, rb->space, rb->lower_bound,
                      (unsigned)range_id);
            trans_fail(ctx, "shader uses a resource the root signature "
                            "does not bind");
            return false;
         }

         if (count && (count - 1 > UINT32_MAX - rb->lower_bound ||
                       count - 1 > UINT32_MAX - base_slot)) {
            trans_fail(ctx, "resource range overflows its register or slot");
            return false;
         }
         if (count && !rooted_heap) {
            unsigned last_slot = d12_binding_map_find(
               ctx->map, (unsigned)klass, rb->space,
               rb->lower_bound + count - 1);
            unsigned root_range = d12_binding_map_root_range(
               ctx->map, (unsigned)klass, rb->space, rb->lower_bound);
            if (last_slot != base_slot + count - 1 ||
                root_range == ~0u ||
                d12_binding_map_root_range(
                   ctx->map, (unsigned)klass, rb->space,
                   rb->lower_bound + count - 1) != root_range) {
               /* The stride/kind shortcut and the slot arithmetic both
                * require one contiguous API descriptor range. Adjacent root
                * ranges do not become one just because their register and
                * slot numbers happen to touch: a sized shader array cannot
                * spill across that boundary.
                */
               trans_fail(ctx, "shader resource array crosses root "
                               "descriptor ranges");
               return false;
            }
         }
         unsigned finite_count = 0;
         if (!count && !rooted_heap) {
            finite_count = finite_root_range_count(
               ctx->map, (unsigned)klass, rb->space, rb->lower_bound,
               base_slot);
            if (!finite_count) {
               mesa_loge("dxil: unbounded shader range class %u space %u "
                         "register %u has no contiguous API descriptor "
                         "range at root slot %u",
                         (unsigned)klass, rb->space, rb->lower_bound,
                         base_slot);
               trans_fail(ctx, "unbounded shader range is not backed by a "
                               "contiguous root range");
               return false;
            }
         }

         nir_def *idx = get_def(ctx, p[2]);
         if (!idx)
            return false;

         nir_def *index32 = nir_u2u32(b, idx);
         nir_def *relative = nir_iadd_imm(b, index32, -rb->lower_bound);
         nir_def *valid =
            nir_uge(b, index32, nir_imm_int(b, rb->lower_bound));
         if (count)
            valid = nir_iand(b, valid,
                             nir_ult(b, relative, nir_imm_int(b, count)));

         nir_def *slot_def;
         if (rooted_heap) {
            slot_def = nir_iadd_imm(b, index32, -rooted_reg);
            unsigned entry = d12_binding_map_find_entry(
               ctx->map, (unsigned)klass, rb->space, rooted_reg);
            if (entry == ~0u) {
               trans_fail(ctx, "rooted resource array has no API range");
               return false;
            }
            unsigned root_count = ctx->map->entries[entry].range_count;
            if (root_count)
               valid = nir_iand(b, valid,
                                nir_ult(b, slot_def,
                                        nir_imm_int(b, root_count)));
         } else {
            slot_def = nir_iadd_imm(b, relative, base_slot);
            if (!count)
               valid = nir_iand(b, valid,
                                nir_ult(b, relative,
                                        nir_imm_int(b, finite_count)));
         }

         /* Avoid wrapped/out-of-range root-table lookup even before the
          * resource operation consumes validity for robust buffer access. */
         slot_def = nir_bcsel(b, valid, slot_def,
                              nir_imm_int(b, rooted_heap ? 0 : base_slot));
         ctx->handle_valid[inst->result_value] = valid;

         ctx->handle_base[inst->result_value] = base_slot;
         if (rooted_heap) {
            ctx->handle_heap[inst->result_value] = 3;
            ctx->handle_kind_v[inst->result_value] = rb->kind;
            ctx->handle_stride_v[inst->result_value] = rb->stride_B;
         }
         def_result(ctx, inst, slot_def);
         return true;
      }

      /* A constant access into a rooted range uses the same rooted
       * descriptor-array handle as the dynamic case above.  It must not be
       * mistaken for base_slot + index in the finite root table. */
      for (unsigned i = 0; i < ctx->mod->num_resources; i++) {
         const struct d12_resource_binding *rb = &ctx->mod->resources[i];
         if (rb->klass != klass || rb->id != range_id)
            continue;

         unsigned base_slot = 0, rooted_reg = 0;
         if (!unbounded_root_for_register(
                ctx->map, (unsigned)klass, rb->space, rb->lower_bound,
                &base_slot, &rooted_reg))
            continue;

         unsigned count = d12_range_count(rb);
         if (index > UINT32_MAX || index < rb->lower_bound ||
             (count && index - rb->lower_bound >= count)) {
            trans_fail(ctx, "resource index is outside its declared range");
            return false;
         }
         unsigned reg = (unsigned)index;
         unsigned checked_slot = 0, checked_reg = 0;
         if (!unbounded_root_for_register(
                ctx->map, (unsigned)klass, rb->space, reg,
                &checked_slot, &checked_reg) ||
             checked_slot != base_slot || checked_reg != rooted_reg) {
            trans_fail(ctx, "resource index is outside its API root range");
            return false;
         }

         ctx->handle_base[inst->result_value] = base_slot;
         ctx->handle_heap[inst->result_value] = 3;
         ctx->handle_kind_v[inst->result_value] = rb->kind;
         ctx->handle_stride_v[inst->result_value] = rb->stride_B;
         def_result(ctx, inst,
                    nir_imm_int(b, reg - rooted_reg));
         return true;
      }

      if (index > UINT32_MAX || klass > UINT32_MAX || range_id > UINT32_MAX) {
         trans_fail(ctx, "resource handle operand exceeds 32 bits");
         return false;
      }
      unsigned slot;
      if (!resolve_binding_slot(ctx, (unsigned)klass, (unsigned)range_id,
                                (unsigned)index, &slot))
         return false;

      /* The handle IS the binding slot for our model. */
      def_result(ctx, inst, nir_imm_int(b, slot));
      return true;
   }

   case D12_DXOP_ATOMIC_BINOP: {
      /* (handle, atomicOp, offset0, offset1, offset2, value)
       *
       * The three offsets address the resource the same way a load or
       * store does; a buffer uses offset0 and, when structured, offset1.
       */
      NEED(6);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      if (h.rooted_heap &&
          h.kind != D12_RES_KIND_TYPED_BUFFER &&
          h.kind != D12_RES_KIND_RAW_BUFFER &&
          h.kind != D12_RES_KIND_STRUCTURED_BUFFER) {
         trans_fail(ctx, "heap-indexed texture atomics unsupported");
         return false;
      }

      uint64_t dx_op = 0;
      if (!const_uint(ctx, p[1], &dx_op)) {
         trans_fail(ctx, "non-constant atomic operation");
         return false;
      }

      nir_atomic_op op;
      switch (dx_op) {
      case 0: op = nir_atomic_op_iadd; break;
      case 1: op = nir_atomic_op_iand; break;
      case 2: op = nir_atomic_op_ior; break;
      case 3: op = nir_atomic_op_ixor; break;
      case 4: op = nir_atomic_op_imin; break;
      case 5: op = nir_atomic_op_imax; break;
      case 6: op = nir_atomic_op_umin; break;
      case 7: op = nir_atomic_op_umax; break;
      case 8: op = nir_atomic_op_xchg; break;
      default:
         trans_fail(ctx, "unhandled atomic operation");
         return false;
      }

      bool image = h.kind == D12_RES_KIND_TEXTURE_2D ||
                   h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY;
      if (!d12_native_pbe_software_addressing_supported(
             ctx->descriptor_generation) &&
          d12_texture_shape_from_kind(h.kind) != D12_TEXTURE_SHAPE_NONE) {
         trans_fail(ctx, "texture atomics require unproven AGX3 PBE "
                         "software addressing");
         return false;
      }
      if (image) {
         nir_def *zero = nir_imm_int(b, 0);
         nir_def *handle = native_image_handle(ctx, &h, true);
         nir_def *coord =
            nir_vec4(b, get_def(ctx, p[2]), get_def(ctx, p[3]),
                     h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY
                        ? get_def(ctx, p[4])
                        : zero,
                     zero);

         def_result(
            ctx, inst,
            nir_bindless_image_atomic(
               b, 32, handle, coord, zero, get_def(ctx, p[5]),
               .atomic_op = op,
               .image_dim = GLSL_SAMPLER_DIM_2D,
               .image_array = h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY,
               .format = op == nir_atomic_op_imin ||
                                op == nir_atomic_op_imax
                            ? PIPE_FORMAT_R32_SINT
                            : PIPE_FORMAT_R32_UINT,
               .access = 0));
         return true;
      }

      nir_def *accessible;
      nir_def *addr = buffer_atomic_address(
         ctx, &h, get_def(ctx, p[2]), get_def(ctx, p[3]), 32,
         &accessible);
      nir_def *result = nir_global_atomic(
         b, 32, addr, get_def(ctx, p[5]), .atomic_op = op);
      def_result(ctx, inst,
                 nir_bcsel(b, accessible, result, nir_imm_int(b, 0)));
      return true;
   }

   case D12_DXOP_ATOMIC_CMPXCHG: {
      /* (handle, offset0, offset1, offset2, compareValue, newValue) */
      NEED(6);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      if (h.rooted_heap &&
          h.kind != D12_RES_KIND_TYPED_BUFFER &&
          h.kind != D12_RES_KIND_RAW_BUFFER &&
          h.kind != D12_RES_KIND_STRUCTURED_BUFFER) {
         trans_fail(ctx, "heap-indexed texture atomics unsupported");
         return false;
      }

      unsigned bits = type_bit_size(ctx->mod, inst->result_type);
      if (bits != 32 && bits != 64)
         bits = 32;

      bool image = h.kind == D12_RES_KIND_TEXTURE_2D ||
                   h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY;
      if (!d12_native_pbe_software_addressing_supported(
             ctx->descriptor_generation) &&
          d12_texture_shape_from_kind(h.kind) != D12_TEXTURE_SHAPE_NONE) {
         trans_fail(ctx, "texture atomics require unproven AGX3 PBE "
                         "software addressing");
         return false;
      }
      if (image) {
         if (bits != 32) {
            trans_fail(ctx, "64-bit texture atomics unsupported");
            return false;
         }

         nir_def *zero = nir_imm_int(b, 0);
         nir_def *handle = native_image_handle(ctx, &h, true);
         nir_def *coord =
            nir_vec4(b, get_def(ctx, p[1]), get_def(ctx, p[2]),
                     h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY
                        ? get_def(ctx, p[3])
                        : zero,
                     zero);

         def_result(
            ctx, inst,
            nir_bindless_image_atomic_swap(
               b, 32, handle, coord, zero, get_def(ctx, p[4]),
               get_def(ctx, p[5]), .atomic_op = nir_atomic_op_cmpxchg,
               .image_dim = GLSL_SAMPLER_DIM_2D,
               .image_array = h.kind == D12_RES_KIND_TEXTURE_2D_ARRAY,
               .format = PIPE_FORMAT_R32_UINT, .access = 0));
         return true;
      }

      nir_def *accessible;
      nir_def *addr = buffer_atomic_address(
         ctx, &h, get_def(ctx, p[1]), get_def(ctx, p[2]), bits,
         &accessible);
      nir_def *result = nir_global_atomic_swap(
         b, bits, addr, get_def(ctx, p[4]), get_def(ctx, p[5]),
         .atomic_op = nir_atomic_op_cmpxchg);
      def_result(ctx, inst,
                 nir_bcsel(b, accessible, result,
                           nir_imm_intN_t(b, 0, bits)));
      return true;
   }

   case D12_DXOP_SAMPLE:
   case D12_DXOP_SAMPLE_BIAS:
   case D12_DXOP_SAMPLE_LEVEL:
   case D12_DXOP_SAMPLE_GRAD:
   case D12_DXOP_SAMPLE_CMP:
   case D12_DXOP_SAMPLE_CMP_LEVEL_ZERO:
   case D12_DXOP_SAMPLE_CMP_LEVEL:
   case D12_DXOP_SAMPLE_CMP_BIAS:
   case D12_DXOP_SAMPLE_CMP_GRAD:
   case D12_DXOP_TEXTURE_GATHER:
   case D12_DXOP_TEXTURE_GATHER_RAW: {
      /* (srv, sampler, c0..c3, offsets..., extras...). Only the 2D
       * layouts the texture heap can describe are accepted; everything
       * else refuses by name rather than sampling something that is not
       * there.
       *
       * AGX samples through a hardware sampler descriptor exactly as it
       * fetches through a texture descriptor: both handles are
       * base-uniform-plus-byte-offset pairs (agx_translate_bindless_handle),
       * so the sampler heap mirrors the texture heap at
       * D12_UNIFORM_SAMPLERS.
       */
      NEED(6);

      struct handle_ref th, sh;
      if (!handle_info(ctx, p[0], &th) || !handle_info(ctx, p[1], &sh))
         return false;

      nir_def *x = get_def(ctx, p[2]);
      nir_def *y = get_def(ctx, p[3]);
      if (!x || !y) {
         trans_fail(ctx, "malformed sample coordinates");
         return false;
      }

      /* Sample offsets, constant or computed: agx_nir_lower_texture
       * packs a nir_tex_src_offset into the instruction's nibble-packed
       * backend2 source, so passing the vector through IS the support.
       * Undef operands mean "no offset on this axis".
       */
      nir_def *offset = NULL;
      {
         nir_def *ox = NULL, *oy = NULL;
         for (unsigned i = 0; i < 2 && 6 + i < np; i++) {
            if (operand_is_undef(ctx, p[6 + i]))
               continue;
            uint64_t c = 0;
            if (const_uint(ctx, p[6 + i], &c) && c == 0)
               continue;
            nir_def *d = get_def(ctx, p[6 + i]);
            if (!d)
               continue;
            if (i == 0)
               ox = d;
            else
               oy = d;
         }
         if (ox || oy) {
            offset = nir_vec2(b, ox ? ox : nir_imm_int(b, 0),
                              oy ? oy : nir_imm_int(b, 0));
         }
      }

      /* DXC truncates trailing undef operands from the fixed signature
       * (a 1D gradient sample legitimately ends early); reads past np
       * mean "absent", value zero.
       */
#define TAILD(i)                                                            \
   ((i) < np && !operand_is_undef(ctx, p[i]) ? get_def(ctx, p[i])           \
                                             : nir_imm_float(b, 0.0f))

      nir_texop op;
      unsigned num_srcs = 3;
      switch (opcode) {
      case D12_DXOP_SAMPLE: op = nir_texop_tex; break;
      case D12_DXOP_SAMPLE_BIAS: op = nir_texop_txb; num_srcs = 4; break;
      case D12_DXOP_SAMPLE_LEVEL: op = nir_texop_txl; num_srcs = 4; break;
      case D12_DXOP_SAMPLE_GRAD: op = nir_texop_txd; num_srcs = 5; break;
      case D12_DXOP_SAMPLE_CMP: op = nir_texop_tex; num_srcs = 4; break;
      case D12_DXOP_SAMPLE_CMP_LEVEL_ZERO:
      case D12_DXOP_SAMPLE_CMP_LEVEL:
         /* Comparison at an explicit level is txl; level zero is the
          * same instruction with a zero lod.
          */
         op = nir_texop_txl;
         num_srcs = 5;
         break;
      case D12_DXOP_SAMPLE_CMP_BIAS:
         op = nir_texop_txb;
         num_srcs = 5;
         break;
      case D12_DXOP_SAMPLE_CMP_GRAD:
         op = nir_texop_txd;
         num_srcs = 6;
         break;
      default: op = nir_texop_tg4; break;
      }

      nir_def *tex_handle = native_image_handle(ctx, &th, false);

      /* Finite/static samplers stay in the per-stage USC state file. Heap
       * handles instead name the command's global sampler heap and carry
       * both the packed-descriptor byte offset and hardware index. */
      bool heap_samp = sh.heap;
      bool const_samp = !heap_samp &&
                        nir_src_is_const(nir_src_for_ssa(sh.slot));

      /* num_srcs was counted with the sampler as a source; a constant
       * sampler is an index attribute instead, not a source.
       */
      nir_tex_instr *tex = nir_tex_instr_create(
         b->shader,
         num_srcs - 1 + (const_samp ? 0 : 1) + (offset ? 1 : 0));
      tex->op = op;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->is_array = false;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;

      unsigned si = 0;
      tex->src[si++] =
         nir_tex_src_for_ssa(nir_tex_src_texture_handle, tex_handle);
      if (heap_samp) {
         tex->src[si++] = nir_tex_src_for_ssa(
            nir_tex_src_sampler_handle,
            native_sampler_handle(ctx, &sh));
      } else if (const_samp) {
         tex->sampler_index = nir_src_as_uint(nir_src_for_ssa(sh.slot));
      } else {
         tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_sampler_offset,
                                              nir_u2u16(b, sh.slot));
      }
      tex->src[si++] =
         nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec2(b, x, y));
      if (offset)
         tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);

      if (opcode == D12_DXOP_SAMPLE_BIAS) {
         NEED(10);
         tex->src[si++] =
            nir_tex_src_for_ssa(nir_tex_src_bias, get_def(ctx, p[9]));
      } else if (opcode == D12_DXOP_SAMPLE_LEVEL) {
         NEED(10);
         tex->src[si++] =
            nir_tex_src_for_ssa(nir_tex_src_lod, get_def(ctx, p[9]));
      } else if (opcode == D12_DXOP_SAMPLE_GRAD) {
         /* The trailing clamp may be omitted entirely. */
         NEED(14);
         tex->src[si++] = nir_tex_src_for_ssa(
            nir_tex_src_ddx,
            nir_vec2(b, get_def(ctx, p[9]), get_def(ctx, p[10])));
         tex->src[si++] = nir_tex_src_for_ssa(
            nir_tex_src_ddy,
            nir_vec2(b, get_def(ctx, p[12]), get_def(ctx, p[13])));
      } else if (opcode == D12_DXOP_SAMPLE_CMP ||
                 opcode == D12_DXOP_SAMPLE_CMP_LEVEL_ZERO ||
                 opcode == D12_DXOP_SAMPLE_CMP_LEVEL ||
                 opcode == D12_DXOP_SAMPLE_CMP_BIAS) {
         NEED(10);
         tex->is_shadow = true;
         tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                              get_def(ctx, p[9]));
         if (opcode == D12_DXOP_SAMPLE_CMP_LEVEL_ZERO) {
            tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                                 nir_imm_int(b, 0));
         } else if (opcode == D12_DXOP_SAMPLE_CMP_LEVEL) {
            NEED(11);
            tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                                 get_def(ctx, p[10]));
         } else if (opcode == D12_DXOP_SAMPLE_CMP_BIAS) {
            NEED(11);
            tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_bias,
                                                 get_def(ctx, p[10]));
         }
      } else if (opcode == D12_DXOP_SAMPLE_CMP_GRAD) {
         /* (srv, samp, c0..c3, off0..2, cmp, ddx0..2, ddy0..2[, clamp]),
          * with the same trailing truncation as sampleGrad.
          */
         NEED(12);
         tex->is_shadow = true;
         tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                              get_def(ctx, p[9]));
         tex->src[si++] = nir_tex_src_for_ssa(
            nir_tex_src_ddx, nir_vec2(b, TAILD(10), TAILD(11)));
         tex->src[si++] = nir_tex_src_for_ssa(
            nir_tex_src_ddy, nir_vec2(b, TAILD(13), TAILD(14)));
      } else if (opcode == D12_DXOP_TEXTURE_GATHER_RAW) {
         /* Raw gather: the texel bits, unconverted, channel 0. The
          * result width follows the declared overload.
          */
         tex->dest_type =
            nir_type_uint | type_bit_size(ctx->mod, inst->result_type);
         tex->component = 0;
      } else if (opcode == D12_DXOP_TEXTURE_GATHER) {
         NEED(9);
         uint64_t channel = 0;
         const_uint(ctx, p[8], &channel);
         tex->component = (unsigned)channel & 3;
      }

#undef TAILD
      /* Optional clamp positions match Mesa's MIT DXIL writer's sample
       * prototypes. DXC may truncate trailing undef operands. */
      unsigned clamp_index = UINT_MAX;
      switch (opcode) {
      case D12_DXOP_SAMPLE: clamp_index = 9; break;
      case D12_DXOP_SAMPLE_BIAS:
      case D12_DXOP_SAMPLE_CMP: clamp_index = 10; break;
      case D12_DXOP_SAMPLE_GRAD: clamp_index = 15; break;
      case D12_DXOP_SAMPLE_CMP_BIAS: clamp_index = 11; break;
      case D12_DXOP_SAMPLE_CMP_GRAD: clamp_index = 16; break;
      default: break;
      }
      nir_def *minimum = clamp_index < np &&
                         !operand_is_undef(ctx, p[clamp_index])
                            ? get_def(ctx, p[clamp_index]) : NULL;
      /* The resource descriptor decides whether memory is actually sparse,
       * but DXIL handles are late-bound. Always request the hardware feedback
       * lane: dense descriptors report resident, while AGX sparse descriptors
       * report each sampled page exactly. */
      tex->is_sparse = true;
      nir_def_init(&tex->instr, &tex->def,
                   nir_tex_instr_dest_size(tex), 32);
      nir_builder_instr_insert(b, &tex->instr);
      if (minimum)
         d12_nir_sample_min_lod(b, tex, minimum);

      /* DXIL sample/gather results are {vec4, status}. AGX places its sparse
       * residency code after the four data components. */
      nir_def *with_status[5] = {
         nir_channel(b, &tex->def, 0), nir_channel(b, &tex->def, 1),
         nir_channel(b, &tex->def, 2), nir_channel(b, &tex->def, 3),
         dxil_sparse_feedback_status(b, nir_channel(b, &tex->def, 4)),
      };
      def_result(ctx, inst, nir_vec(b, with_status, 5));
      return true;
   }

   case D12_DXOP_BUFFER_UPDATE_COUNTER: {
      /* (handle, inc): the append/consume counter. IncrementCounter
       * returns the PRE-increment value and DecrementCounter the
       * POST-decrement one (D3D semantics), and a hardware atomic add
       * returns the pre-value: the decrement result is pre - 1.
       *
       * The counter's address lives beside the resource's: second array
       * of the root table for range handles, the descriptor's
       * counter_address field for heap handles.
       */
      NEED(2);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      uint64_t inc_raw = 1;
      if (!const_uint(ctx, p[1], &inc_raw)) {
         trans_fail(ctx, "non-constant counter direction");
         return false;
      }
      bool dec = (int8_t)inc_raw < 0;

      nir_def *counter_addr;
      if (h.heap) {
         counter_addr = nir_load_global(
            b, 1, 64,
            nir_iadd_imm(b, descriptor_address(ctx, &h),
                         D12_DESC_COUNTER_OFFSET),
            .align_mul = 8);
      } else {
         nir_def *counter_table = root_plane(ctx, D12_ROOT_PLANE_COUNTER);
         counter_addr = nir_load_global(
            b, 1, 64,
            nir_iadd(b, counter_table,
                     nir_imul_imm(b, nir_u2u64(b, h.slot), 8)),
            .align_mul = 8);
      }

      if (h.valid)
         counter_addr = nir_bcsel(b, h.valid, counter_addr,
                                  root_plane(ctx, D12_ROOT_PLANE_DISCARD));

      nir_def *pre = nir_global_atomic(b, 32, counter_addr,
                                       nir_imm_int(b, dec ? -1 : 1),
                                       .atomic_op = nir_atomic_op_iadd);
      nir_def *result = dec ? nir_iadd_imm(b, pre, -1) : pre;
      if (h.valid)
         result = nir_bcsel(b, h.valid, result, nir_imm_int(b, 0));
      def_result(ctx, inst, result);
      return true;
   }

   case D12_DXOP_CHECK_ACCESS_FULLY_MAPPED:
      /* (status) -> bool. DXIL defines zero as not fully mapped and the
       * texture/buffer paths above use all bits set for fully mapped. */
      NEED(1);
      def_result(ctx, inst, nir_ine_imm(b, get_def(ctx, p[0]), 0));
      return true;

   case D12_DXOP_GET_DIMENSIONS: {
      /* (handle, mipLevel) -> (width, height, depth/layers, mip count),
       * consumed by extractvalue. txs and query_levels both lower to
       * reads of the descriptor in memory (agx_nir_lower_texture), which
       * is exactly where our heap keeps it.
       */
      NEED(2);

      struct handle_ref th;
      if (!handle_info(ctx, p[0], &th))
         return false;

      nir_def *lod = get_def(ctx, p[1]);
      if (!lod || operand_is_undef(ctx, p[1]))
         lod = nir_imm_int(b, 0);

      /* Buffers answer with their element count; the query goes to the
       * same descriptor with buffer dimension.
       */
      bool is_buf = th.kind == D12_RES_KIND_TYPED_BUFFER ||
                    th.kind == D12_RES_KIND_RAW_BUFFER ||
                    th.kind == D12_RES_KIND_STRUCTURED_BUFFER;
      bool is_1d = th.kind == D12_RES_KIND_TEXTURE_1D ||
                   th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY;
      bool is_array = th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY ||
                      th.kind == D12_RES_KIND_TEXTURE_2D_ARRAY;

      /* Both finite root tables and shader-visible heaps expose the same
       * count/stride metadata. ByteAddressBuffer reports its byte size;
       * structured and typed buffers report NumElements. */
      if (is_buf) {
         nir_def *count, *stride;
         resource_view_info(ctx, &th, &count, &stride);
         nir_def *dimension =
            th.kind == D12_RES_KIND_RAW_BUFFER
               ? nir_bcsel(b, nir_ine_imm(b, stride, 0),
                           nir_imul(b, count, stride), count)
               : count;
         def_result(
            ctx, inst,
            th.kind == D12_RES_KIND_STRUCTURED_BUFFER
               ? nir_vec4(b, count, stride, nir_imm_int(b, 0),
                          nir_imm_int(b, 1))
               : nir_vec4(b, dimension, nir_imm_int(b, 1),
                          nir_imm_int(b, 1), nir_imm_int(b, 1)));
         return true;
      }

      nir_def *handle = native_image_handle(ctx, &th, false);

      /* A buffer's size query is one component (its element count), a
       * 2D texture's is two, and a 2D array also returns the layer count.
       */
      unsigned txs_comps = is_buf     ? 1
                           : is_1d    ? (is_array ? 2 : 1)
                           : is_array ? 3
                                      : 2;

      nir_tex_instr *txs = nir_tex_instr_create(b->shader, 2);
      txs->op = nir_texop_txs;
      txs->sampler_dim = is_buf  ? GLSL_SAMPLER_DIM_BUF
                         : is_1d ? GLSL_SAMPLER_DIM_1D
                                 : GLSL_SAMPLER_DIM_2D;
      txs->is_array = is_array;
      txs->dest_type = nir_type_uint32;
      txs->src[0] =
         nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
      txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
      nir_def_init(&txs->instr, &txs->def, txs_comps, 32);
      nir_builder_instr_insert(b, &txs->instr);

      nir_tex_instr *lv = nir_tex_instr_create(b->shader, 1);
      lv->op = nir_texop_query_levels;
      lv->sampler_dim = is_1d ? GLSL_SAMPLER_DIM_1D : GLSL_SAMPLER_DIM_2D;
      lv->is_array = is_array;
      lv->dest_type = nir_type_uint32;
      lv->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_texture_handle, handle);
      nir_def_init(&lv->instr, &lv->def, 1, 32);
      nir_builder_instr_insert(b, &lv->instr);

      nir_def *one = nir_imm_int(b, 1);
      nir_def *width = nir_channel(b, &txs->def, 0);
      nir_def *second = is_1d ? (is_array ? nir_channel(b, &txs->def, 1) : one)
                              : (is_buf ? one : nir_channel(b, &txs->def, 1));
      nir_def *third = !is_1d && is_array ? nir_channel(b, &txs->def, 2) : one;

      /* AGX clamps an out-of-range TCS LOD to the last level.  D3D12
       * GetDimensions instead returns zero for every queried dimension while
       * retaining the resource's level count.  Keep DXIL aligned with the
       * DXBC RESINFO path above. */
      if (!is_buf) {
         nir_def *in_range = nir_ult(b, lod, &lv->def);
         nir_def *zero = nir_imm_int(b, 0);
         width = nir_bcsel(b, in_range, width, zero);
         second = nir_bcsel(b, in_range, second, zero);
         third = nir_bcsel(b, in_range, third, zero);
      }

      def_result(ctx, inst, nir_vec4(b, width, second, third, &lv->def));
      return true;
   }

   case D12_DXOP_START_VERTEX_LOCATION:
   case D12_DXOP_START_INSTANCE_LOCATION:
      /* SM 6.8 extended command info, published per draw in reserved
       * uniforms (see D12_UNIFORM_DRAW). The prolog ABI's
       * AGX_ABI_VUNI_FIRST_VERTEX slots are mapped only by the
       * prolog-building side, so the main shader reads ours instead.
       */
      def_result(ctx, inst,
                 nir_load_preamble(b, 1, 32,
                                   .base = D12_UNIFORM_DRAW +
                                           (opcode ==
                                                  D12_DXOP_START_INSTANCE_LOCATION
                                               ? 2
                                               : 0)));
      return true;

   case D12_DXOP_WAVE_MATCH: {
      /* WaveMatch(val): the mask of lanes holding MY value. No single
       * instruction computes it, but a waterfall converges: each
       * iteration, every lane matching the first active lane's value
       * learns its mask and leaves. Termination is by construction --
       * at least the first lane exits every trip.
       *
       * The loop is structured NIR, so only build it in shaders that
       * are being built structured.
       */
      NEED(1);
      if (ctx->num_blocks > 1) {
         trans_fail(ctx, "waveMatch inside control flow unsupported");
         return false;
      }

      nir_def *val = get_def(ctx, p[0]);
      if (!val)
         return false;

      nir_def *reg = nir_decl_reg(b, 1, 32, 0);
      nir_push_loop(b);
      {
         nir_def *first = nir_read_first_invocation(b, val);
         nir_def *mine = nir_ieq(b, val, first);
         nir_def *mask = nir_ballot(b, 1, 32, mine);
         nir_push_if(b, mine);
         {
            nir_store_reg(b, mask, reg);
            nir_jump(b, nir_jump_break);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_loop(b, NULL);

      nir_def *zero = nir_imm_int(b, 0);
      def_result(ctx, inst,
                 nir_vec4(b, nir_load_reg(b, reg), zero, zero, zero));
      return true;
   }

   case D12_DXOP_TEXTURE_STORE_SAMPLE:
   case D12_DXOP_TEXTURE_STORE: {
      /* (uav, c0, c1, c2, v0..v3, mask[, sample]). Compiles against the
       * same heap offsets as loads; a correct RUNTIME needs writable
       * PBE descriptors in a parallel heap, which does not exist yet --
       * the texture read path has to land first. The multisampled form
       * carries the sample index last, and
       * agx_nir_lower_multisampled_image_store handles the rest.
       */
      NEED(9);

      struct handle_ref th;
      if (!handle_info(ctx, p[0], &th))
         return false;

      bool ms = opcode == D12_DXOP_TEXTURE_STORE_SAMPLE;
      if (ms && !d12_native_pbe_software_addressing_supported(
                   ctx->descriptor_generation)) {
         trans_fail(ctx, "multisampled image stores require unproven AGX3 "
                         "PBE software addressing");
         return false;
      }

      bool one_d = th.kind == D12_RES_KIND_TEXTURE_1D ||
                   th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY;
      bool array = th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY ||
                   th.kind == D12_RES_KIND_TEXTURE_2D_ARRAY;
      nir_def *cx = get_def(ctx, p[1]);
      nir_def *cy = one_d ? nir_imm_int(b, 0) : get_def(ctx, p[2]);
      if (!cx || !cy) {
         trans_fail(ctx, "malformed texture store coordinates");
         return false;
      }

      nir_def *handle = native_image_handle(ctx, &th, true);

      nir_def *vals[4];
      for (unsigned i = 0; i < 4; i++) {
         vals[i] = get_def(ctx, p[4 + i]);
         if (!vals[i] || operand_is_undef(ctx, p[4 + i]))
            vals[i] = nir_imm_int(b, 0);
      }

      nir_def *sample = nir_imm_int(b, 0);
      if (ms && np >= 10) {
         nir_def *si_def = get_def(ctx, p[9]);
         if (si_def)
            sample = si_def;
      }

      nir_def *zero = nir_imm_int(b, 0);
      nir_def *layer = !array  ? zero
                       : one_d ? get_def(ctx, p[2])
                               : get_def(ctx, p[3]);
      if (!layer)
         layer = zero;
      nir_bindless_image_store(b, handle, nir_vec4(b, cx, cy, layer, zero),
                               sample, nir_vec(b, vals, 4), nir_imm_int(b, 0),
                               .image_dim = ms      ? GLSL_SAMPLER_DIM_MS
                                            : one_d ? GLSL_SAMPLER_DIM_1D
                                                    : GLSL_SAMPLER_DIM_2D,
                               .image_array = array, .format = PIPE_FORMAT_NONE,
                               .access = ACCESS_NON_READABLE);
      return true;
   }

   case D12_DXOP_TEXTURE_LOAD: {
      /*
       * textureLoad(handle, mipLevel, x, y, z, ox, oy, oz)
       *
       * The hardware takes a 64-bit uniform BASE plus a 32-bit register
       * OFFSET and adds them, so the handle is a byte offset into the
       * texture heap and `desc_set` names the uniform holding its base --
       * not a Vulkan descriptor set.
       */
      NEED(4);

      struct handle_ref th;
      if (!handle_info(ctx, p[0], &th))
         return false;

      bool one_d = th.kind == D12_RES_KIND_TEXTURE_1D ||
                   th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY;
      bool array = th.kind == D12_RES_KIND_TEXTURE_1D_ARRAY ||
                   th.kind == D12_RES_KIND_TEXTURE_2D_ARRAY;
      nir_def *x = get_def(ctx, p[2]);
      nir_def *y = one_d ? NULL : get_def(ctx, p[3]);
      if (!x || (!one_d && !y)) {
         trans_fail(ctx, "malformed texture coordinates");
         return false;
      }

      /* Texel-fetch offsets are integers added to integer coordinates:
       * folding them here IS the operation, no hardware support needed.
       * Undef means absent. Dynamic offsets stay refused.
       */
      for (unsigned i = 0; i < (one_d ? 1u : 2u) && 5 + i < np; i++) {
         if (operand_is_undef(ctx, p[5 + i]))
            continue;
         uint64_t off = 0;
         if (const_uint(ctx, p[5 + i], &off)) {
            if (!off)
               continue;
            nir_def **c = i ? &y : &x;
            *c = nir_iadd_imm(b, *c, (int32_t)(int8_t)off);
         } else {
            trans_fail(ctx, "dynamic texel-fetch offsets unsupported");
            return false;
         }
      }

      /* DXIL encodes the mip operand as undef for UAV texture loads.  Do not
       * turn that SSA undef into a real (and potentially huge) hardware LOD;
       * UAVs have no mip selection at the use site. */
      nir_def *lod = operand_is_undef(ctx, p[1]) ? NULL : get_def(ctx, p[1]);
      if (d12_debug()->dump_tex) {
         fprintf(stderr, "textureLoad: num_ops=%u lod=%s\n",
                 inst->num_ops, lod ? "yes" : "no");
      }
      nir_def *tex_handle = native_image_handle(ctx, &th, false);

      nir_tex_instr *tex = nir_tex_instr_create(b->shader, lod ? 3 : 2);
      tex->op = nir_texop_txf;
      /* Texture1D is represented by a height-one 2D AGX descriptor.  Use the
       * physical fetch dimension so explicit LOD and array-layer coordinates
       * survive backend lowering. */
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->is_array = array;
      tex->dest_type = nir_type_float32;
      tex->coord_components = array ? 3 : 2;

      nir_def *coords[3] = {x, one_d ? nir_imm_int(b, 0) : y, NULL};
      if (array)
         coords[2] = get_def(ctx, p[one_d ? 3 : 4]);
      for (unsigned i = 0; i < tex->coord_components; i++) {
         if (!coords[i]) {
            trans_fail(ctx, "malformed texture array coordinate");
            return false;
         }
      }

      tex->src[0] =
         nir_tex_src_for_ssa(nir_tex_src_texture_handle, tex_handle);
      tex->src[1] = nir_tex_src_for_ssa(
         nir_tex_src_coord, nir_vec(b, coords, tex->coord_components));
      if (lod)
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);

      tex->is_sparse = true;
      nir_def_init(&tex->instr, &tex->def,
                   nir_tex_instr_dest_size(tex), 32);
      nir_builder_instr_insert(b, &tex->instr);

      /* TextureLoad returns the same hardware sparse feedback lane as sample
       * and gather. Dense image descriptors still report fully mapped. */
      nir_def *with_status[5] = {
         nir_channel(b, &tex->def, 0), nir_channel(b, &tex->def, 1),
         nir_channel(b, &tex->def, 2), nir_channel(b, &tex->def, 3),
         dxil_sparse_feedback_status(b, nir_channel(b, &tex->def, 4)),
      };
      def_result(ctx, inst, nir_vec(b, with_status, 5));
      return true;
   }

   case D12_DXOP_DERIV_COARSE_X:
      NEED(1);
      def_result(ctx, inst,
                 emit_dxil_derivative(ctx, get_def(ctx, p[0]),
                                      nir_intrinsic_ddx_coarse,
                                      dxil_value_is_cbuffer_data(ctx, p[0])));
      return true;
   case D12_DXOP_DERIV_COARSE_Y:
      NEED(1);
      def_result(ctx, inst,
                 emit_dxil_derivative(ctx, get_def(ctx, p[0]),
                                      nir_intrinsic_ddy_coarse,
                                      dxil_value_is_cbuffer_data(ctx, p[0])));
      return true;
   case D12_DXOP_DERIV_FINE_X:
      NEED(1);
      def_result(ctx, inst,
                 emit_dxil_derivative(ctx, get_def(ctx, p[0]),
                                      nir_intrinsic_ddx_fine,
                                      dxil_value_is_cbuffer_data(ctx, p[0])));
      return true;
   case D12_DXOP_DERIV_FINE_Y:
      NEED(1);
      def_result(ctx, inst,
                 emit_dxil_derivative(ctx, get_def(ctx, p[0]),
                                      nir_intrinsic_ddy_fine,
                                      dxil_value_is_cbuffer_data(ctx, p[0])));
      return true;

   case D12_DXOP_DOT2:
   case D12_DXOP_DOT3:
   case D12_DXOP_DOT4: {
      /* Scalarized operands: (ax.., bx..) pairs. */
      unsigned n = opcode - D12_DXOP_DOT2 + 2;
      NEED(2 * n);

      nir_def *acc = NULL;
      for (unsigned i = 0; i < n; i++) {
         nir_def *prod =
            nir_fmul(b, get_def(ctx, p[i]), get_def(ctx, p[n + i]));
         acc = acc ? nir_fadd(b, acc, prod) : prod;
      }
      def_result(ctx, inst, acc);
      return true;
   }

   case D12_DXOP_DOT2_ADD_HALF: {
      /* (acc f32, ax, ay, bx, by halves): products at half precision,
       * accumulation at single, per the SM 6.4 spec.
       */
      NEED(5);
      nir_def *acc = get_def(ctx, p[0]);
      nir_def *px =
         nir_fmul(b, get_def(ctx, p[1]), get_def(ctx, p[3]));
      nir_def *py =
         nir_fmul(b, get_def(ctx, p[2]), get_def(ctx, p[4]));
      def_result(ctx, inst,
                 nir_fadd(b, nir_fadd(b, acc, nir_f2f32(b, px)),
                          nir_f2f32(b, py)));
      return true;
   }

   case D12_DXOP_DOT4_ADD_I8_PACKED:
   case D12_DXOP_DOT4_ADD_U8_PACKED: {
      /* (acc, a, b): packed 8-bit dot product with accumulate. NIR has
       * both exactly.
       */
      NEED(3);
      nir_def *acc = get_def(ctx, p[0]);
      nir_def *a = get_def(ctx, p[1]);
      nir_def *bb = get_def(ctx, p[2]);
      def_result(ctx, inst,
                 opcode == D12_DXOP_DOT4_ADD_I8_PACKED
                    ? nir_sdot_4x8_iadd(b, a, bb, acc)
                    : nir_udot_4x8_uadd(b, a, bb, acc));
      return true;
   }

   case D12_DXOP_WAVE_MULTI_PREFIX_OP:
   case D12_DXOP_WAVE_MULTI_PREFIX_BIT_COUNT: {
      /* (value, mask0..mask3, op, sop) -- or (value, mask0..3) for the
       * bit-count form. The mask is a PER-LANE uint4 naming the lanes
       * that share this lane's group; the result folds the values of the
       * group members STRICTLY BEFORE this lane.
       *
       * There is no hardware masked scan, so it is unrolled across the
       * wave: for each lane i, take its value iff i is in my mask and
       * i < my lane. The shift amount is uniform, so read_invocation
       * lowers to a plain broadcast per step. Wave width here is 32 --
       * mask1..3 address lanes that do not exist on this hardware.
       */
      bool bitcount = opcode == D12_DXOP_WAVE_MULTI_PREFIX_BIT_COUNT;
      NEED(bitcount ? 5 : 7);

      nir_def *value = get_def(ctx, p[0]);
      nir_def *mask = get_def(ctx, p[1]);
      if (!value || !mask)
         return false;

      /* The bit-count form counts predicates; fold 0/1 with a sum. */
      uint64_t kind = 0;
      if (!bitcount && !const_uint(ctx, p[5], &kind)) {
         trans_fail(ctx, "non-constant wave multi-prefix operation");
         return false;
      }
      if (bitcount)
         value = nir_b2i32(b, to_bool(ctx, value));

      /* Kind numbering verified empirically against a shader calling the
       * six variants in a known order: Sum=0, And=1, Or=2, Xor=3,
       * Product=4. It is NOT dxil_wave_op_kind (Sum/Product/Min/Max) nor
       * dxil_wave_bit_op_kind -- WaveMultiPrefix has its own enum.
       */
      unsigned bits = value->bit_size;
      nir_def *acc;
      switch (bitcount ? 0 : kind) {
      case 0: acc = nir_imm_intN_t(b, 0, bits); break;        /* Sum */
      case 1: acc = nir_imm_intN_t(b, ~0ull, bits); break;    /* And */
      case 2: acc = nir_imm_intN_t(b, 0, bits); break;        /* Or */
      case 3: acc = nir_imm_intN_t(b, 0, bits); break;        /* Xor */
      case 4: acc = nir_imm_intN_t(b, 1, bits); break;        /* Product */
      default:
         trans_fail(ctx, "unhandled wave multi-prefix operation");
         return false;
      }

      nir_def *lane = nir_load_subgroup_invocation(b);

      for (unsigned i = 0; i < 32; i++) {
         nir_def *v = nir_read_invocation(b, value, nir_imm_int(b, i));
         nir_def *in_group =
            nir_ine_imm(b, nir_iand_imm(b, nir_ushr_imm(b, mask, i), 1), 0);
         nir_def *before =
            nir_ilt(b, nir_imm_int(b, i), nir_u2u32(b, lane));

         nir_def *folded;
         switch (bitcount ? 0 : kind) {
         case 0: folded = nir_iadd(b, acc, v); break;
         case 1: folded = nir_iand(b, acc, v); break;
         case 2: folded = nir_ior(b, acc, v); break;
         case 3: folded = nir_ixor(b, acc, v); break;
         default: folded = nir_imul(b, acc, v); break;
         }

         acc = nir_bcsel(b, nir_iand(b, in_group, before), folded, acc);
      }

      def_result(ctx, inst, acc);
      return true;
   }

   case D12_DXOP_WAVE_ACTIVE_BALLOT: {
      /* Returns a 4 x u32 mask. The wave is at most 32 wide here, so
       * lanes 32..127 do not exist and their words are zero -- the true
       * value, not padding.
       */
      NEED(1);
      nir_def *mask =
         nir_ballot(b, 1, 32, to_bool(ctx, get_def(ctx, p[0])));
      nir_def *zero = nir_imm_int(b, 0);
      def_result(ctx, inst, nir_vec4(b, mask, zero, zero, zero));
      return true;
   }

   case D12_DXOP_MSAD: {
      /* dx.op.tertiary.i32(50): masked sum of absolute differences.
       * NIR has the exact operation.
       */
      NEED(3);

      /* The AGX backend has no msad op, so expand: per byte, a zero
       * REFERENCE byte is masked out, otherwise accumulate |ref - src|.
       */
      nir_def *ref = get_def(ctx, p[0]);
      nir_def *src = get_def(ctx, p[1]);
      nir_def *acc = get_def(ctx, p[2]);

      for (unsigned i = 0; i < 4; i++) {
         nir_def *rb =
            nir_iand_imm(b, nir_ushr_imm(b, ref, i * 8), 0xff);
         nir_def *sb =
            nir_iand_imm(b, nir_ushr_imm(b, src, i * 8), 0xff);
         nir_def *diff = nir_iabs(b, nir_isub(b, rb, sb));
         nir_def *keep = nir_ine_imm(b, rb, 0);
         acc = nir_iadd(b, acc,
                        nir_bcsel(b, keep, diff, nir_imm_int(b, 0)));
      }
      def_result(ctx, inst, acc);
      return true;
   }

   case D12_DXOP_DISCARD: {
      /* (condition i1). demote rather than terminate: helpers stay alive
       * for derivatives, matching D3D's discard semantics. The AGX
       * backend consumes this only after agx_nir_lower_sample_mask.
       */
      NEED(1);
      nir_demote_if(b, to_bool(ctx, get_def(ctx, p[0])));
      /* agx_nir_lower_discard_zs_emit gates its demote lowering on this
       * flag; without it the raw demote reaches the backend and asserts.
       */
      b->shader->info.fs.uses_discard = true;
      return true;
   }

   case D12_DXOP_BARRIER: {
      /* (mode i32 const): bit 0 sync group, bit 1 UAV global fence,
       * bit 2 UAV group fence, bit 3 groupshared fence. From DXIL.rst's
       * BarrierMode, mirrored in DirectXShaderCompiler's DxilConstants.h.
       */
      NEED(1);
      uint64_t mode = 0;
      if (!const_uint(ctx, p[0], &mode)) {
         trans_fail(ctx, "non-constant barrier mode");
         return false;
      }

      nir_variable_mode modes = 0;
      if (mode & 0x2)
         modes |= nir_var_mem_global | nir_var_image;
      if (mode & 0x4)
         modes |= nir_var_mem_global | nir_var_image;
      if (mode & 0x8)
         modes |= nir_var_mem_shared;

      nir_barrier(b,
                  .execution_scope = (mode & 0x1) ? SCOPE_WORKGROUP
                                                  : SCOPE_NONE,
                  .memory_scope = (mode & 0x2) ? SCOPE_DEVICE
                                               : SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = modes);
      return true;
   }

   case D12_DXOP_SAMPLE_INDEX:
      def_result(ctx, inst, nir_load_sample_id(b));
      return true;

   case D12_DXOP_RENDER_TARGET_GET_SAMPLE_COUNT:
      def_result(ctx, inst,
                 nir_ishl(b, nir_imm_int(b, 1),
                          nir_u2u32(b, nir_load_samples_log2_agx(b))));
      return true;

   case D12_DXOP_TEXTURE2DMS_GET_SAMPLE_POSITION: {
      NEED(2);
      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;
      if (h.rooted_heap) {
         trans_fail(ctx, "heap-indexed sample-position query unsupported");
         return false;
      }

      nir_def *tex_handle = nir_bindless_image_agx(
         b, nir_imul_imm(b, h.slot, ctx->native_layout.texture_stride_B),
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
      nir_def *index = get_def(ctx, p[1]);
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
      nir_def *two = nir_bcsel(b, nir_ieq_imm(b, index, 1),
                               nir_imm_float(b, -0.25f),
                               nir_imm_float(b, 0.25f));
      x = nir_bcsel(b, nir_ieq_imm(b, count, 2), two, x);
      y = nir_bcsel(b, nir_ieq_imm(b, count, 2), two, y);
      x = nir_bcsel(b, nir_ieq_imm(b, count, 1),
                    nir_imm_float(b, 0.0f), x);
      y = nir_bcsel(b, nir_ieq_imm(b, count, 1),
                    nir_imm_float(b, 0.0f), y);
      def_result(ctx, inst, nir_vec2(b, x, y));
      return true;
   }

   case D12_DXOP_COVERAGE:
      def_result(ctx, inst, nir_load_sample_mask_in(b));
      return true;

   case D12_DXOP_INNER_COVERAGE:
      /* Conservative rasterization is not exposed (no tier reported), so
       * no PSO that requires InnerCoverage can be created; a shader that
       * reads it anyway sees "not fully covered", the conservative
       * answer.
       */
      def_result(ctx, inst, nir_imm_int(b, 0));
      return true;

   case D12_DXOP_PRIMITIVE_ID:
      def_result(ctx, inst, nir_load_primitive_id(b));
      return true;

   case D12_DXOP_VIEW_ID:
      /* View instancing is not exposed (CheckFeatureSupport reports no
       * tiers), so every view a shader can run in is view 0. This is the
       * true value, not a placeholder.
       */
      def_result(ctx, inst, nir_imm_int(b, 0));
      return true;

   case D12_DXOP_LEGACY_F32TOF16: {
      /* f32tof16 returns the half in the LOW 16 bits of a u32. Packing
       * against a zero high half gives exactly that.
       */
      NEED(1);
      def_result(ctx, inst,
                 nir_pack_half_2x16_split(b, get_def(ctx, p[0]),
                                          nir_imm_float(b, 0.0f)));
      return true;
   }

   case D12_DXOP_LEGACY_F16TOF32:
      NEED(1);
      def_result(ctx, inst,
                 nir_channel(b, nir_unpack_half_2x16(b, get_def(ctx, p[0])),
                             0));
      return true;

   case D12_DXOP_MAKE_DOUBLE:
      /* (lo, hi) -> the 64-bit value. Pure bit assembly: no fp64
       * arithmetic happens here, so this works on hardware with none.
       * Shaders that go on to ADD doubles still fail, in the backend,
       * honestly.
       */
      NEED(2);
      def_result(ctx, inst,
                 nir_pack_64_2x32_split(b, get_def(ctx, p[0]),
                                        get_def(ctx, p[1])));
      return true;

   case D12_DXOP_SPLIT_DOUBLE: {
      /* (double) -> {lo, hi}, consumed by extractvalue. */
      NEED(1);
      nir_def *v = get_def(ctx, p[0]);
      def_result(ctx, inst,
                 nir_vec2(b, nir_unpack_64_2x32_split_x(b, v),
                          nir_unpack_64_2x32_split_y(b, v)));
      return true;
   }

   case D12_DXOP_IS_HELPER_LANE:
      def_result(ctx, inst, nir_is_helper_invocation(b, 1));
      return true;

   case D12_DXOP_QUAD_VOTE: {
      /* (cond, op): 0 = any, 1 = all. */
      NEED(2);
      uint64_t qop = 0;
      const_uint(ctx, p[1], &qop);
      nir_def *cond = to_bool(ctx, get_def(ctx, p[0]));
      def_result(ctx, inst, qop ? nir_quad_vote_all(b, 1, cond)
                                : nir_quad_vote_any(b, 1, cond));
      return true;
   }

   case D12_DXOP_WAVE_PREFIX_BIT_COUNT:
      /* WavePrefixCountBits: how many lanes BEFORE mine hold true.
       * An exclusive scan of the 0/1 predicate is exactly that.
       */
      NEED(1);
      def_result(ctx, inst,
                 nir_exclusive_scan(
                    b, nir_b2i32(b, to_bool(ctx, get_def(ctx, p[0]))),
                    .reduction_op = nir_op_iadd));
      return true;

   case D12_DXOP_WAVE_ALL_BIT_COUNT:
      /* WaveActiveCountBits: how many active lanes hold true. */
      NEED(1);
      def_result(ctx, inst,
                 nir_bit_count(b, nir_ballot(b, 1, 32,
                                             to_bool(ctx, get_def(ctx, p[0])))));
      return true;

   case D12_DXOP_CALCULATE_LOD: {
      /* (srv, sampler, c0, c1, c2, clamped) -> the LOD the sample would
       * use. NIR's lod query returns (clamped, raw); the operand picks.
       */
      NEED(6);

      struct handle_ref th, sh;
      if (!handle_info(ctx, p[0], &th) || !handle_info(ctx, p[1], &sh))
         return false;

      if (th.rooted_heap || sh.rooted_heap) {
         trans_fail(ctx, "heap-indexed texture LOD queries unsupported");
         return false;
      }

      nir_def *x = get_def(ctx, p[2]);
      nir_def *y = get_def(ctx, p[3]);
      if (!x || !y) {
         trans_fail(ctx, "malformed calculateLOD coordinates");
         return false;
      }

      uint64_t clamped = 1;
      const_uint(ctx, p[5], &clamped);

      bool const_samp = nir_src_is_const(nir_src_for_ssa(sh.slot));

      nir_tex_instr *tex =
         nir_tex_instr_create(b->shader, const_samp ? 2 : 3);
      tex->op = nir_texop_lod;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->is_array = false;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;

      unsigned si = 0;
      tex->src[si++] = nir_tex_src_for_ssa(
         nir_tex_src_texture_handle,
         nir_bindless_image_agx(
            b, nir_imul_imm(b, th.slot,
                            ctx->native_layout.texture_stride_B),
            .desc_set = D12_UNIFORM_TEXTURES));
      if (const_samp) {
         tex->sampler_index = nir_src_as_uint(nir_src_for_ssa(sh.slot));
      } else {
         tex->src[si++] = nir_tex_src_for_ssa(nir_tex_src_sampler_offset,
                                              nir_u2u16(b, sh.slot));
      }
      tex->src[si++] =
         nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec2(b, x, y));
      nir_def_init(&tex->instr, &tex->def, 2, 32);
      nir_builder_instr_insert(b, &tex->instr);

      def_result(ctx, inst,
                 nir_channel(b, &tex->def, clamped ? 0 : 1));
      return true;
   }

   case D12_DXOP_QUAD_READ_LANE_AT: {
      NEED(2);
      def_result(ctx, inst,
                 nir_quad_broadcast(b, get_def(ctx, p[0]),
                                    get_def(ctx, p[1])));
      return true;
   }

   case D12_DXOP_QUAD_OP: {
      /* (value, op): 0 across X, 1 across Y, 2 diagonal, from DXIL.rst's
       * QuadOpKind.
       */
      NEED(2);
      uint64_t qop = 0;
      if (!const_uint(ctx, p[1], &qop)) {
         trans_fail(ctx, "non-constant quad op kind");
         return false;
      }

      nir_def *v = get_def(ctx, p[0]);
      switch (qop) {
      case 0: def_result(ctx, inst, nir_quad_swap_horizontal(b, v)); return true;
      case 1: def_result(ctx, inst, nir_quad_swap_vertical(b, v)); return true;
      case 2: def_result(ctx, inst, nir_quad_swap_diagonal(b, v)); return true;
      default:
         trans_fail(ctx, "unknown quad op kind");
         return false;
      }
   }

   case D12_DXOP_WAVE_GET_LANE_INDEX:
      def_result(ctx, inst, nir_load_subgroup_invocation(b));
      return true;

   case D12_DXOP_WAVE_GET_LANE_COUNT:
      def_result(ctx, inst, nir_load_subgroup_size(b));
      return true;

   case D12_DXOP_WAVE_IS_FIRST_LANE:
      def_result(ctx, inst, nir_elect(b, 1));
      return true;

   case D12_DXOP_WAVE_ANY_TRUE:
      NEED(1);
      def_result(ctx, inst, nir_vote_any(b, 1, to_bool(ctx, get_def(ctx, p[0]))));
      return true;

   case D12_DXOP_WAVE_ALL_TRUE:
      NEED(1);
      def_result(ctx, inst, nir_vote_all(b, 1, to_bool(ctx, get_def(ctx, p[0]))));
      return true;

   case D12_DXOP_WAVE_ACTIVE_ALL_EQUAL:
      NEED(1);
      if (p[0] >= ctx->mod->num_values) {
         trans_fail(ctx, "wave all-equal operand is out of range");
         return false;
      }
      def_result(ctx, inst,
                 type_is_float(ctx->mod, ctx->mod->values[p[0]].type)
                    ? nir_vote_feq(b, 1, get_def(ctx, p[0]))
                    : nir_vote_ieq(b, 1, get_def(ctx, p[0])));
      return true;

   case D12_DXOP_WAVE_READ_LANE_FIRST:
      NEED(1);
      def_result(ctx, inst, nir_read_first_invocation(b, get_def(ctx, p[0])));
      return true;

   case D12_DXOP_WAVE_READ_LANE_AT:
      NEED(2);
      def_result(ctx, inst,
                 nir_read_invocation(b, get_def(ctx, p[0]),
                                     get_def(ctx, p[1])));
      return true;

   case D12_DXOP_WAVE_ACTIVE_BIT: {
      /* DXIL WaveBitOpKind is And=0, Or=1, Xor=2. All three map directly
       * to NIR subgroup reductions and preserve the operand's integer width.
       */
      NEED(2);
      uint64_t dx_op = 0;
      if (!const_uint(ctx, p[1], &dx_op)) {
         trans_fail(ctx, "non-constant wave bit operation");
         return false;
      }

      nir_op op;
      switch (dx_op) {
      case 0: op = nir_op_iand; break;
      case 1: op = nir_op_ior; break;
      case 2: op = nir_op_ixor; break;
      default:
         trans_fail(ctx, "unhandled wave bit operation kind");
         return false;
      }

      def_result(ctx, inst,
                 nir_reduce(b, get_def(ctx, p[0]), .reduction_op = op));
      return true;
   }

   case D12_DXOP_WAVE_ALL_OP:
   case D12_DXOP_WAVE_PREFIX_OP: {
      /* (value, op, signedness). WaveActiveSum and friends reduce across
       * the wave; WavePrefixSum scans it exclusively.
       */
      NEED(3);

      uint64_t dx_op = 0, sign = 0;
      if (!const_uint(ctx, p[1], &dx_op) || !const_uint(ctx, p[2], &sign)) {
         trans_fail(ctx, "non-constant wave operation");
         return false;
      }

      nir_def *val = get_def(ctx, p[0]);
      bool is_float = val && val->bit_size >= 16 &&
                      type_is_float(ctx->mod, inst->result_type);
      bool is_signed = sign == 0;

      nir_op op;
      switch (dx_op) {
      case 0: op = is_float ? nir_op_fadd : nir_op_iadd; break; /* Sum */
      case 1: op = is_float ? nir_op_fmul : nir_op_imul; break; /* Product */
      case 2: /* Min */
         op = is_float ? nir_op_fmin : (is_signed ? nir_op_imin : nir_op_umin);
         break;
      case 3: /* Max */
         op = is_float ? nir_op_fmax : (is_signed ? nir_op_imax : nir_op_umax);
         break;
      default:
         trans_fail(ctx, "unhandled wave operation kind");
         return false;
      }

      if (opcode == D12_DXOP_WAVE_ALL_OP) {
         def_result(ctx, inst, nir_reduce(b, val, .reduction_op = op));
      } else {
         /* DXIL's WavePrefix* is exclusive. */
         def_result(ctx, inst, nir_exclusive_scan(b, val, .reduction_op = op));
      }
      return true;
   }

   case D12_DXOP_CREATE_HANDLE_FOR_LIB:
   case D12_DXOP_ANNOTATE_HANDLE: {
      /* (handle, resourceProperties) -> the same handle.
       *
       * SM 6.6 attaches the resource's type and dimensions to a handle so
       * a runtime can validate them; the handle itself is unchanged. It
       * appears in nearly every SM 6.6 shader that touches a resource,
       * which made it the single largest blocker in the corpus -- 58 of
       * 334 shaders -- for what amounts to a rename.
       */
      NEED(1);
      if (p[0] < ctx->mod->num_values &&
          inst->result_value < ctx->mod->num_values) {
         ctx->handle_base[inst->result_value] = ctx->handle_base[p[0]];
         ctx->handle_heap[inst->result_value] = ctx->handle_heap[p[0]];
         ctx->handle_kind_v[inst->result_value] = ctx->handle_kind_v[p[0]];
         ctx->handle_stride_v[inst->result_value] =
            ctx->handle_stride_v[p[0]];
         ctx->handle_valid[inst->result_value] = ctx->handle_valid[p[0]];
      }

      /* annotateHandle's second operand is a constant
       * DxilResourceProperties pair: dword0's low byte is the resource
       * kind, dword1 is the element stride for a structured buffer. For
       * a heap handle this is the ONLY place kind and stride exist --
       * there is no declared range behind it.
       */
      if (opcode == D12_DXOP_ANNOTATE_HANDLE && np >= 2 &&
          p[1] < ctx->mod->num_values &&
          inst->result_value < ctx->mod->num_values) {
         const struct d12_value *props = &ctx->mod->values[p[1]];
         if (props->kind == D12_VALUE_CONSTANT &&
             props->num_agg_elems >= 2) {
            uint64_t d0 = 0, d1 = 0;
            if (const_uint(ctx, props->agg_elems[0], &d0) &&
                const_uint(ctx, props->agg_elems[1], &d1)) {
               ctx->handle_kind_v[inst->result_value] = (uint32_t)d0 & 0xff;
               ctx->handle_stride_v[inst->result_value] = (uint32_t)d1;
            }
         }
      }

      def_result(ctx, inst, get_def(ctx, p[0]));
      return true;
   }

   case D12_DXOP_CREATE_HANDLE_FROM_HEAP: {
      /* (index, samplerHeap, nonUniformIndex): SM 6.6's
       * ResourceDescriptorHeap[i] / SamplerDescriptorHeap[i]. The handle
       * is the raw heap index; what it addresses is decided by the
       * annotateHandle that always follows.
       */
      NEED(2);

      nir_def *idx = get_def(ctx, p[0]);
      if (!idx)
         return false;

      uint64_t sampler_heap = 0;
      const_uint(ctx, p[1], &sampler_heap);

      if (inst->result_value < ctx->mod->num_values) {
         ctx->handle_heap[inst->result_value] = sampler_heap ? 2 : 1;
         /* Raw buffer until annotateHandle says otherwise. */
         ctx->handle_kind_v[inst->result_value] = D12_RES_KIND_RAW_BUFFER;
      }
      def_result(ctx, inst, nir_u2u32(b, idx));
      return true;
   }

   case D12_DXOP_CREATE_HANDLE_FROM_BINDING: {
      /* (bind, index, nonUniformIndex). `bind` is a constant ResBind
       * aggregate { rangeLowerBound, rangeUpperBound, spaceID,
       * resourceClass }; the class must come from it, because the
       * register alone does not identify a resource -- t2 and u2 are
       * different bindings with the same number.
      */
      NEED(2);
      unsigned klass, lower, upper, space;
      if (!decode_res_bind(ctx, p[0], &lower, &upper, &space, &klass))
         return false;

      if (klass >= D12_RES_CLASS_COUNT) {
         trans_fail(ctx, "ResBind names an unknown resource class");
         return false;
      }

      unsigned base_slot = ~0u;
      unsigned rooted_reg = 0;
      const bool rooted_heap = unbounded_root_for_register(
         ctx->map, klass, space, lower, &base_slot, &rooted_reg);
      if (!rooted_heap)
         base_slot = d12_binding_map_find(ctx->map, klass, space, lower);
      if (base_slot == ~0u) {
         trans_fail(ctx, "shader uses a resource the root signature does "
                         "not bind");
         return false;
      }

      nir_def *index_value = get_def(ctx, p[1]);
      if (!index_value)
         return false;
      nir_def *index = nir_u2u32(b, index_value);

      /* DXIL defines this as an index into ResBind's range. Rooted ranges
       * may begin inside a larger API descriptor table, so add that static
       * register bias before indexing the heap-relative handle. */
      nir_def *slot = index;
      if (rooted_heap)
         slot = nir_iadd_imm(b, slot, lower - rooted_reg);

      if (!rooted_heap) {
         uint64_t declared_count =
            upper >= lower ? (uint64_t)upper - lower + 1 : 0;
         if (!declared_count ||
             declared_count > D12_MAX_RESOURCE_BINDINGS ||
             finite_root_range_count(ctx->map, klass, space, lower,
                                     base_slot) < declared_count) {
            trans_fail(ctx, "dynamic resource binding crosses root "
                            "descriptor ranges");
            return false;
         }
         slot = nir_iadd_imm(b, slot, base_slot);
      }

      if (inst->result_value < ctx->mod->num_values) {
         ctx->handle_base[inst->result_value] = base_slot;
         if (rooted_heap)
            ctx->handle_heap[inst->result_value] = 3;
      }

      if (d12_debug()->debug_bindings) {
         fprintf(stderr,
                 "binding: fromBinding class %u range %u..%u space %u, "
                 "base slot %u%s\n",
                 klass, lower, upper, space, base_slot,
                 rooted_heap ? " heap-rooted" : "");
      }

      def_result(ctx, inst, slot);
      return true;
   }

   case D12_DXOP_CBUFFER_LOAD_LEGACY: {
      /* (handle, regIndex) -> one 16-byte constant-buffer row, which DXIL
       * consumes with extractvalue. A row is a vec4.
       */
      NEED(2);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      /* Rows are 16 bytes; the row index scales, no second coordinate.
       * The row's SHAPE follows the declared result: four 32-bit
       * elements normally, eight 16-bit or two 64-bit ones for cbuffers
       * of halves/shorts or doubles/longs -- same bytes, different
       * extractvalue indexing.
       */
      unsigned bits = 32;
      if (inst->result_type < ctx->mod->num_types) {
         const struct d12_type *rt = &ctx->mod->types[inst->result_type];
         if ((rt->kind == D12_TYPE_STRUCT_ANON ||
              rt->kind == D12_TYPE_STRUCT_NAMED) &&
             rt->strct.num_members) {
            unsigned mb = type_bit_size(ctx->mod, rt->strct.members[0]);
            if (mb == 16 || mb == 64)
               bits = mb;
         }
      }

      nir_def *offset = nir_imul_imm(b, get_def(ctx, p[1]), 16);
      nir_def *addr = resource_address(ctx, &h, offset,
                                       nir_imm_int(b, 0));

      def_result(ctx, inst,
                 nir_load_global(b, 128 / bits, bits, addr,
                                 .align_mul = 16));
      return true;
   }

   case D12_DXOP_CBUFFER_LOAD: {
      /*
       * (handle, byteOffset, alignment) -> one scalar.
       *
       * The non-legacy form is byte-addressed, not row-addressed: the
       * offset is already in bytes and must NOT be scaled by 16 the way
       * cbufferLoadLegacy's register index is. Scaling it would read the
       * right buffer at sixteen times the right offset -- in bounds for a
       * large cbuffer, and therefore silent.
       *
       * The width likewise comes from the result type rather than being
       * assumed 32: DXC emits .i16 and .i64 overloads, and a 32-bit load
       * standing in for a 64-bit one drops half the value.
       */
      NEED(3);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      unsigned bits = type_bit_size(ctx->mod, inst->result_type);

      /* DXIL states the alignment; trust it only when it is a sane power
       * of two, since align_mul is a promise to the backend and an
       * overstated one miscompiles rather than merely deoptimises.
       */
      uint64_t align = 0;
      if (!const_uint(ctx, p[2], &align) || !align || (align & (align - 1)) ||
          align > 16)
         align = bits / 8;

      nir_def *addr = resource_address(ctx, &h,
                                       get_def(ctx, p[1]),
                                       nir_imm_int(b, 0));

      def_result(ctx, inst,
                 nir_load_global(b, 1, bits, addr, .align_mul = align));
      return true;
   }

   case D12_DXOP_BUFFER_STORE:
   case D12_DXOP_RAW_BUFFER_STORE: {
      /* (handle, coord0, coord1, val0..val3, mask [, align]) */
      NEED(8);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      uint64_t mask = 0;
      if (!const_uint(ctx, p[7], &mask))
         mask = 0xf;

      if (opcode == D12_DXOP_BUFFER_STORE &&
          h.kind == D12_RES_KIND_TYPED_BUFFER) {
         if (!d12_native_pbe_software_addressing_supported(
                ctx->descriptor_generation)) {
            trans_fail(ctx, "typed-buffer stores require unproven AGX3 PBE "
                            "software addressing");
            return false;
         }

         /* Format conversion on write: an image store with buffer
          * dimension, mirroring the typed-load path above. Runtime
          * needs a writable (PBE) descriptor, same caveat as
          * textureStore. native_image_handle also resolves a dynamically
          * indexed root range into the parallel PBE heap's absolute slot.
          */
         nir_def *vals4[4];
         for (unsigned i = 0; i < 4; i++) {
            vals4[i] = get_def(ctx, p[3 + i]);
            if (!vals4[i] || operand_is_undef(ctx, p[3 + i]))
               vals4[i] = nir_imm_int(b, 0);
         }

         nir_def *zero = nir_imm_int(b, 0);
         nir_bindless_image_store(
            b, native_image_handle(ctx, &h, true),
            nir_vec4(b, get_def(ctx, p[1]), zero, zero, zero),
            nir_imm_int(b, 0), nir_vec(b, vals4, 4), nir_imm_int(b, 0),
            .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false,
            .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_READABLE);
         return true;
      }

      nir_def *addr = resource_address(ctx, &h,
                                       get_def(ctx, p[1]),
                                       get_def(ctx, p[2]));
      nir_def *byte_offset =
         h.kind == D12_RES_KIND_STRUCTURED_BUFFER
            ? nir_iadd(
                 b,
                 nir_imul_imm(b, nir_u2u64(b, get_def(ctx, p[1])),
                              h.stride),
                 nir_u2u64(b, get_def(ctx, p[2])))
            : get_def(ctx, p[1]);
      nir_def *size_bytes = resource_view_size_bytes(ctx, &h);
      nir_def *element_valid =
         h.kind == D12_RES_KIND_STRUCTURED_BUFFER
            ? nir_ult(b, get_def(ctx, p[1]),
                      resource_view_count_or_bytes(ctx, &h))
            : NULL;

      /* Contiguous low components only; DXIL emits the common case. */
      unsigned comps = util_last_bit((unsigned)mask & 0xf);
      if (!comps || ((unsigned)mask & 0xf) != BITFIELD_MASK(comps)) {
         trans_fail(ctx, "sparse buffer store write mask unsupported");
         return false;
      }

      nir_def *vals[4];
      for (unsigned i = 0; i < comps; i++)
         vals[i] = get_def(ctx, p[3 + i]);

      robust_store_buffer_masked(ctx, addr, byte_offset, size_bytes,
                                 element_valid, vals, comps);
      return true;
   }

   case D12_DXOP_BUFFER_LOAD:
   case D12_DXOP_RAW_BUFFER_LOAD: {
      /* (handle, coord0, coord1 [, mask, align]) */
      NEED(3);

      struct handle_ref h;
      if (!handle_info(ctx, p[0], &h))
         return false;

      /* A TYPED buffer (Buffer<T>/RWBuffer<T>) is not byte-addressed
       * memory: the element format converts on access, which on AGX is
       * the texture unit's job. It reads the same heap descriptor a
       * texture does, with buffer dimension -- exactly how Honeykrisp
       * models texel buffers (hk_buffer_view.c).
       */
      nir_def *data;
      nir_def *byte_offset;
      nir_def *status_byte_offset;
      nir_def *access_size_B;
      nir_def *element_valid = NULL;
      nir_def *residency = buffer_residency_pointer(ctx, &h);
      nir_def *buffer_base = resource_base_address(ctx, &h);
      unsigned bits = 32;
      if (inst->result_type < ctx->mod->num_types) {
         const struct d12_type *rt = &ctx->mod->types[inst->result_type];
         if ((rt->kind == D12_TYPE_STRUCT_ANON ||
              rt->kind == D12_TYPE_STRUCT_NAMED) && rt->strct.num_members)
            bits = type_bit_size(ctx->mod, rt->strct.members[0]);
      }
      if (bits != 16 && bits != 32 && bits != 64)
         bits = 32;

      if (opcode == D12_DXOP_BUFFER_LOAD &&
          h.kind == D12_RES_KIND_TYPED_BUFFER) {
         nir_def *stride = resource_view_stride_bytes(ctx, &h);
         byte_offset = nir_imul(b, get_def(ctx, p[1]),
                                nir_umax_imm(b, stride, 1));
         status_byte_offset = nir_imul(
            b, nir_u2u64(b, get_def(ctx, p[1])),
            nir_u2u64(b, nir_umax_imm(b, stride, 1)));
         access_size_B = nir_umax_imm(b, stride, 1);
         nir_def *tex_handle = native_image_handle(ctx, &h, false);

         nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
         tex->op = nir_texop_txf;
         tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
         tex->is_array = false;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 1;
         tex->src[0] =
            nir_tex_src_for_ssa(nir_tex_src_texture_handle, tex_handle);
         tex->src[1] =
            nir_tex_src_for_ssa(nir_tex_src_coord, get_def(ctx, p[1]));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(b, &tex->instr);
         data = &tex->def;
      } else {
         byte_offset =
            h.kind == D12_RES_KIND_STRUCTURED_BUFFER
               ? nir_iadd(
                    b,
                    nir_imul_imm(b, nir_u2u64(b, get_def(ctx, p[1])),
                                 h.stride),
                    nir_u2u64(b, get_def(ctx, p[2])))
               : get_def(ctx, p[1]);
         status_byte_offset = nir_u2u64(b, byte_offset);
         nir_def *addr = nir_iadd(
            b, resource_base_address_read(ctx, buffer_base, residency),
            nir_u2u64(b, byte_offset));
         nir_def *size_bytes = resource_view_size_bytes(ctx, &h);
         element_valid =
            h.kind == D12_RES_KIND_STRUCTURED_BUFFER
               ? nir_ult(b, get_def(ctx, p[1]),
                         resource_view_count_or_bytes(ctx, &h))
               : NULL;
         data = robust_load_buffer_vec4(
            ctx, addr, byte_offset, size_bytes, element_valid, bits);
         unsigned first_component = 0;
         unsigned component_count = 4;
         if (opcode == D12_DXOP_RAW_BUFFER_LOAD && np > 3) {
            uint64_t mask = 0xf;
            if (const_uint(ctx, p[3], &mask) && (mask & 0xf)) {
               unsigned first = ffs((unsigned)mask & 0xf) - 1;
               unsigned last = util_last_bit((unsigned)mask & 0xf);
               first_component = first;
               component_count = last - first;
            }
         }
         status_byte_offset = nir_iadd_imm(
            b, status_byte_offset, first_component * (bits / 8));
         access_size_B =
            nir_imm_int(b, component_count * (bits / 8));
      }

      nir_def *status = buffer_residency_status(
         ctx, &h, residency, buffer_base, status_byte_offset, access_size_B,
         element_valid);
      nir_def *mapped[5] = {
         nir_channel(b, data, 0), nir_channel(b, data, 1),
         nir_channel(b, data, 2), nir_channel(b, data, 3),
         nir_u2uN(b, status, data->bit_size),
      };
      def_result(ctx, inst, nir_vec(b, mapped, 5));
      return true;
   }

   default:
      /* The ray-query state accessors: transforms and scalars read back
       * after a traversal that never visited anything. An identity
       * transform and zero are what an uncommitted query reports.
       */
      if (opcode >= D12_DXOP_RQ_MATRIX_FIRST &&
          opcode <= D12_DXOP_RQ_MATRIX_LAST) {
         NEED(3);
         uint64_t row = 0, col = 0;
         const_uint(ctx, p[1], &row);
         const_uint(ctx, p[2], &col);
         def_result(ctx, inst,
                    nir_imm_float(b, row == col ? 1.0f : 0.0f));
         return true;
      }

      if (opcode >= D12_DXOP_RQ_STATE_SCALAR_FIRST &&
          opcode <= D12_DXOP_RQ_STATE_SCALAR_LAST && callee &&
          !strncmp(callee, "dx.op.rayQuery_", 15)) {
         unsigned bits = type_bit_size(ctx->mod, inst->result_type);
         def_result(ctx, inst,
                    type_is_float(ctx->mod, inst->result_type)
                       ? nir_imm_floatN_t(b, 0.0, bits ? bits : 32)
                       : nir_imm_intN_t(b, 0, bits ? bits : 32));
         return true;
      }

      if (d12_debug()->dump_insts) {
         fprintf(stderr, "  op %llu operands:", (unsigned long long)opcode);
         for (unsigned i = 0; i < np; i++) {
            uint64_t c;
            if (const_uint(ctx, p[i], &c))
               fprintf(stderr, " [%u]=%llu", i, (unsigned long long)c);
            else
               fprintf(stderr, " [%u]=dyn", i);
         }
         fprintf(stderr, "\n");
      }
      /* Name the intrinsic, not just its number: the name is right there
       * in the module and is what makes a corpus run actionable.
       */
      mesa_loge("dxil: unhandled dx.op opcode %llu (%s)",
                (unsigned long long)opcode, callee ? callee : "?");
      trans_fail(ctx, "unhandled dx.op intrinsic");
      return false;
   }
#undef NEED
}

static bool
translate_binop(struct trans_ctx *ctx, const struct d12_inst *inst)
{
   nir_builder *b = ctx->b;

   if (inst->num_ops < 2 || inst->num_extra < 1) {
      trans_fail(ctx, "malformed binop");
      return false;
   }

   nir_def *x = get_def(ctx, inst->ops[0]);
   nir_def *y = get_def(ctx, inst->ops[1]);

   /* get_def RECORDS a failure for an out-of-range index but still
    * returns a dummy def, so translation continues -- and indexing
    * values[] with the index it just rejected reads off the end of a
    * fixed array whose bound comes from unvalidated bitstream data.
    */
   bool is_float = inst->ops[0] < ctx->mod->num_values &&
                   x->bit_size >= 16 &&
                   type_is_float(ctx->mod,
                                 ctx->mod->values[inst->ops[0]].type);

   bool is_shift = inst->extra[0] == D12_BINOP_SHL ||
                   inst->extra[0] == D12_BINOP_LSHR ||
                   inst->extra[0] == D12_BINOP_ASHR;

   /* Width drift between our IO choices and DXIL's declared types; the
    * same harmonization translate_cmp does. Shifts are exempt: NIR
    * fixes their AMOUNT operand at 32 bits regardless of the value's
    * width, so a 64-bit shift takes a 32-bit count, not a widened one.
    */
   if (is_shift) {
      if (y && y->bit_size != 32)
         y = nir_u2u32(b, y);
   } else if (x && y && x->bit_size != y->bit_size && x->bit_size > 1 &&
              y->bit_size > 1) {
      unsigned bits = MAX2(x->bit_size, y->bit_size);
      if (x->bit_size < bits)
         x = is_float ? nir_f2fN(b, x, bits) : nir_u2uN(b, x, bits);
      if (y->bit_size < bits)
         y = is_float ? nir_f2fN(b, y, bits) : nir_u2uN(b, y, bits);
   }

   nir_def *res = NULL;

   switch (inst->extra[0]) {
   case D12_BINOP_ADD:  res = is_float ? nir_fadd(b, x, y) : nir_iadd(b, x, y); break;
   case D12_BINOP_SUB:  res = is_float ? nir_fsub(b, x, y) : nir_isub(b, x, y); break;
   case D12_BINOP_MUL:  res = is_float ? nir_fmul(b, x, y) : nir_imul(b, x, y); break;
   case D12_BINOP_UDIV: res = d12_nir_integer_divmod(b, x, y, false, false); break;
   case D12_BINOP_SDIV: res = is_float ? nir_fdiv(b, x, y)
                                      : d12_nir_integer_divmod(b, x, y, true, false); break;
   case D12_BINOP_UREM: res = d12_nir_integer_divmod(b, x, y, false, true); break;
   case D12_BINOP_SREM: res = is_float ? nir_frem(b, x, y)
                                      : d12_nir_integer_divmod(b, x, y, true, true); break;
   case D12_BINOP_SHL:  res = nir_ishl(b, x, y); break;
   case D12_BINOP_LSHR: res = nir_ushr(b, x, y); break;
   case D12_BINOP_ASHR: res = nir_ishr(b, x, y); break;
   case D12_BINOP_AND:  res = nir_iand(b, x, y); break;
   case D12_BINOP_OR:   res = nir_ior(b, x, y); break;
   case D12_BINOP_XOR:  res = nir_ixor(b, x, y); break;
   default:
      trans_fail(ctx, "unhandled LLVM binop");
      return false;
   }

   def_result(ctx, inst, res);
   return true;
}

/* LLVM compare predicates (float 0-15, integer 32-41) */
static bool
translate_cmp(struct trans_ctx *ctx, const struct d12_inst *inst)
{
   nir_builder *b = ctx->b;

   if (inst->num_ops < 2 || inst->num_extra < 1) {
      trans_fail(ctx, "malformed compare");
      return false;
   }

   nir_def *x = get_def(ctx, inst->ops[0]);
   nir_def *y = get_def(ctx, inst->ops[1]);
   nir_def *res = NULL;

   /* Our IO widths can drift from DXIL's declared types (fp16 inputs,
    * widened varyings), so two well-typed DXIL operands may reach here
    * at different bit sizes. Widen the smaller: value-preserving for
    * both integer and float compares.
    */
   if (x && y && x->bit_size != y->bit_size && x->bit_size > 1 &&
       y->bit_size > 1) {
      bool flt = inst->extra[0] < 32;
      unsigned bits = MAX2(x->bit_size, y->bit_size);
      if (x->bit_size < bits)
         x = flt ? nir_f2fN(b, x, bits) : nir_u2uN(b, x, bits);
      if (y->bit_size < bits)
         y = flt ? nir_f2fN(b, y, bits) : nir_u2uN(b, y, bits);
   }

   switch (inst->extra[0]) {
   /* Float predicates. Ordered and unordered forms are mapped to the
    * same NIR comparison; AGX has no NaN-signalling distinction here.
    */
   case 1: case 9:  res = nir_feq(b, x, y); break;  /* OEQ / UEQ */
   case 2: case 10: res = nir_flt(b, y, x); break;  /* OGT / UGT */
   case 3: case 11: res = nir_fge(b, x, y); break;  /* OGE / UGE */
   case 4: case 12: res = nir_flt(b, x, y); break;  /* OLT / ULT */
   case 5: case 13: res = nir_fge(b, y, x); break;  /* OLE / ULE */
   case 6: case 14: res = nir_fneu(b, x, y); break; /* ONE / UNE */

   /* Integer predicates */
   case 32: res = nir_ieq(b, x, y); break;
   case 33: res = nir_ine(b, x, y); break;
   case 34: res = nir_ult(b, y, x); break;
   case 35: res = nir_uge(b, x, y); break;
   case 36: res = nir_ult(b, x, y); break;
   case 37: res = nir_uge(b, y, x); break;
   case 38: res = nir_ilt(b, y, x); break;
   case 39: res = nir_ige(b, x, y); break;
   case 40: res = nir_ilt(b, x, y); break;
   case 41: res = nir_ige(b, y, x); break;

   default:
      trans_fail(ctx, "unhandled compare predicate");
      return false;
   }

   def_result(ctx, inst, res);
   return true;
}

/* select: LLVM orders operands (true, false, cond) */
static bool
translate_select(struct trans_ctx *ctx, const struct d12_inst *inst)
{
   if (inst->num_ops < 3) {
      trans_fail(ctx, "malformed select");
      return false;
   }

   /* A select BETWEEN POINTERS (cond ? &a[i] : &b[j]) has no SSA value
    * behind either arm; both are (space, base, dyn) pointers. When the
    * spaces agree, the selected pointer is the same space with a
    * selected byte offset.
    */
   uint8_t sa = pointer_for_value(ctx, inst->ops[0]);
   uint8_t sb = pointer_for_value(ctx, inst->ops[1]);
   if (sa != D12_PTR_NONE && sa == sb) {
      nir_builder *b = ctx->b;

      nir_def *cond0 = get_def(ctx, inst->ops[2]);
      if (!cond0)
         return false;
      if (cond0->bit_size != 1)
         cond0 = nir_ine_imm(b, cond0, 0);

      nir_def *ta = ctx->defs[inst->ops[0]];
      nir_def *fa = ctx->defs[inst->ops[1]];
      nir_def *toff = nir_iadd_imm(b, ta ? ta : nir_imm_int(b, 0),
                                   ctx->ptr_base[inst->ops[0]]);
      nir_def *foff = nir_iadd_imm(b, fa ? fa : nir_imm_int(b, 0),
                                   ctx->ptr_base[inst->ops[1]]);

      ctx->ptr_space[inst->result_value] = sa;
      ctx->ptr_base[inst->result_value] = 0;
      ctx->ptr_pointee[inst->result_value] =
         ctx->ptr_pointee[inst->ops[0]];
      ctx->defs[inst->result_value] = nir_bcsel(b, cond0, toff, foff);
      return true;
   }

   nir_def *t = get_def(ctx, inst->ops[0]);
   nir_def *f = get_def(ctx, inst->ops[1]);
   nir_def *cond = get_def(ctx, inst->ops[2]);

   /* NIR wants a 1-bit boolean condition. */
   if (cond->bit_size != 1)
      cond = nir_ine_imm(ctx->b, cond, 0);

   def_result(ctx, inst, nir_bcsel(ctx->b, cond, t, f));
   return true;
}

static bool
translate_cast(struct trans_ctx *ctx, const struct d12_inst *inst)
{
   nir_builder *b = ctx->b;

   if (inst->num_ops < 1 || inst->num_extra < 1) {
      trans_fail(ctx, "malformed cast");
      return false;
   }

   /* A bitcast of a POINTER (DXC retypes allocas freely) changes the
    * pointee type and nothing else: same space, same offset. The
    * pointee follows the cast's target so later GEP walks use the
    * retyped layout.
    */
   if (inst->extra[0] == D12_CAST_BITCAST &&
       pointer_for_value(ctx, inst->ops[0]) != D12_PTR_NONE &&
       inst->result_value < ctx->mod->num_values) {
      ctx->ptr_space[inst->result_value] = ctx->ptr_space[inst->ops[0]];
      ctx->ptr_base[inst->result_value] = ctx->ptr_base[inst->ops[0]];
      ctx->ptr_global[inst->result_value] = ctx->ptr_global[inst->ops[0]];
      ctx->defs[inst->result_value] = ctx->defs[inst->ops[0]];

      uint32_t pointee = ctx->ptr_pointee[inst->ops[0]];
      if (inst->result_type < ctx->mod->num_types &&
          ctx->mod->types[inst->result_type].kind == D12_TYPE_POINTER)
         pointee = ctx->mod->types[inst->result_type].ptr.pointee;
      ctx->ptr_pointee[inst->result_value] = pointee;
      return true;
   }

   nir_def *src = get_def(ctx, inst->ops[0]);
   unsigned dst_bits = type_bit_size(ctx->mod, inst->result_type);
   nir_def *res = NULL;

   /* Truncation TO i1 keeps the low bit; the backend has no 1-bit
    * conversion op, but it has compares.
    */
   if (dst_bits == 1 && (inst->extra[0] == D12_CAST_TRUNC ||
                         inst->extra[0] == D12_CAST_ZEXT ||
                         inst->extra[0] == D12_CAST_SEXT)) {
      def_result(ctx, inst,
                 nir_ine_imm(b, nir_iand_imm(b, src, 1), 0));
      return true;
   }

   switch (inst->extra[0]) {
   case D12_CAST_TRUNC:   res = nir_u2uN(b, src, dst_bits); break;
   case D12_CAST_ZEXT:    res = nir_u2uN(b, src, dst_bits); break;
   case D12_CAST_SEXT:    res = nir_i2iN(b, src, dst_bits); break;
   case D12_CAST_FPTOUI:  res = nir_f2uN(b, src, dst_bits); break;
   case D12_CAST_FPTOSI:  res = nir_f2iN(b, src, dst_bits); break;
   case D12_CAST_UITOFP:  res = nir_u2fN(b, src, dst_bits); break;
   case D12_CAST_SITOFP:  res = nir_i2fN(b, src, dst_bits); break;
   case D12_CAST_FPTRUNC:
      /* LLVM fptrunc must perform an observable narrowing. Using NIR's
       * rounding-unspecified f2f16 lets the AGX optimizer fold the adjacent
       * f2f32 back into a full-precision use. DXIL permits native FP16 to
       * round to nearest here, and the explicit opcode preserves that
       * quantization across backend optimization. */
      res = dst_bits == 16 ? nir_f2f16_rtne(b, src)
                           : nir_f2fN(b, src, dst_bits);
      break;
   case D12_CAST_FPEXT:   res = nir_f2fN(b, src, dst_bits); break;
   case D12_CAST_BITCAST: res = src; break;
   default:
      trans_fail(ctx, "unhandled LLVM cast");
      return false;
   }

   def_result(ctx, inst, res);
   return true;
}

/* Is this call a dx.op intrinsic, and if so what opcode? */
static bool
call_dxop_opcode(struct trans_ctx *ctx, const struct d12_inst *inst,
                 uint64_t *opcode, const char **name)
{
   if (inst->num_ops < 2)
      return false;

   uint32_t callee_val = inst->ops[0];
   if (callee_val >= ctx->mod->num_values)
      return false;

   const struct d12_value *v = &ctx->mod->values[callee_val];
   if (v->kind != D12_VALUE_FUNCTION || !v->name)
      return false;

   if (strncmp(v->name, "dx.op.", 6))
      return false;

   *name = v->name;
   return const_uint(ctx, inst->ops[1], opcode);
}

nir_shader *
d12_dxil_to_nir(const struct d12_module *mod, mesa_shader_stage stage,
                const struct d12_binding_map *map,
                uint32_t descriptor_generation, const char **error)
{
   struct d12_native_descriptor_layout native_layout;
   if (!d12_native_descriptor_layout(descriptor_generation, &native_layout)) {
      *error = "unsupported native descriptor generation";
      return NULL;
   }

   /* The entry point is the module's single function definition --
    * EXCEPT in a hull shader, which has two: the per-control-point entry
    * and a patch constant function that writes SV_TessFactor. Skip the
    * latter when looking for the entry, or the shader that gets compiled
    * is whichever the bitcode happened to emit first.
    */
   /* A hull shader has TWO definitions: the per-control-point entry and
    * the patch constant function that writes SV_TessFactor. DXC emits the
    * patch constant one FIRST, so "the first definition" picked the wrong
    * body -- 18 corpus hull shaders were compiling their patch constant
    * function and counting as passes.
    */
   const struct d12_function *entry = NULL;
   for (unsigned i = 0; i < mod->num_functions; i++) {
      if (mod->functions[i].is_declaration)
         continue;
      if (mod->has_patch_constant && i == mod->patch_constant_func)
         continue;
      entry = &mod->functions[i];
      break;
   }

   if (!entry) {
      *error = "module has no function definition";
      return NULL;
   }

   /*
    * MERGE THE PATCH CONSTANT FUNCTION INTO THE HULL ENTRY. DXIL keeps
    * the SV_TessFactor writes in a separate function; NIR and poly's TCS
    * lowering (which follows GLSL) expect one shader that writes both.
    */
   struct d12_function merged;
   struct d12_inst *merged_insts = NULL;
   unsigned patch_constant_start = ~0u;

   if (stage == MESA_SHADER_TESS_CTRL && mod->has_patch_constant &&
       mod->patch_constant_func < mod->num_functions) {
      const struct d12_function *pc =
         &mod->functions[mod->patch_constant_func];

      unsigned keep = entry->num_insts;
      while (keep > 0 && entry->insts[keep - 1].code == D12_FUNC_INST_RET)
         keep--;

      merged_insts = calloc(keep + pc->num_insts, sizeof(*merged_insts));
      if (!merged_insts) {
         *error = "out of memory merging the patch constant function";
         return NULL;
      }

      memcpy(merged_insts, entry->insts, keep * sizeof(*merged_insts));

      unsigned entry_blocks = 1;
      for (unsigned i = 0; i < keep; i++)
         entry_blocks = MAX2(entry_blocks, entry->insts[i].block + 1);

      /* Append into the entry's LAST block rather than adding blocks.
       * Offsetting the patch constant's blocks instead leaves the entry's
       * final block with no edge to them -- the entry's RET was just
       * dropped -- and nir_lower_goto_ifs segfaults structurizing the
       * result.
       *
       * A patch constant function with control flow of its own would need
       * real CFG stitching, so refuse it by name rather than mis-compile
       * it.
       */
      unsigned pc_blocks = 1;
      for (unsigned i = 0; i < pc->num_insts; i++)
         pc_blocks = MAX2(pc_blocks, pc->insts[i].block + 1);

      if (pc_blocks > 1) {
         free(merged_insts);
         *error = "hull patch constant function has control flow";
         return NULL;
      }

      for (unsigned i = 0; i < pc->num_insts; i++) {
         merged_insts[keep + i] = pc->insts[i];
         merged_insts[keep + i].block = entry_blocks - 1;
      }

      merged = *entry;
      merged.insts = merged_insts;
      merged.num_insts = keep + pc->num_insts;
      merged.num_blocks = entry_blocks;
      entry = &merged;
      patch_constant_start = keep;
   }

   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                  "dxil");

   /* These are application shaders. The builder marks its shaders
    * internal, and the AGX backend treats internal shaders to a scratch
    * optimization pass pair that leaves small arrays as variable derefs
    * it then cannot emit.
    */
   b.shader->info.internal = false;

   struct trans_ctx ctx = {
      .mod = mod,
      .map = map,
      .descriptor_generation = descriptor_generation,
      .native_layout = native_layout,
      .func = entry,
      .b = &b,
      .num_defs = mod->num_values,
   };

   ctx.defs = calloc(mod->num_values, sizeof(*ctx.defs));
   ctx.phi_reg = calloc(mod->num_values, sizeof(*ctx.phi_reg));
   ctx.ptr_space = calloc(mod->num_values, sizeof(*ctx.ptr_space));
   ctx.ptr_base = calloc(mod->num_values, sizeof(*ctx.ptr_base));
   ctx.ptr_pointee = calloc(mod->num_values, sizeof(*ctx.ptr_pointee));
   ctx.ptr_global = calloc(mod->num_values, sizeof(*ctx.ptr_global));
   ctx.handle_base = malloc(mod->num_values * sizeof(*ctx.handle_base));
   if (ctx.handle_base)
      memset(ctx.handle_base, 0xff, mod->num_values * sizeof(*ctx.handle_base));
   ctx.handle_heap = calloc(mod->num_values, sizeof(*ctx.handle_heap));
   ctx.handle_kind_v = calloc(mod->num_values, sizeof(*ctx.handle_kind_v));
   ctx.handle_stride_v = calloc(mod->num_values, sizeof(*ctx.handle_stride_v));
   ctx.handle_valid = calloc(mod->num_values, sizeof(*ctx.handle_valid));
   if (!ctx.defs || !ctx.phi_reg || !ctx.ptr_space || !ctx.ptr_base ||
       !ctx.ptr_pointee || !ctx.ptr_global || !ctx.handle_base ||
       !ctx.handle_heap ||
       !ctx.handle_kind_v || !ctx.handle_stride_v || !ctx.handle_valid) {
      *error = "out of memory";
      free(ctx.defs);
      free(ctx.phi_reg);
      free(ctx.ptr_space);
      free(ctx.ptr_base);
      free(ctx.ptr_pointee);
      free(ctx.ptr_global);
      free(ctx.handle_base);
      free(ctx.handle_heap);
      free(ctx.handle_kind_v);
      free(ctx.handle_stride_v);
      free(ctx.handle_valid);
      ralloc_free(b.shader);
      return NULL;
   }

   /* Index the declared strides by slot so the access sites do not have to
    * search the resource list again.
    */
   for (unsigned i = 0; i < mod->num_resources; i++) {
      const struct d12_resource_binding *rb = &mod->resources[i];
      if (rb->klass >= D12_RES_CLASS_COUNT)
         continue;

      unsigned count = d12_range_count(rb);

      if (!count) {
         unsigned slot = d12_binding_map_find(map, rb->klass, rb->space,
                                              rb->lower_bound);
         if (slot != ~0u) {
            ctx.slot_stride[slot] = rb->stride_B;
            ctx.slot_kind[slot] = rb->kind;
         }
         continue;
      }

      for (unsigned r = 0; r < count; r++) {
         unsigned slot = d12_binding_map_find(map, rb->klass, rb->space,
                                              rb->lower_bound + r);
         if (slot == ~0u)
            continue; /* not bound by the signature; caught at first use */

         ctx.slot_stride[slot] = rb->stride_B;
         ctx.slot_kind[slot] = rb->kind;
      }
   }

   if (stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_TESS_EVAL) {
      /* DXIL's TessellatorDomain/Partitioning/OutputPrimitive, from the
       * entry-point properties. NIR needs them before any lowering runs
       * because they decide the shape of the emulation.
       */
      static const enum tess_primitive_mode domains[] = {
         [1] = TESS_PRIMITIVE_ISOLINES,
         [2] = TESS_PRIMITIVE_TRIANGLES,
         [3] = TESS_PRIMITIVE_QUADS,
      };
      static const enum gl_tess_spacing spacings[] = {
         [1] = TESS_SPACING_EQUAL,
         [2] = TESS_SPACING_FRACTIONAL_ODD,
         [3] = TESS_SPACING_FRACTIONAL_EVEN,
      };

      b.shader->info.tess._primitive_mode =
         mod->tess_domain < ARRAY_SIZE(domains) && domains[mod->tess_domain]
            ? domains[mod->tess_domain]
            : TESS_PRIMITIVE_TRIANGLES;

      /* DXIL: 1 = integer, 2 = pow2, 3 = fractional_odd,
       * 4 = fractional_even. AGX has no pow2 spacing; it rounds up to
       * equal, which is what D3D's pow2 degenerates to here.
       */
      unsigned part = mod->tess_partitioning;
      b.shader->info.tess.spacing =
         part == 3   ? TESS_SPACING_FRACTIONAL_ODD
         : part == 4 ? TESS_SPACING_FRACTIONAL_EVEN
                     : TESS_SPACING_EQUAL;
      (void)spacings;

      /* DXIL TessellatorOutputPrimitive: 1 = point, 2 = line,
       * 3 = triangle_cw, 4 = triangle_ccw.
       */
      b.shader->info.tess.point_mode = mod->tess_output_prim == 1;
      b.shader->info.tess.ccw = mod->tess_output_prim == 4;

      if (stage == MESA_SHADER_TESS_CTRL) {
         b.shader->info.tess.tcs_vertices_out =
            mod->tess_control_points ? mod->tess_control_points : 3;
      }
   } else if (stage == MESA_SHADER_GEOMETRY) {
      /* DXIL InputPrimitive: 1 = point, 2 = line, 3 = triangle,
       * 6 = line_adj, 7 = triangle_adj.
       */
      unsigned ip = mod->gs_input_prim;
      b.shader->info.gs.input_primitive =
         ip == 1   ? MESA_PRIM_POINTS
         : ip == 2 ? MESA_PRIM_LINES
         : ip == 6 ? MESA_PRIM_LINES_ADJACENCY
         : ip == 7 ? MESA_PRIM_TRIANGLES_ADJACENCY
                   : MESA_PRIM_TRIANGLES;

      /* DXIL PrimitiveTopology: 1 = point, 3 = line strip,
       * 5 = triangle strip.
       */
      unsigned ot = mod->gs_output_topology;
      b.shader->info.gs.output_primitive =
         ot == 1   ? MESA_PRIM_POINTS
         : ot == 3 ? MESA_PRIM_LINE_STRIP
                   : MESA_PRIM_TRIANGLE_STRIP;

      b.shader->info.gs.vertices_out = mod->gs_max_vertices;
      b.shader->info.gs.invocations =
         mod->gs_instance_count ? mod->gs_instance_count : 1;
      b.shader->info.gs.vertices_in =
         mesa_vertices_per_prim(b.shader->info.gs.input_primitive);
      b.shader->info.gs.active_stream_mask = 1;
   }

   if (stage == MESA_SHADER_FRAGMENT) {
      b.shader->info.fs.untyped_color_outputs = false;
   } else if (stage == MESA_SHADER_COMPUTE) {
      /* [numthreads(x,y,z)]; without it the launch grid is empty. */
      for (unsigned i = 0; i < 3; i++) {
         b.shader->info.workgroup_size[i] =
            mod->num_threads[i] ? mod->num_threads[i] : 1;
      }
   }

   /*
    * Control flow.
    *
    * DXIL is LLVM: an unstructured CFG of basic blocks joined by branches.
    * NIR wants structured control flow. Rather than reconstruct if/loop
    * shapes ourselves, the function is built UNSTRUCTURED -- one NIR block
    * per DXIL block, joined by nir_goto/nir_goto_if -- and handed to
    * nir_lower_goto_ifs, the structurizer Mesa already uses for SPIR-V
    * kernels. Doing it any other way means writing a structurizer.
    */
   unsigned num_blocks = 1;
   for (unsigned i = 0; i < entry->num_insts; i++)
      num_blocks = MAX2(num_blocks, entry->insts[i].block + 1);

   if (num_blocks > D12_LIMIT_BLOCKS) {
      *error = "function has too many basic blocks";
      free(ctx.defs);
      free(ctx.phi_reg);
      free(ctx.ptr_space);
      free(ctx.ptr_base);
      free(ctx.ptr_pointee);
      free(ctx.ptr_global);
      free(ctx.handle_base);
      free(ctx.handle_heap);
      free(ctx.handle_kind_v);
      free(ctx.handle_stride_v);
      free(ctx.handle_valid);
      ralloc_free(b.shader);
      return NULL;
   }

   ctx.num_blocks = num_blocks;
   ctx.blocks = calloc(num_blocks, sizeof(*ctx.blocks));
   if (!ctx.blocks) {
      *error = "out of memory for the basic block table";
      free(merged_insts);
      free(ctx.defs);
      free(ctx.phi_reg);
      free(ctx.ptr_space);
      free(ctx.ptr_base);
      free(ctx.ptr_pointee);
      free(ctx.ptr_global);
      free(ctx.handle_base);
      free(ctx.handle_heap);
      free(ctx.handle_kind_v);
      free(ctx.handle_stride_v);
      free(ctx.handle_valid);
      ralloc_free(b.shader);
      return NULL;
   }

   /* Multi-block functions get a synthetic EPILOGUE block: every return
    * branches to it, and the buffered-output flush is emitted into it.
    * Flushing after the last instruction instead would place stores
    * behind whatever goto terminated the final block, which NIR refuses
    * (nothing may follow a jump).
    */
   nir_block *epilogue = NULL;

   if (num_blocks > 1) {
      b.impl->structured = false;

      ctx.blocks[0] = nir_start_block(b.impl);
      for (unsigned i = 1; i < num_blocks; i++) {
         nir_block *blk = nir_block_create(b.shader);
         exec_list_push_tail(&b.impl->body, &blk->cf_node.node);
         blk->cf_node.parent = &b.impl->cf_node;
         ctx.blocks[i] = blk;
      }

      epilogue = nir_block_create(b.shader);
      exec_list_push_tail(&b.impl->body, &epilogue->cf_node.node);
      epilogue->cf_node.parent = &b.impl->cf_node;

      /*
       * Phis become registers rather than NIR phi nodes: each predecessor
       * stores its incoming value just before branching, and the phi reads
       * it back. nir_lower_reg_intrinsics_to_ssa then rebuilds real phis
       * with correct placement. Constructing them here instead would mean
       * doing SSA construction by hand, for no gain.
       */
      b.cursor = nir_before_block(ctx.blocks[0]);
      for (unsigned i = 0; i < entry->num_insts; i++) {
         const struct d12_inst *inst = &entry->insts[i];
         if (inst->code != D12_FUNC_INST_PHI || inst->result_value == ~0u)
            continue;

         unsigned bits = type_bit_size(mod, inst->result_type);
         ctx.phi_reg[inst->result_value] = nir_decl_reg(&b, 1, bits, 0);
      }
   }

   unsigned cur_block = ~0u;

   for (unsigned i = 0; i < entry->num_insts && !ctx.error; i++) {
      const struct d12_inst *inst = &entry->insts[i];

      if (d12_debug()->dump_insts) {
         fprintf(stderr, "inst %u: code %u result %d block %u ops", i,
                 inst->code, (int)inst->result_value, inst->block);
         for (unsigned o = 0; o < inst->num_ops; o++)
            fprintf(stderr, " %u", inst->ops[o]);
         fprintf(stderr, "\n");
      }

      /* Move to this instruction's block when it changes. */
      if (num_blocks > 1 && inst->block != cur_block) {
         cur_block = inst->block;
         if (cur_block >= num_blocks) {
            trans_fail(&ctx, "instruction in a block that does not exist");
            break;
         }
         b.cursor = nir_after_block(ctx.blocks[cur_block]);
      }

      /* DXIL's patch-constant function is a distinct phase: it may read the
       * complete control-point outputs written by every hull invocation.
       * The translator buffers ordinary outputs, so publish them before the
       * appended function and order the global TCS-buffer traffic. */
      if (i == patch_constant_start) {
         flush_outputs(&ctx);
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_global);
      }

      switch (inst->code) {
      case D12_FUNC_INST_CALL: {
         uint64_t opcode;
         const char *callee = NULL;
         if (call_dxop_opcode(&ctx, inst, &opcode, &callee)) {
            translate_dxop(&ctx, inst, opcode, callee);
            break;
         }

         /* Report what the callee operand actually resolved to.
          * "some call failed" is not actionable, and the answer is
          * usually that the value is not a function at all.
          */
         uint32_t cv = inst->num_ops >= 1 ? inst->ops[0] : ~0u;
         const struct d12_value *cval =
            cv < mod->num_values ? &mod->values[cv] : NULL;

         /*
          * llvm.lifetime.start/end bracket the live range of an alloca so
          * LLVM can reuse the stack slot. They describe storage, not
          * computation, and there is nothing for a GPU backend to emit --
          * dropping them is the whole implementation.
          *
          * Skipping is safe for value numbering because these return
          * void, and the decoder allocates a value index from the
          * callee's return type rather than assuming every call defines
          * one. Emitting nothing therefore cannot shift the indices of
          * later instructions the way it would if a slot had been
          * reserved. Matching the "llvm.lifetime." prefix rather than the
          * two exact names also covers the pointer-suffixed spellings a
          * future DXC could emit.
          */
         if (cval && cval->kind == D12_VALUE_FUNCTION && cval->name &&
             !strncmp(cval->name, "llvm.lifetime.", 14))
            break;

         mesa_loge("dxil: call to non-dx.op function: callee value %u "
                   "kind %d name '%s' (module has %u values)",
                   cv, cval ? (int)cval->kind : -1,
                   cval && cval->name ? cval->name : "(none)",
                   mod->num_values);
         trans_fail(&ctx, "call to a non-dx.op function (inlining "
                          "required)");
         break;
      }

      case D12_FUNC_INST_BINOP:
         translate_binop(&ctx, inst);
         break;

      case D12_FUNC_INST_CAST:
         translate_cast(&ctx, inst);
         break;

      case D12_FUNC_INST_CMP:
      case D12_FUNC_INST_CMP2:
         translate_cmp(&ctx, inst);
         break;

      case D12_FUNC_INST_EXTRACTVAL:
         /* Struct-returning dx.op results are vectors in NIR, so an
          * extractvalue of member N is channel N.
          */
         if (inst->num_ops >= 1 && inst->num_extra >= 1) {
            nir_def *agg = get_def(&ctx, inst->ops[0]);
            unsigned idx = (unsigned)inst->extra[0];
            if (idx < agg->num_components) {
               def_result(&ctx, inst, nir_channel(&b, agg, idx));
            } else {
               mesa_loge("dxil: extractvalue %u of %u-component value",
                         idx, agg->num_components);
               trans_fail(&ctx, "extractvalue index past aggregate");
            }
         } else {
            trans_fail(&ctx, "malformed extractvalue");
         }
         break;

      case D12_FUNC_INST_SELECT:
      case D12_FUNC_INST_VSELECT:
         translate_select(&ctx, inst);
         break;

      case D12_FUNC_INST_LOAD: {
         /* SM 6.x reaches a resource by loading the global that stands
          * for it and converting the result with createHandleForLib. The
          * load itself carries no data: its value IS the resource, so it
          * resolves to that resource's binding slot exactly as
          * createHandle does.
          */
         if (inst->num_ops >= 1) {
            unsigned slot = resource_slot_for_global(&ctx, inst->ops[0]);
            if (slot != ~0u) {
               def_result(&ctx, inst, nir_imm_int(&b, slot));
               break;
            }
         }

         /* Otherwise it is real memory traffic: groupshared or scratch. */
         uint8_t space = inst->num_ops >= 1
                            ? pointer_for_value(&ctx, inst->ops[0])
                            : D12_PTR_NONE;
         if (space == D12_PTR_NONE) {
            mesa_loge("dxil: load from value %u kind %d addrspace %u type %u",
                      inst->ops[0],
                      inst->ops[0] < mod->num_values
                         ? (int)mod->values[inst->ops[0]].kind
                         : -1,
                      inst->ops[0] < mod->num_values
                         ? mod->values[inst->ops[0]].addr_space
                         : 0,
                      inst->ops[0] < mod->num_values
                         ? mod->values[inst->ops[0]].type
                         : 0);
            trans_fail(&ctx, "load from memory that is neither groupshared "
                             "nor local");
            break;
         }

         unsigned comps, bits;
         if (!memory_value_shape(mod, inst->result_type, &comps, &bits)) {
            trans_fail(&ctx, "loaded type has no memory shape");
            break;
         }

         uint32_t base = ctx.ptr_base[inst->ops[0]];
         bool dynamic = ctx.defs[inst->ops[0]] != NULL;
         nir_def *dyn = ctx.defs[inst->ops[0]];
         if (dynamic && nir_src_is_const(nir_src_for_ssa(dyn))) {
            base += nir_src_as_uint(nir_src_for_ssa(dyn));
            dyn = NULL;
            dynamic = false;
         }
         if (!dyn)
            dyn = nir_imm_int(&b, 0);

         nir_def *val;
         if (space == D12_PTR_SHARED) {
            val = nir_load_shared(&b, comps, bits, dyn, .base = base,
                                  .align_mul = bits / 8);
         } else if (space == D12_PTR_GLOBAL) {
            nir_def *addr = nir_iadd(
               &b, ctx.ptr_global[inst->ops[0]],
               nir_u2u64(&b, nir_iadd_imm(&b, dyn, base)));
            val = nir_load_global(&b, comps, bits, addr,
                                  .align_mul = bits / 8);
         } else {
            unsigned bytes = comps * (bits / 8);

            /* For a small single-block alloca whose constant stores are still
             * available as SSA, select those values directly. This avoids the
             * AGX stack path without changing multi-block memory semantics. */
            bool shadow = dynamic && ctx.num_blocks == 1 && bits == 32 &&
                          bytes > 0 && ctx.scratch_size <= 256;
            if (shadow) {
               nir_def *addr = nir_iadd_imm(&b, dyn, base);
               val = nir_undef(&b, comps, bits);
               bool any = false;

               for (unsigned off = 0; off + bytes <= ctx.scratch_size;
                    off += 4) {
                  nir_def *cells[4];
                  bool complete = true;
                  for (unsigned c = 0; c < comps; c++) {
                     cells[c] = ctx.scratch_shadow[off / 4 + c];
                     complete &= cells[c] != NULL;
                  }
                  if (!complete)
                     continue;

                  nir_def *candidate =
                     comps == 1 ? cells[0] : nir_vec(&b, cells, comps);
                  val = nir_bcsel(&b, nir_ieq_imm(&b, addr, off), candidate,
                                  val);
                  any = true;
               }

               shadow = any;
            }

            if (!shadow) {
               val = nir_load_scratch(&b, comps, bits,
                                      nir_iadd_imm(&b, dyn, base),
                                      .align_mul = bits / 8);
            }
         }
         def_result(&ctx, inst, val);
         break;
      }

      case D12_FUNC_INST_STORE: {
         /* [ptr, value] */
         if (inst->num_ops < 2) {
            trans_fail(&ctx, "malformed store");
            break;
         }

         uint8_t space = pointer_for_value(&ctx, inst->ops[0]);
         if (space == D12_PTR_NONE) {
            trans_fail(&ctx, "store to memory that is neither groupshared "
                             "nor local");
            break;
         }

         nir_def *val = get_def(&ctx, inst->ops[1]);
         if (!val) {
            trans_fail(&ctx, "stored value is not available");
            break;
         }

         uint32_t base = ctx.ptr_base[inst->ops[0]];
         bool dynamic = ctx.defs[inst->ops[0]] != NULL;
         nir_def *dyn = ctx.defs[inst->ops[0]];
         if (dynamic && nir_src_is_const(nir_src_for_ssa(dyn))) {
            base += nir_src_as_uint(nir_src_for_ssa(dyn));
            dyn = NULL;
            dynamic = false;
         }
         if (!dyn)
            dyn = nir_imm_int(&b, 0);

         if (space == D12_PTR_SHARED) {
            nir_store_shared(&b, val, dyn, .base = base,
                             .write_mask = BITFIELD_MASK(val->num_components),
                             .align_mul = val->bit_size / 8);
         } else if (space == D12_PTR_GLOBAL) {
            nir_def *addr = nir_iadd(
               &b, ctx.ptr_global[inst->ops[0]],
               nir_u2u64(&b, nir_iadd_imm(&b, dyn, base)));
            nir_store_global(&b, val, addr,
                             .write_mask = BITFIELD_MASK(val->num_components),
                             .align_mul = val->bit_size / 8);
         } else {
            if (!dynamic && ctx.num_blocks == 1 && val->bit_size == 32 &&
                (base & 3) == 0 &&
                base + val->num_components * 4 <= 256) {
               for (unsigned c = 0; c < val->num_components; c++)
                  ctx.scratch_shadow[base / 4 + c] =
                     nir_channel(&b, val, c);
            } else {
               memset(ctx.scratch_shadow, 0, sizeof(ctx.scratch_shadow));
            }

            nir_store_scratch(&b, val, nir_iadd_imm(&b, dyn, base),
                              .write_mask =
                                 BITFIELD_MASK(val->num_components),
                              .align_mul = val->bit_size / 8);
         }
         break;
      }

      case D12_FUNC_INST_ALLOCA: {
         /* LLVM 3.7's ALLOCA record names the ALLOCATED type directly
          * (BitcodeWriter pushes getAllocatedType), so the recorded type
          * is the object itself; carve it out of per-thread scratch. DXC
          * emits allocas only in the entry block with constant sizes, so
          * a running total is a correct allocator. Accept a pointer too,
          * in case a different writer recorded the result type instead.
          */
         if (inst->result_type >= mod->num_types) {
            trans_fail(&ctx, "alloca with no type");
            break;
         }

         uint32_t pointee =
            mod->types[inst->result_type].kind == D12_TYPE_POINTER
               ? mod->types[inst->result_type].ptr.pointee
               : inst->result_type;
         uint32_t size, align;
         if (!type_size_align(mod, pointee, &size, &align)) {
            trans_fail(&ctx, "alloca of a type with no memory layout");
            break;
         }

         ctx.scratch_size = ALIGN_POT(ctx.scratch_size, align);
         ctx.ptr_space[inst->result_value] = D12_PTR_SCRATCH;
         ctx.ptr_base[inst->result_value] = ctx.scratch_size;
         ctx.ptr_pointee[inst->result_value] = pointee;
         ctx.scratch_size += size;
         break;
      }

      case D12_FUNC_INST_GEP:
      case D12_FUNC_INST_GEP_OLD: {
         /* [ptr, idx...]: byte arithmetic over the pointee layout. The
          * result inherits the pointer's space; constant index chains
          * stay constant.
          */
         if (inst->num_ops < 2) {
            trans_fail(&ctx, "malformed getelementptr");
            break;
         }

         uint32_t ptr_id = inst->ops[0];
         uint8_t space = pointer_for_value(&ctx, ptr_id);
         if (space == D12_PTR_NONE) {
            mesa_loge("dxil: gep on value %u kind %d type %u",
                      ptr_id,
                      ptr_id < mod->num_values
                         ? (int)mod->values[ptr_id].kind
                         : -1,
                      ptr_id < mod->num_values ? mod->values[ptr_id].type : 0);
            trans_fail(&ctx, "getelementptr on a non-memory pointer");
            break;
         }

         uint32_t base = ctx.ptr_base[ptr_id];
         nir_def *dyn = ctx.defs[ptr_id];
         uint32_t cur = ctx.ptr_pointee[ptr_id];
         bool ok = true;

         for (unsigned gep_index = 1;
              gep_index < inst->num_ops && ok; gep_index++) {
            /* Index 1 strides over the whole pointee; later indices step
             * into it.
             */
            uint32_t stride, walk_align, next = cur;

            if (gep_index == 1) {
               ok = type_size_align(mod, cur, &stride, &walk_align);
            } else {
               const struct d12_type *t = &mod->types[cur];
               if (t->kind == D12_TYPE_ARRAY || t->kind == D12_TYPE_VECTOR) {
                  next = t->seq.elem;
                  ok = type_size_align(mod, next, &stride, &walk_align);
               } else if (t->kind == D12_TYPE_STRUCT_ANON ||
                          t->kind == D12_TYPE_STRUCT_NAMED) {
                  /* Struct indices must be constant in LLVM. */
                  uint64_t member;
                  uint32_t moff;
                  if (!const_uint(&ctx, inst->ops[gep_index], &member) ||
                      !struct_member_offset(mod, t, (unsigned)member, &moff,
                                            &next)) {
                     ok = false;
                     break;
                  }
                  base += moff;
                  cur = next;
                  continue;
               } else {
                  ok = false;
                  break;
               }
            }
            if (!ok)
               break;

            uint64_t const_idx;
            if (const_uint(&ctx, inst->ops[gep_index], &const_idx)) {
               base += (uint32_t)const_idx * stride;
            } else {
               nir_def *idx = get_def(&ctx, inst->ops[gep_index]);
               if (!idx) {
                  ok = false;
                  break;
               }
               idx = nir_u2u32(&b, idx);
               nir_def *scaled = nir_imul_imm(&b, idx, stride);
               dyn = dyn ? nir_iadd(&b, dyn, scaled) : scaled;
            }
            cur = next;
         }

         if (!ok) {
            trans_fail(&ctx, "getelementptr walk failed");
            break;
         }

         ctx.ptr_space[inst->result_value] = space;
         ctx.ptr_base[inst->result_value] = base;
         ctx.ptr_pointee[inst->result_value] = cur;
         ctx.ptr_global[inst->result_value] = ctx.ptr_global[ptr_id];
         if (dyn)
            ctx.defs[inst->result_value] = dyn;
         break;
      }

      case D12_FUNC_INST_ATOMICRMW: {
         /* LLVM atomicrmw, which DXC emits only for groupshared -- UAV
          * atomics go through dx.op.atomicBinOp. Operation numbering is
          * LLVM's RMWOperations (Bitcode/LLVMBitCodes.h).
          */
         if (inst->num_ops < 2 || inst->num_extra < 1) {
            trans_fail(&ctx, "malformed atomicrmw");
            break;
         }

         uint8_t space = pointer_for_value(&ctx, inst->ops[0]);
         if (space != D12_PTR_SHARED) {
            trans_fail(&ctx, "atomicrmw outside groupshared unsupported");
            break;
         }

         nir_atomic_op op;
         switch (inst->extra[0]) {
         case 0: op = nir_atomic_op_xchg; break;
         case 1: op = nir_atomic_op_iadd; break;
         case 3: op = nir_atomic_op_iand; break;
         case 5: op = nir_atomic_op_ior; break;
         case 6: op = nir_atomic_op_ixor; break;
         case 7: op = nir_atomic_op_imax; break;
         case 8: op = nir_atomic_op_imin; break;
         case 9: op = nir_atomic_op_umax; break;
         case 10: op = nir_atomic_op_umin; break;
         default:
            trans_fail(&ctx, "unhandled atomicrmw operation");
            break;
         }
         if (ctx.error)
            break;

         nir_def *val = get_def(&ctx, inst->ops[1]);
         if (!val)
            break;

         uint32_t base = ctx.ptr_base[inst->ops[0]];
         nir_def *dyn = ctx.defs[inst->ops[0]];
         if (!dyn)
            dyn = nir_imm_int(&b, 0);

         def_result(&ctx, inst,
                    nir_shared_atomic(&b, val->bit_size,
                                      nir_iadd_imm(&b, dyn, base), val,
                                      .atomic_op = op));
         break;
      }

      case D12_FUNC_INST_CMPXCHG: {
         /* Groupshared InterlockedCompareExchange. The LLVM result is
          * {oldvalue, i1 success}; DXC only ever extracts the value, so
          * the success channel is derived rather than stored.
          */
         if (inst->num_ops < 3) {
            trans_fail(&ctx, "malformed cmpxchg");
            break;
         }

         uint8_t space = pointer_for_value(&ctx, inst->ops[0]);
         if (space != D12_PTR_SHARED) {
            trans_fail(&ctx, "cmpxchg outside groupshared unsupported");
            break;
         }

         nir_def *cmp = get_def(&ctx, inst->ops[1]);
         nir_def *new_val = get_def(&ctx, inst->ops[2]);
         if (!cmp || !new_val)
            break;

         uint32_t base = ctx.ptr_base[inst->ops[0]];
         nir_def *dyn = ctx.defs[inst->ops[0]];
         if (!dyn)
            dyn = nir_imm_int(&b, 0);

         nir_def *old = nir_shared_atomic_swap(
            &b, cmp->bit_size, nir_iadd_imm(&b, dyn, base), cmp, new_val,
            .atomic_op = nir_atomic_op_cmpxchg);
         nir_def *ok = nir_ieq(&b, old, cmp);
         def_result(&ctx, inst,
                    nir_vec2(&b, old, nir_b2iN(&b, ok, old->bit_size)));
         break;
      }

      case D12_FUNC_INST_EXTRACTELT: {
         /* extractelement <n x T> %v, i32 %idx */
         if (inst->num_ops < 2) {
            trans_fail(&ctx, "malformed extractelement");
            break;
         }
         nir_def *vec = get_def(&ctx, inst->ops[0]);
         uint64_t idx;
         if (!vec) {
            break;
         } else if (const_uint(&ctx, inst->ops[1], &idx)) {
            def_result(&ctx, inst,
                       nir_channel(&b, vec, (unsigned)idx % vec->num_components));
         } else {
            def_result(&ctx, inst,
                       nir_vector_extract(&b, vec,
                                          get_def(&ctx, inst->ops[1])));
         }
         break;
      }

      case D12_FUNC_INST_INSERTELT: {
         /* insertelement <n x T> %v, T %elt, i32 %idx */
         if (inst->num_ops < 3) {
            trans_fail(&ctx, "malformed insertelement");
            break;
         }
         nir_def *vec = get_def(&ctx, inst->ops[0]);
         nir_def *elt = get_def(&ctx, inst->ops[1]);
         uint64_t idx;
         if (!vec || !elt) {
            break;
         } else if (const_uint(&ctx, inst->ops[2], &idx)) {
            def_result(&ctx, inst,
                       nir_vector_insert_imm(&b, vec, elt, (unsigned)idx));
         } else {
            def_result(&ctx, inst,
                       nir_vector_insert(&b, vec, elt,
                                         get_def(&ctx, inst->ops[2])));
         }
         break;
      }

      case D12_FUNC_INST_PHI:
         /* Read the register its predecessors wrote. */
         if (inst->result_value != ~0u && ctx.phi_reg[inst->result_value]) {
            def_result(&ctx, inst,
                       nir_load_reg(&b, ctx.phi_reg[inst->result_value]));
         } else {
            trans_fail(&ctx, "phi outside a multi-block function");
         }
         break;

      case D12_FUNC_INST_BR:
         translate_branch(&ctx, entry, inst);
         break;

      case D12_FUNC_INST_RET:
         /* In an unstructured function EVERY block needs a terminator,
          * including the last: the structurizer walks successors and
          * follows a null pointer off a block that has none. A return
          * branches to the epilogue, where the buffered outputs flush.
          */
         if (num_blocks > 1)
            nir_goto(&b, epilogue);
         break;

      case D12_FUNC_INST_UNREACHABLE:
         /* No outputs on a path that cannot execute. */
         if (num_blocks > 1)
            nir_goto(&b, b.impl->end_block);
         break;

      case D12_FUNC_DECLAREBLOCKS:
         break;

      default:
         mesa_loge("dxil: unhandled LLVM instruction code %u", inst->code);
         trans_fail(&ctx, "unhandled LLVM instruction");
         break;
      }
   }

   /* Flush what the shader buffered. With an epilogue block the stores
    * go there, since the final translated block ended with a goto and
    * nothing may follow a jump.
    */
   if (epilogue)
      b.cursor = nir_after_block(epilogue);
   flush_outputs(&ctx);

   if (epilogue && !ctx.error)
      nir_goto(&b, b.impl->end_block);

   /* Sizes discovered while translating: the launch words read shared
    * size from here, and the backend sizes its stack from scratch_size.
    */
   b.shader->info.shared_size = ctx.shared_size;
   b.shader->scratch_size = ctx.scratch_size;

   free(merged_insts);
   free(ctx.blocks);
   free(ctx.defs);
   free(ctx.phi_reg);
   free(ctx.ptr_space);
   free(ctx.ptr_base);
   free(ctx.ptr_pointee);
   free(ctx.ptr_global);
   free(ctx.handle_base);
   free(ctx.handle_heap);
   free(ctx.handle_kind_v);
   free(ctx.handle_stride_v);
   free(ctx.handle_valid);

   if (ctx.error) {
      *error = ctx.error;
      ralloc_free(b.shader);
      return NULL;
   }

   /*
    * Turn the unstructured CFG into structured NIR, then turn the phi
    * registers back into real SSA phis. Both passes are Mesa's own; the
    * translator's job was only to present the CFG in the shape they
    * expect.
    */
   if (num_blocks > 1)
      nir_lower_goto_ifs(b.shader);

   /* Registers appear from phis AND from builder-time loops (waveMatch),
    * so the SSA rebuild is unconditional; it is a no-op without them.
    */
   nir_lower_reg_intrinsics_to_ssa(b.shader);

   nir_validate_shader(b.shader, "after DXIL translation");
   return b.shader;
}

/*
 * agx_nir_lower_vbo leaves three intrinsics for the driver to answer:
 * the per-attribute base address, the robustness clamp, and (with
 * dynamic strides) the stride. Honeykrisp answers them from its root
 * table; we answer them from a parameter block, for the reason spelled
 * out at D12_UNIFORM_VB_PARAMS -- AGX's own vertex ABI starts at uniform
 * register 0 and would overwrite ours.
 *
 * The buffer handle is the DXIL input signature element id, because that
 * is what our loadInput lowering puts in nir_intrinsic_base and what
 * agx_nir_lower_vbo therefore indexes its key array by.
 */
static bool
lower_vbo_params(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned off, bits;

   /*
    * A PER-INSTANCE attribute is fetched at
    * (instance_id / divisor) + base_instance, so agx_nir_lower_vbo
    * introduces load_base_instance -- AFTER d12_dxil_lower_root has
    * already run, which is why it is answered here rather than there.
    * Missing it is not subtle: the backend asserts on an unhandled
    * intrinsic.
    *
    * D12_UNIFORM_DRAW holds {StartVertexLocation, StartInstanceLocation}
    * as two 32-bit words, the same pair SM 6.8's
    * StartInstanceLocation() reads.
    */
   if (intr->intrinsic == nir_intrinsic_load_base_instance) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def,
                      nir_load_preamble(b, 1, 32, .base = D12_UNIFORM_DRAW + 2));
      return true;
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_load_vbo_base_agx:
      off = D12_VB_OFF_BASE;
      bits = 64;
      break;
   case nir_intrinsic_load_attrib_clamp_agx:
      off = D12_VB_OFF_CLAMP;
      bits = 32;
      break;
   case nir_intrinsic_load_vbo_stride_agx:
      off = D12_VB_OFF_STRIDE;
      bits = 32;
      break;
   default:
      return false;
   }

   /* The handle is always a constant: agx_nir_lower_vbo builds it with
    * nir_imm_int from the attribute index it just resolved.
    */
   if (!nir_src_is_const(intr->src[0]))
      return false;

   unsigned idx = nir_src_as_uint(intr->src[0]);
   if (idx >= D12_MAX_VB_ATTRIBS)
      return false;

   /* This pass runs after d12_dxil_lower_root, so it reads the uniform
    * itself rather than emitting an intrinsic for that pass to answer.
    */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *params =
      nir_load_preamble(b, 1, 64, .base = D12_UNIFORM_VB_PARAMS);
   nir_def *addr = nir_iadd_imm(b, params, off + idx * (bits / 8));

   nir_def_replace(&intr->def,
                   nir_load_global_constant(b, 1, bits, addr,
                                            .align_mul = bits / 8));
   return true;
}

enum d12_signature_kind {
   D12_SIGNATURE_INPUT,
   D12_SIGNATURE_OUTPUT,
   D12_SIGNATURE_PATCH,
};

/* Normalize DXIL's component enum to the serialized D3D signature classes:
 * 1 = uint, 2 = sint, 3 = float. Keep minimum-width compatibility separate. */
static void
d12_dxil_signature_type(uint32_t dxil_type, uint32_t *type,
                        uint32_t *min_precision)
{
   *type = 0;
   *min_precision = 0;

   switch (dxil_type) {
   case 1: /* i1 */
   case 3: /* u16 */
   case 5: /* u32 */
      *type = 1;
      if (dxil_type == 3)
         *min_precision = 5;
      break;
   case 2: /* i16 */
   case 4: /* i32 */
      *type = 2;
      if (dxil_type == 2)
         *min_precision = 4;
      break;
   case 8: /* f16 */
   case 9: /* f32 */
      *type = 3;
      if (dxil_type == 8)
         *min_precision = 1;
      break;
   default:
      break;
   }
}

/* Signature reads parse the whole module, and a graphics PSO asks for the
 * input, output and patch signatures of each stage several times while it
 * validates stage links, input layouts and flat transport. Parse once per
 * bytecode blob, keyed on its content hash, and answer later reads from a
 * small cache. */
#define D12_SIG_CACHE_ENTRIES 16

struct d12_sig_cache_entry {
   uint64_t hash;
   size_t size_B;
   bool valid;
   unsigned count[3];
   struct d12_input_sig_elem elems[3][D12_MAX_SIG_ELEMENTS];
};

static simple_mtx_t d12_sig_cache_lock = SIMPLE_MTX_INITIALIZER;
static struct d12_sig_cache_entry d12_sig_cache[D12_SIG_CACHE_ENTRIES];
static unsigned d12_sig_cache_next;

static void
d12_dxil_copy_signature(const struct d12_sig_element *sig, unsigned count,
                        struct d12_input_sig_elem *out)
{
   for (unsigned i = 0; i < count; i++) {
      const struct d12_sig_element *se = &sig[i];

      out[i].id = se->id;
      out[i].register_index = se->start_row;
      out[i].semantic_index = se->semantic_index;
      out[i].start_component = se->start_col;
      out[i].component_count = se->cols;
      out[i].rows = MAX2(se->rows, 1);
      out[i].system_value = se->system_value;
      d12_dxil_signature_type(se->component_type, &out[i].component_type,
                              &out[i].min_precision);
      out[i].name[0] = '\0';
      if (se->name) {
         strncpy(out[i].name, se->name, sizeof(out[i].name) - 1);
         out[i].name[sizeof(out[i].name) - 1] = '\0';
      }
   }
}

static unsigned
d12_dxil_signature(const void *dxil, size_t size_B,
                   struct d12_input_sig_elem *out, unsigned max,
                   enum d12_signature_kind kind)
{
   const uint8_t *bc;
   size_t bc_size;
   const char *why = NULL;

   if (!d12_dxil_find_bitcode(dxil, size_B, &bc, &bc_size, &why))
      return 0;

   const uint64_t hash = XXH64(dxil, size_B, 0);

   simple_mtx_lock(&d12_sig_cache_lock);
   for (unsigned i = 0; i < D12_SIG_CACHE_ENTRIES; i++) {
      const struct d12_sig_cache_entry *e = &d12_sig_cache[i];
      if (!e->valid || e->hash != hash || e->size_B != size_B)
         continue;
      unsigned n = MIN2(e->count[kind], max);
      memcpy(out, e->elems[kind], n * sizeof(*out));
      simple_mtx_unlock(&d12_sig_cache_lock);
      return n;
   }
   simple_mtx_unlock(&d12_sig_cache_lock);

   struct d12_module *mod = d12_dxil_module_alloc();
   if (!mod)
      return 0;
   if (!d12_dxil_module_parse(mod, bc, bc_size)) {
      d12_dxil_module_free(mod);
      return 0;
   }

   struct d12_sig_cache_entry entry = {
      .hash = hash,
      .size_B = size_B,
      .valid = true,
      .count = {
         MIN2(mod->num_input_sig, D12_MAX_SIG_ELEMENTS),
         MIN2(mod->num_output_sig, D12_MAX_SIG_ELEMENTS),
         MIN2(mod->num_patch_sig, D12_MAX_SIG_ELEMENTS),
      },
   };
   d12_dxil_copy_signature(mod->input_sig, entry.count[D12_SIGNATURE_INPUT],
                           entry.elems[D12_SIGNATURE_INPUT]);
   d12_dxil_copy_signature(mod->output_sig, entry.count[D12_SIGNATURE_OUTPUT],
                           entry.elems[D12_SIGNATURE_OUTPUT]);
   d12_dxil_copy_signature(mod->patch_sig, entry.count[D12_SIGNATURE_PATCH],
                           entry.elems[D12_SIGNATURE_PATCH]);
   d12_dxil_module_free(mod);

   unsigned n = MIN2(entry.count[kind], max);
   memcpy(out, entry.elems[kind], n * sizeof(*out));

   simple_mtx_lock(&d12_sig_cache_lock);
   d12_sig_cache[d12_sig_cache_next] = entry;
   d12_sig_cache_next = (d12_sig_cache_next + 1) % D12_SIG_CACHE_ENTRIES;
   simple_mtx_unlock(&d12_sig_cache_lock);
   return n;
}

unsigned
d12_dxil_input_signature(const void *dxil, size_t size_B,
                         struct d12_input_sig_elem *out, unsigned max)
{
   return d12_dxil_signature(dxil, size_B, out, max, D12_SIGNATURE_INPUT);
}

unsigned
d12_dxil_output_signature(const void *dxil, size_t size_B,
                          struct d12_input_sig_elem *out, unsigned max)
{
   return d12_dxil_signature(dxil, size_B, out, max, D12_SIGNATURE_OUTPUT);
}

unsigned
d12_dxil_patch_signature(const void *dxil, size_t size_B,
                         struct d12_input_sig_elem *out, unsigned max)
{
   return d12_dxil_signature(dxil, size_B, out, max, D12_SIGNATURE_PATCH);
}

bool
d12_dxil_lower_vbo_params(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_vbo_params,
                                     nir_metadata_control_flow, NULL);
}

/*
 * agx_nir_lower_uvs emits load_uvs_index_agx for indexed output stores
 * and leaves it "to be lowered by the driver": Honeykrisp reads the
 * index out of its root descriptor because linking decides the layout at
 * draw time. Ours is decided at compile time by the same
 * agx_nir_lower_uvs call, so the index is a constant from the layout it
 * just produced.
 */
static bool
lower_uvs_index(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned *slots = data;

   if (intr->intrinsic != nir_intrinsic_load_uvs_index_agx)
      return false;

   gl_varying_slot loc = nir_intrinsic_io_semantics(intr).location;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, slots[loc], 16));
   return true;
}

bool
d12_dxil_lower_uvs_index(nir_shader *nir, const unsigned *slots)
{
   return nir_shader_intrinsics_pass(nir, lower_uvs_index,
                                     nir_metadata_control_flow,
                                     (void *)slots);
}

static bool
lower_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /*
    * The geometry count pass asks for the INPUT TOPOLOGY. Honeykrisp
    * answers per draw because it supports every primitive type; this
    * driver only accepts triangle lists (IASetPrimitiveTopology refuses
    * anything else), so the answer is fixed and known at compile time.
    */
   /*
    * D3D12 has no provoking-vertex control: the convention is always
    * first-vertex. Honeykrisp reads this per pipeline because Vulkan can
    * switch it; here it is a constant of the API.
    */
   if (intr->intrinsic == nir_intrinsic_load_provoking_last) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int(b, 0));
      return true;
   }

   /* D3D12 exposes no triangle-fan topology, so AGX's special first-fan
    * coefficient layout is never selected. Flat interpolation can therefore
    * use the ordinary provoking-vertex coefficient unconditionally. */
   if (intr->intrinsic == nir_intrinsic_load_is_first_fan_agx) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_false(b));
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_input_topology_poly) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int(b, MESA_PRIM_TRIANGLES));
      return true;
   }

   /*
    * Fragment shaders with memory side effects get a synthetic depth write
    * from agx_nir_lower_frag_sidefx. Honeykrisp sources this flag from its
    * draw root; the Gallium AGX path uses zero when no API depth-NEVER
    * workaround is wired. D3D12 currently has no such draw flag, so use the
    * same conservative zero here. Leaving the AGX intrinsic intact makes
    * NIR divergence analysis abort before compilation.
    */
   if (intr->intrinsic == nir_intrinsic_load_depth_never_agx) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_intN_t(b, 0, 16));
      return true;
   }

   /* Driver-owned sysvals Honeykrisp reads from its root descriptor at
    * draw time. Ours are compile-time constants: the API sample mask comes
    * from the PSO, and a monolithic fragment shader is the final shader part,
    * so it must run tests for every sample that survives discard. AGX requires
    * each sample affected by sample_mask to be killed or tested exactly once;
    * spelling tests_zs as zero made a false conditional discard suppress the
    * following tilebuffer store.
    */
   if (intr->intrinsic == nir_intrinsic_load_api_sample_mask_agx) {
      b->cursor = nir_before_instr(&intr->instr);
      const uint16_t sample_mask = data ? *(const uint16_t *)data : 0xffff;
      nir_def_replace(&intr->def, nir_imm_intN_t(b, sample_mask, 16));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_shader_part_tests_zs_agx) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_intN_t(b, 0xff, 16));
      return true;
   }

   /* poly's software input assembly adds the base vertex after indexing.
    * D12_UNIFORM_DRAW already carries it as a 32-bit preamble register --
    * the same one D12_DXOP_START_VERTEX_LOCATION reads.
    */
   if (intr->intrinsic == nir_intrinsic_load_first_vertex) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def,
                      nir_load_preamble(b, 1, 32, .base = D12_UNIFORM_DRAW));
      return true;
   }

   /* The software vertex shader's input assembly reads the index size out
    * of poly_vertex_params rather than from a uniform of its own. It is a
    * uint32 at offset 8 (poly/geometry.h), so answer it with a load from
    * the parameter block instead of a preamble register.
    */
   if (intr->intrinsic == nir_intrinsic_load_index_size_poly) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *params =
         nir_load_preamble(b, 1, 64, .base = D12_UNIFORM_VERTEX_PARAMS);
      /* poly_vertex_params: uint64 index_buffer, then uint32
       * index_size_B -- offset 8. The struct is PACKED and static-asserted
       * at 16 words, so the literal is safe and avoids pulling the
       * OpenCL-flavoured header into the translator.
       */
      nir_def *addr = nir_iadd_imm(b, params, 8);
      nir_def_replace(&intr->def,
                      nir_load_global_constant(b, 1, 32, addr, .align_mul = 4));
      return true;
   }

   unsigned base;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_root_agx:
      base = b->shader->info.stage == MESA_SHADER_FRAGMENT
                ? D12_UNIFORM_FRAGMENT_ROOT
                : D12_UNIFORM_ROOT;
      break;
   case nir_intrinsic_load_descriptor_heap_agx:
      base = D12_UNIFORM_HEAP;
      break;
   case nir_intrinsic_load_custom_border_table_agx:
      base = D12_UNIFORM_CUSTOM_BORDERS;
      break;
   case nir_intrinsic_load_mesh_param_buffer_agx:
      base = D12_UNIFORM_MESH_PARAMS;
      break;
   case nir_intrinsic_load_node_param_buffer_agx:
      base = D12_UNIFORM_NODE_PARAMS;
      break;
   /* Emitted by the GS/tess lowering in src/poly, which leaves them for
    * the driver to answer.
    */
   case nir_intrinsic_load_geometry_param_buffer_poly:
      base = D12_UNIFORM_GEOM_PARAMS;
      break;
   case nir_intrinsic_load_tess_param_buffer_poly:
      base = D12_UNIFORM_TESS_PARAMS;
      break;
   case nir_intrinsic_load_vs_outputs_poly:
      base = D12_UNIFORM_VS_OUTPUTS;
      break;
   case nir_intrinsic_load_vertex_param_buffer_poly:
      base = D12_UNIFORM_VERTEX_PARAMS;
      break;
   case nir_intrinsic_load_stat_query_address_poly:
      base = D12_UNIFORM_STAT_QUERY;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep = nir_load_preamble(b, 1, 64, .base = base);
   nir_def_replace(&intr->def, rep);
   return true;
}

bool
d12_dxil_lower_root(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_root,
                                     nir_metadata_control_flow, NULL);
}

bool
d12_dxil_lower_root_mask(nir_shader *nir, uint16_t sample_mask)
{
   return nir_shader_intrinsics_pass(nir, lower_root,
                                     nir_metadata_control_flow,
                                     &sample_mask);
}

static bool
lower_sample_positions(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_sample_positions_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_imm_int(b, *(const uint32_t *)data));
   return true;
}

bool
d12_dxil_lower_sample_positions(nir_shader *nir, uint32_t packed_positions)
{
   return nir_shader_intrinsics_pass(nir, lower_sample_positions,
                                     nir_metadata_control_flow,
                                     &packed_positions);
}


/*
 * Vertex-shader sysvals for the SOFTWARE vertex shader -- the compute
 * program that runs when a geometry or tessellation shader follows.
 *
 * poly_nir_lower_sw_vs answers load_vertex_id and load_instance_id, but
 * poly_nir_lower_vs_before_gs also emits load_vertex_id_zero_base to
 * index the output buffer and nothing lowers that. In a compute dispatch
 * it is not a sysval at all: it reaches the backend as an undefined
 * register, so every thread computes its output address from garbage and
 * the vertex buffer stays zero. The zero-based index IS the global
 * invocation ID -- that is what "zero based" means here.
 *
 * Exported so the standalone tool runs exactly what the driver runs. A
 * tool that lowers a different set of passes disassembles a shader the
 * driver never builds (docs/findings.md).
 */
static bool
mark_sw_vs_vertex_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_load_vertex_id_zero_base(b));
   return true;
}

bool
d12_dxil_mark_sw_vs_vertex_id(nir_shader *nir)
{
   /* Mark the APPLICATION's SV_VertexID before agx_nir_lower_vbo inserts its
    * own load_vertex_id for attribute addressing.  D3D12's SV_VertexID is the
    * raw index (or zero-based invocation for a non-indexed draw), while the
    * VBO address must still include StartVertex/BaseVertex. */
   return nir_shader_intrinsics_pass(nir, mark_sw_vs_vertex_id,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_sw_vs_vertex_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *linear =
      nir_channel(b, nir_load_global_invocation_id(b, 32), 0);

   /* Reuse Poly's robust indexed-input assembly, then remove the D3D draw
    * bias it adds for attribute addressing.  Unsigned wrap makes this exact
    * for negative BaseVertexLocation as well. */
   nir_def *effective = poly_nir_load_vertex_id(b, linear);
   nir_def *raw = nir_isub(b, effective, nir_load_first_vertex(b));
   nir_def_replace(&intr->def, raw);
   return true;
}

bool
d12_dxil_lower_sw_vs_vertex_id(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_sw_vs_vertex_id,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_sw_vs_zero_base(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_channel(b, nir_load_global_invocation_id(b, 32), 0));
   return true;
}

bool
d12_dxil_lower_sw_vs(nir_shader *nir)
{
   bool progress = false;
   NIR_PASS(progress, nir, poly_nir_lower_sw_vs);
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_sw_vs_zero_base,
            nir_metadata_control_flow, NULL);
   return progress;
}
