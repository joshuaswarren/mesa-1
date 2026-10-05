/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DXIL front end for d3d12agx: DXBC container + LLVM 3.7-era bitstream
 * reader. This is the bottom of the "no SPIR-V" shader path —
 *
 *    DXIL container -> bitstream -> module -> NIR -> AGX
 *
 * with no SPIR-V and no external compiler anywhere in it.
 *
 * Format references (all public): LLVM's BitCodeFormat documentation for
 * the bitstream container, and DirectXShaderCompiler's docs/DXIL.rst for
 * the DXIL layer on top. Block/abbrev numbering cross-checked against
 * Mesa's own MIT-licensed DXIL *writer* in src/microsoft/compiler, which
 * emits this exact format.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "shader_enums.h"

/* Bitstream container magic: 'B','C',0xC0,0xDE */
#define D12_BITCODE_MAGIC 0xDEC04342u

/* DXBC container magic: 'D','X','B','C' */
#define D12_DXBC_MAGIC 0x43425844u

/* Standard and application block IDs (LLVM bitcode) */
enum d12_block_id {
   D12_BLOCK_BLOCKINFO = 0,
   D12_BLOCK_MODULE = 8,
   D12_BLOCK_PARAMATTR = 9,
   D12_BLOCK_PARAMATTR_GROUP = 10,
   D12_BLOCK_CONSTANTS = 11,
   D12_BLOCK_FUNCTION = 12,
   D12_BLOCK_VALUE_SYMTAB = 14,
   D12_BLOCK_METADATA = 15,
   D12_BLOCK_METADATA_ATTACHMENT = 16,
   D12_BLOCK_TYPE = 17,
};

/* Fixed abbreviation IDs present in every block */
enum d12_abbrev_id {
   D12_ABBREV_END_BLOCK = 0,
   D12_ABBREV_ENTER_SUBBLOCK = 1,
   D12_ABBREV_DEFINE_ABBREV = 2,
   D12_ABBREV_UNABBREV_RECORD = 3,
   D12_ABBREV_FIRST_APPLICATION = 4,
};

/* BLOCKINFO record codes */
enum d12_blockinfo_code {
   D12_BLOCKINFO_SETBID = 1,
   D12_BLOCKINFO_BLOCKNAME = 2,
   D12_BLOCKINFO_SETRECORDNAME = 3,
};

/* Abbreviation operand encodings */
enum d12_abbrev_encoding {
   D12_ENC_LITERAL = 0,
   D12_ENC_FIXED = 1,
   D12_ENC_VBR = 2,
   D12_ENC_ARRAY = 3,
   D12_ENC_CHAR6 = 4,
   D12_ENC_BLOB = 5,
};

struct d12_abbrev_op {
   enum d12_abbrev_encoding enc;
   uint64_t value; /* literal value, or bit width for FIXED/VBR */
};

#define D12_MAX_ABBREV_OPS 32

struct d12_abbrev {
   struct d12_abbrev_op ops[D12_MAX_ABBREV_OPS];
   unsigned num_ops;
};

#define D12_MAX_ABBREVS 128

struct d12_abbrev_list {
   struct d12_abbrev abbrevs[D12_MAX_ABBREVS];
   unsigned count;
};

/* Per-block-ID abbreviations declared in the BLOCKINFO block */
#define D12_MAX_BLOCKINFO_BLOCKS 16

struct d12_blockinfo {
   struct {
      unsigned block_id;
      struct d12_abbrev_list abbrevs;
   } entries[D12_MAX_BLOCKINFO_BLOCKS];
   unsigned count;
};

/*
 * A decoded record. The operand storage belongs to the bitstream walker,
 * which grows it on demand: a CST_DATA lookup table or a long MD_STRING
 * (source paths, embedded HLSL) can carry far more operands than any
 * fixed scratch would allow. Records are transient -- a visitor must copy
 * what it keeps before returning.
 */
#define D12_LIMIT_RECORD_OPS (1u << 26)

struct d12_record {
   uint32_t code;
   uint64_t *ops;
   unsigned num_ops;

   /* BLOB payload, if the abbreviation ended in one */
   const uint8_t *blob;
   size_t blob_size;
};

struct d12_bitstream {
   const uint8_t *data;
   size_t size_B;

   /* Absolute bit position */
   uint64_t pos_bits;

   /* Record operand scratch, owned by the walk and reused by every record
    * it decodes. Grown by record_push. */
   uint64_t *record_ops;
   unsigned record_ops_capacity;

   bool error;
   const char *error_msg;
};

/*
 * Container handling: returns a pointer to raw LLVM bitcode within the
 * input. Accepts either a DXBC container (extracts its DXIL part) or bare
 * bitcode. Returns false and sets *why on malformed input.
 */
bool d12_dxil_find_bitcode(const void *data, size_t size_B,
                           const uint8_t **out_bitcode, size_t *out_size_B,
                           const char **why);

/*
 * Recover the shader stage from a DXBC container's DXIL program version.
 * Returns false for bare bitcode (no container to ask).
 */
bool d12_dxil_container_kind(const void *data, size_t size_B,
                             uint32_t *out_kind);
const char *d12_dxil_kind_name(uint32_t kind);
bool d12_dxil_container_stage(const void *data, size_t size_B,
                              mesa_shader_stage *out_stage);

/* Bitstream primitives */
void d12_bitstream_init(struct d12_bitstream *bs, const uint8_t *data,
                        size_t size_B);
uint32_t d12_bitstream_read(struct d12_bitstream *bs, unsigned bits);
uint64_t d12_bitstream_read64(struct d12_bitstream *bs, unsigned bits);
uint64_t d12_bitstream_read_vbr(struct d12_bitstream *bs, unsigned bits);
void d12_bitstream_align32(struct d12_bitstream *bs);
bool d12_bitstream_at_end(const struct d12_bitstream *bs);

/*
 * Block walking. The visitor is called for every record encountered; return
 * false from it to abort the walk. Sub-blocks are entered depth-first;
 * enter/exit callbacks may be NULL.
 */
struct d12_block_visitor {
   void *user;
   bool (*enter_block)(void *user, unsigned block_id, unsigned depth);
   bool (*record)(void *user, unsigned block_id, unsigned depth,
                  const struct d12_record *rec, unsigned abbrev_id);
   bool (*exit_block)(void *user, unsigned block_id, unsigned depth);
};

/*
 * Walk a whole bitcode module (magic already at bs position 0). Returns
 * false on malformed input or if the visitor aborted; bs->error_msg
 * explains malformed input.
 */
bool d12_dxil_walk(struct d12_bitstream *bs,
                   const struct d12_block_visitor *visitor);

/*
 * ---------------------------------------------------------------------
 * Module decoding (G4b)
 *
 * Record codes below are the LLVM 3.7 numbering, cross-checked against
 * Mesa's DXIL writer (src/microsoft/compiler/dxil_module.c).
 * ---------------------------------------------------------------------
 */

enum d12_module_code {
   D12_MODULE_VERSION = 1,
   D12_MODULE_TRIPLE = 2,
   D12_MODULE_DATALAYOUT = 3,
   D12_MODULE_GLOBALVAR = 7,
   D12_MODULE_FUNCTION = 8,
   D12_MODULE_ALIAS = 9,
};

enum d12_type_code {
   D12_TYPE_NUMENTRY = 1,
   D12_TYPE_VOID = 2,
   D12_TYPE_FLOAT = 3,
   D12_TYPE_DOUBLE = 4,
   D12_TYPE_LABEL = 5,
   D12_TYPE_OPAQUE = 6,
   D12_TYPE_INTEGER = 7,
   D12_TYPE_POINTER = 8,
   D12_TYPE_HALF = 10,
   D12_TYPE_ARRAY = 11,
   D12_TYPE_VECTOR = 12,
   D12_TYPE_METADATA = 16,
   D12_TYPE_STRUCT_ANON = 18,
   D12_TYPE_STRUCT_NAME = 19,
   D12_TYPE_STRUCT_NAMED = 20,
   D12_TYPE_FUNCTION = 21,
};

enum d12_const_code {
   D12_CST_SETTYPE = 1,
   D12_CST_NULL = 2,
   D12_CST_UNDEF = 3,
   D12_CST_INTEGER = 4,
   D12_CST_WIDE_INTEGER = 5,
   D12_CST_FLOAT = 6,
   D12_CST_AGGREGATE = 7,
   D12_CST_STRING = 8,
   D12_CST_CSTRING = 9,
   D12_CST_CE_CAST = 11,
   D12_CST_CE_GEP = 12,
   D12_CST_CE_INBOUNDS_GEP = 20,
   D12_CST_DATA = 22,
};

enum d12_func_code {
   D12_FUNC_DECLAREBLOCKS = 1,
   D12_FUNC_INST_BINOP = 2,
   D12_FUNC_INST_CAST = 3,
   D12_FUNC_INST_SELECT = 5,
   D12_FUNC_INST_EXTRACTELT = 6,
   D12_FUNC_INST_INSERTELT = 7,
   D12_FUNC_INST_SHUFFLEVEC = 8,
   D12_FUNC_INST_CMP = 9,
   D12_FUNC_INST_RET = 10,
   D12_FUNC_INST_BR = 11,
   D12_FUNC_INST_SWITCH = 12,
   D12_FUNC_INST_UNREACHABLE = 15,
   D12_FUNC_INST_PHI = 16,
   D12_FUNC_INST_ALLOCA = 19,
   D12_FUNC_INST_LOAD = 20,
   D12_FUNC_INST_EXTRACTVAL = 26,
   D12_FUNC_INST_CMP2 = 28,
   D12_FUNC_INST_VSELECT = 29,
   D12_FUNC_INST_GEP_OLD = 30,
   D12_FUNC_DEBUG_LOC_AGAIN = 33,
   D12_FUNC_INST_CALL = 34,
   D12_FUNC_DEBUG_LOC = 35,
   D12_FUNC_INST_ATOMICRMW = 38,
   D12_FUNC_INST_GEP = 43,
   D12_FUNC_INST_STORE = 44,
   D12_FUNC_INST_CMPXCHG = 46,
};

enum d12_metadata_code {
   D12_MD_STRING = 1,
   D12_MD_VALUE = 2,
   D12_MD_NODE = 3,
   D12_MD_NAME = 4,
   D12_MD_KIND = 6,
   D12_MD_NAMED_NODE = 10,
};

enum d12_vst_code {
   D12_VST_ENTRY = 1,
   D12_VST_BBENTRY = 2,
};

/* Decoded LLVM type */
struct d12_type {
   enum d12_type_code kind;

   union {
      unsigned int_bits;         /* INTEGER */
      struct {                   /* POINTER */
         uint32_t pointee;
         uint32_t addr_space;
      } ptr;
      struct {                   /* ARRAY / VECTOR */
         uint64_t count;
         uint32_t elem;
      } seq;
      struct {                   /* FUNCTION */
         uint32_t ret;
         uint32_t *params;      /* module arena */
         unsigned num_params;
      } func;
      struct {                   /* STRUCT_ANON / STRUCT_NAMED */
         uint32_t *members;     /* module arena */
         unsigned num_members;
      } strct;
   };

   const char *name; /* struct name, if any (points into the module's arena) */
};

enum d12_value_kind {
   D12_VALUE_UNKNOWN = 0,
   D12_VALUE_GLOBAL,
   D12_VALUE_FUNCTION,
   D12_VALUE_CONSTANT,
   D12_VALUE_ARGUMENT,
   D12_VALUE_INSTRUCTION,
};

/*
 * LLVM address spaces as DXC emits them. Only groupshared is distinguished:
 * everything else a shader declares -- constant lookup tables, the globals
 * that stand for resources -- lands in 0.
 */
#define D12_ADDR_SPACE_DEFAULT     0
#define D12_ADDR_SPACE_GROUPSHARED 3

struct d12_value {
   enum d12_value_kind kind;

   /*
    * Index into the type table, and for a GLOBAL it is the type of the
    * VALUE -- [64 x i32] for `groupshared uint scratch[64]` -- not a
    * pointer to it. LLVM 3.7 moved the address space out of the type and
    * into the record, so `type` alone cannot tell groupshared from
    * anything else; `addr_space` below is the only thing that can.
    */
   uint32_t type;
   uint32_t addr_space; /* GLOBAL only; see D12_ADDR_SPACE_* */

   const char *name;   /* from VALUE_SYMTAB, may be NULL */

   /* GLOBAL only: LLVM's initid (value id + 1, 0 = no initializer) and
    * the isconst flag bit. Distinguishes a mutable static (per-thread
    * storage) from a constant lookup table (data we must materialize).
    */
   uint32_t global_init_id;
   bool global_is_const;

   /* CONSTANT only: a constant-expression GEP. agg_elems[0] is the base
    * value, the rest are index value ids.
    */
   bool is_ce_gep;

   /* CONSTANT only: CST_DATA payload -- the raw elements of a constant
    * array initializer (an HLSL `static const` lookup table). Owned by
    * the module and freed with it.
    */
   uint64_t *data_elems;
   unsigned num_data_elems;

   /* Constants */
   bool is_undef;
   bool is_null;
   uint64_t const_int;   /* CST_INTEGER (already un-rotated) */
   uint64_t const_float; /* CST_FLOAT raw bits */

   /* CST_AGGREGATE: element value ids. Needed because SM 6.6 passes the
    * resource binding to createHandleFromBinding as a constant struct.
    */
   uint32_t agg_elems[8];
   unsigned num_agg_elems;

   /* Functions */
   uint32_t func_index; /* index into module->functions, for FUNCTION */
};

/*
 * Hard ceilings on dynamically grown tables. These are sanity bounds on
 * malformed input, not capacities: every table starts small and doubles.
 * Real uber-shaders run to tens of thousands of instructions and values,
 * which the old fixed arrays (4096 instructions, 8192 values) rejected
 * outright at PSO creation.
 */
#define D12_LIMIT_TYPES     (1u << 20)
#define D12_LIMIT_VALUES    (1u << 24)
#define D12_LIMIT_FUNCTIONS (1u << 20)
#define D12_LIMIT_METADATA  (1u << 24)
#define D12_LIMIT_INSTS     (1u << 24)
#define D12_LIMIT_BLOCKS    (1u << 20)

/* A phi's incoming edge: which value arrives from which basic block. */

struct d12_phi_src {
   uint32_t value; /* ~0u when it could not be resolved */
   uint32_t block;
};

/* One decoded instruction, operands kept as value indices where resolved.
 * The operand and phi-source arrays live in the module arena, sized from
 * the record: a dx.op.traceRay call carries 17 operands, a phi merging a
 * wide switch has one source per case. */
struct d12_inst {
   enum d12_func_code code;
   uint32_t result_type;              /* type index, or ~0 if void */
   uint32_t result_value;             /* value index it defines, or ~0 */
   uint32_t *ops;                     /* value indices (module arena) */
   unsigned num_ops;
   uint64_t extra[4];                 /* opcode-specific immediates */
   unsigned num_extra;

   /* Which basic block this instruction belongs to, filled in by the
    * parser: LLVM marks block boundaries only implicitly, by terminators.
    */
   uint32_t block;

   /* PHI only */
   struct d12_phi_src *phi_srcs;      /* module arena */
   unsigned num_phi_srcs;
};

/* Does this instruction end a basic block? */
static inline bool
d12_inst_is_terminator(const struct d12_inst *inst)
{
   switch (inst->code) {
   case D12_FUNC_INST_RET:
   case D12_FUNC_INST_BR:
   case D12_FUNC_INST_SWITCH:
   case D12_FUNC_INST_UNREACHABLE:
      return true;
   default:
      return false;
   }
}

struct d12_function {
   const char *name;
   uint32_t type;        /* function type index */
   bool is_declaration;  /* proto only (dx.op intrinsics are declarations) */
   uint32_t value_index; /* its slot in the module value list */

   /* Body (definitions only), grown as records arrive */
   struct d12_inst *insts;
   unsigned num_insts;
   unsigned inst_capacity;
   unsigned num_blocks;
   unsigned first_value; /* value index of this function's first argument */
};

struct d12_md_node {
   uint64_t *ops;     /* metadata ids (already un-biased), module arena */
   bool *op_is_null;  /* module arena */
   unsigned num_ops;
};

struct d12_metadata {
   enum d12_metadata_code kind;
   const char *string;         /* MD_STRING */
   struct d12_md_node node;    /* MD_NODE */
   uint32_t value_type;        /* MD_VALUE */
   uint64_t value_id;          /* MD_VALUE */
};

struct d12_named_md {
   const char *name;
   uint64_t *nodes;   /* module arena */
   unsigned num_nodes;
};

/*
 * One DXIL signature element. Layout of the metadata node, read out of
 * real dxc output:
 *   [0] id  [1] semantic name  [2] component type  [3] system-value kind
 *   [4] semantic indices  [5] interpolation mode  [6] rows  [7] cols
 *   [8] start row  [9] start col  [10] name-value pairs
 * The semantic NAME is the dependable key (SV_Position, SV_Target,
 * SV_VertexID, TEXCOORD, ...), so that is what we match on.
 */
struct d12_sig_element {
   uint32_t id;         /* what loadInput/storeOutput reference */
   const char *name;    /* semantic name */
   uint32_t semantic_index; /* SemanticIndex: TEXCOORD0 vs TEXCOORD1 */
   uint32_t component_type;
   uint32_t system_value;
   uint32_t interpolation;
   uint32_t rows;
   uint32_t start_row;
   uint32_t start_col;
   uint32_t cols;
};

#define D12_MAX_SIG_ELEMENTS 32

#define D12_MAX_NAMED_MD  32

/*
 * Declared resources, from the dx.resources named metadata: four lists,
 * one per class, each entry describing a range of shader registers.
 *
 * This is what lets a shader register be resolved to a binding slot. D3D12
 * register namespaces are per class -- t0, u0 and b0 are three different
 * resources -- so a handle cannot be identified by its number alone.
 */
enum d12_resource_class {
   D12_RES_SRV = 0,
   D12_RES_UAV = 1,
   D12_RES_CBV = 2,
   D12_RES_SAMPLER = 3,
   D12_RES_CLASS_COUNT = 4,
};

/*
 * DXIL ResourceKind (subset). The shape decides how a buffer access is
 * addressed, and it is NOT recoverable from which dx.op was called: SM 6.0
 * reaches a ByteAddressBuffer through bufferLoad, the same intrinsic a
 * structured buffer uses.
 */
enum d12_resource_kind {
   D12_RES_KIND_TEXTURE_1D = 1,
   D12_RES_KIND_TEXTURE_2D = 2,
   D12_RES_KIND_TEXTURE_1D_ARRAY = 6,
   D12_RES_KIND_TEXTURE_2D_ARRAY = 7,
   D12_RES_KIND_TYPED_BUFFER = 10,
   D12_RES_KIND_RAW_BUFFER = 11,
   D12_RES_KIND_STRUCTURED_BUFFER = 12,
};

/* Native descriptor shape required by a shader declaration. D3D12 permits
 * one API texture view to be consumed through array and non-array resource
 * declarations in distinct register spaces. The API descriptor therefore
 * cannot, by itself, select the AGX descriptor dimension. */
enum d12_texture_shape {
   D12_TEXTURE_SHAPE_NONE = 0,
   D12_TEXTURE_SHAPE_1D,
   D12_TEXTURE_SHAPE_1D_ARRAY,
   D12_TEXTURE_SHAPE_2D,
   D12_TEXTURE_SHAPE_2D_ARRAY,
};

static inline enum d12_texture_shape
d12_texture_shape_from_kind(unsigned kind)
{
   switch (kind) {
   case D12_RES_KIND_TEXTURE_1D:
      return D12_TEXTURE_SHAPE_1D;
   case D12_RES_KIND_TEXTURE_1D_ARRAY:
      return D12_TEXTURE_SHAPE_1D_ARRAY;
   case D12_RES_KIND_TEXTURE_2D:
      return D12_TEXTURE_SHAPE_2D;
   case D12_RES_KIND_TEXTURE_2D_ARRAY:
      return D12_TEXTURE_SHAPE_2D_ARRAY;
   default:
      return D12_TEXTURE_SHAPE_NONE;
   }
}

/*
 * An unbounded resource array -- `Texture2D t[] : register(t0)`, the basis
 * of bindless -- declares a range size of UINT32_MAX. Anything that
 * enumerates a range has to check for it: a loop to range_size runs four
 * billion times, which is not a hang anyone would look for in a binding
 * table.
 */
#define D12_RANGE_UNBOUNDED UINT32_MAX

struct d12_resource_binding {
   uint32_t id;    /* rangeId: index within its class's list */
   uint32_t klass; /* enum d12_resource_class */
   uint32_t kind;  /* enum d12_resource_kind (shape) */
   uint32_t space;
   uint32_t lower_bound; /* first shader register in the range */
   uint32_t range_size;  /* D12_RANGE_UNBOUNDED for a bindless array */

   /* Structured-buffer element stride, from metadata tag 1. Zero for raw
    * and typed buffers, whose accesses are byte-addressed.
    */
   uint32_t stride_B;

   uint32_t cb_size_B; /* CBV only */

   /* Source-level name from the resource metadata. Library targets leave
    * the global-variable operand undef, so this is the only link back to
    * the global that stands for the resource.
    */
   const char *name;

   /* The global variable that stands for this resource. SM 6.x declares
    * resources as globals, loads them, and converts the loaded value with
    * createHandleForLib -- so a load has to be traceable back to here.
    */
   uint32_t global_value; /* ~0u when unresolved */
};

/* One never-moved block of the module arena. */
struct d12_arena_chunk {
   struct d12_arena_chunk *next;
   char data[];
};

struct d12_module {
   /*
    * The element tables are grown by doubling as records arrive, up to
    * the D12_LIMIT_* ceilings. They MOVE when they grow: nothing may hold
    * a pointer to an element across a push into the same table. Consumers
    * index by number, which is what the bitcode does anyway.
    */
   struct d12_type *types;
   unsigned num_types, type_capacity;

   struct d12_value *values;
   unsigned num_values, value_capacity;

   struct d12_function *functions;
   unsigned num_functions, function_capacity;

   struct d12_metadata *metadata;
   unsigned num_metadata, metadata_capacity;

   struct d12_named_md named_md[D12_MAX_NAMED_MD];
   unsigned num_named_md;

   /*
    * Arena for names and the small per-element arrays parsed out of the
    * bitstream: struct members, function parameters, metadata node
    * operands, instruction operands and phi sources.
    *
    * CHUNKED, and it has to be: everything handed out is a bare pointer
    * into this storage, held by types, values and functions for the life
    * of the module. A single realloc'd buffer moves, and every one of
    * those pointers silently starts reading whatever now occupies the old
    * allocation -- a use-after-free that produces plausible-looking
    * garbage rather than a crash. Chunks are never moved or resized.
    */
   struct d12_arena_chunk *arena;
   size_t chunk_used, chunk_size;

   /* Entry-point signatures, resolved from dx.entryPoints */
   struct d12_sig_element input_sig[D12_MAX_SIG_ELEMENTS];
   unsigned num_input_sig;
   struct d12_sig_element output_sig[D12_MAX_SIG_ELEMENTS];
   unsigned num_output_sig;
   /* Patch-constant signature: hull outputs / domain inputs. */
   struct d12_sig_element patch_sig[D12_MAX_SIG_ELEMENTS];
   unsigned num_patch_sig;

   /* [numthreads(x,y,z)] from the entry-point properties (compute only) */
   uint32_t num_threads[3];

   /*
    * Tessellation and geometry entry-point properties. DXIL keeps these
    * in the same (tag, value) list as numthreads; NIR needs them in
    * shader_info before any lowering runs, because they decide the
    * shape of the emulation.
    */
   uint32_t tess_domain;        /* 1 = isoline, 2 = tri, 3 = quad */
   uint32_t tess_partitioning;  /* DXIL TessellatorPartitioning */
   uint32_t tess_output_prim;   /* DXIL TessellatorOutputPrimitive */
   uint32_t tess_control_points;/* output control points (hull) */

   /* HULL SHADERS HAVE TWO FUNCTION BODIES. The entry runs once per
    * output control point; a separate PATCH CONSTANT function runs once
    * per patch and is what writes SV_TessFactor. Named by HS_STATE
    * operand 0. ~0u when absent.
    *
    * Nothing tessellates without it: the factors stay zero, the
    * tessellator counts zero vertices, and the draw is empty.
    */
   /* Mesh / amplification entry properties (MSState, ASState). The
    * thread count lives INSIDE these nodes rather than in the compute
    * NumThreads tag, so a mesh shader parsed as compute silently gets a
    * 1x1x1 workgroup and only its first thread ever runs.
    */
   uint32_t mesh_max_vertices;
   uint32_t mesh_max_primitives;
   uint32_t mesh_topology;
   uint32_t mesh_payload_size;

   uint32_t patch_constant_func;
   bool has_patch_constant; /* zero is a valid function index */
   uint32_t gs_input_prim;      /* DXIL InputPrimitive */
   uint32_t gs_output_topology; /* DXIL PrimitiveTopology */
   uint32_t gs_max_vertices;
   uint32_t gs_instance_count;

   /* Declared resources, from dx.resources */
   /* Resource records are heap-backed. Legal shaders can declare far more
    * than the old fixed 32-record array, and this module is stack-allocated
    * on Windows shader compilation paths. Keep the stack footprint fixed and
    * grow only to the parser's bounded metadata count. */
   struct d12_resource_binding *resources;
   unsigned num_resources;
   unsigned resource_capacity;

   const char *error;
};

/* Look up a signature element by the id loadInput/storeOutput carries. */
const struct d12_sig_element *d12_dxil_find_sig(const struct d12_sig_element *sig,
                                                unsigned count, uint32_t id);

/*
 * Parse a complete module. Returns false with mod->error set on failure.
 * The module owns heap allocations; free with d12_dxil_module_finish.
 */
struct d12_module *d12_dxil_module_alloc(void);
void d12_dxil_module_free(struct d12_module *mod);
void *d12_dxil_module_arena_alloc(struct d12_module *mod, size_t size);
bool d12_dxil_module_parse(struct d12_module *mod, const uint8_t *bitcode,
                           size_t size_B);
void d12_dxil_module_finish(struct d12_module *mod);

/* Convenience: find a named metadata node list (e.g. "dx.entryPoints") */
const struct d12_named_md *d12_dxil_find_named_md(const struct d12_module *mod,
                                                  const char *name);

/*
 * ---------------------------------------------------------------------
 * Translation to NIR (G4c) — the "no SPIR-V" step
 * ---------------------------------------------------------------------
 */

struct nir_shader;

/*
 * Resource binding ABI (campaign OQ-2).
 *
 * The driver publishes ONE 64-bit pointer — the root table — in AGX
 * uniform registers (four 16-bit registers). The root table is an array
 * of 64-bit resource base addresses indexed by binding slot, so a shader
 * reaches a resource with one indirection:
 *
 *     base = load_global(root + binding * 8)
 *
 * A single uniform slot keeps the ABI stable no matter how many resources
 * are bound. The shader reads the root with load_root_agx, a placeholder
 * that survives agx_preprocess_nir; d12_dxil_lower_root() turns it into a
 * load_preamble afterwards. Emitting load_preamble before preprocessing
 * does NOT work — the compiler reassigns its base.
 */
#define D12_UNIFORM_ROOT            4
#define D12_UNIFORM_ROOT_REGS       4

/* Fragment uniforms 0..11 are reserved by the hardware ABI for the EMRT
 * heap and blend constants. Honeykrisp consequently publishes its fragment
 * root at 12; using the generic slot 4 there reads as zero on hardware.
 */
#define D12_UNIFORM_FRAGMENT_ROOT   12

/*
 * Texture descriptors live in their own heap: AGX reaches a texture
 * through a hardware descriptor, not a bare address. A bindless handle is
 * a byte OFFSET into that heap, and the heap's base sits in these uniform
 * registers -- the instruction takes a 64-bit uniform base plus a 32-bit
 * offset register and adds them.
 */
#define D12_UNIFORM_TEXTURES        16
#define D12_UNIFORM_TEXTURE_REGS    4

/*
 * Uniform registers the driver claims for its own ABI. This must cover
 * EVERY register the driver publishes: the compiler starts push_count
 * here and allocates its own promoted constants above it, so anything
 * written past this point is either never loaded by the hardware or
 * overwritten. Getting it wrong presents as half of a 64-bit pointer
 * arriving -- a fault at address 0.
 */
/*
 * SM 6.6 ResourceDescriptorHeap base: the shader-visible heap's
 * d12_descriptor array, indexed directly by heap index. Its 40-byte stride
 * is sizeof(struct d12_descriptor), static-asserted where the struct is
 * visible.
 */
#define D12_UNIFORM_HEAP            56
#define D12_UNIFORM_HEAP_REGS       4
#define D12_HEAP_DESC_SIZE          40
#define D12_DESC_COUNTER_OFFSET     8
#define D12_DESC_BUFFER_COUNT_OFFSET  24
#define D12_DESC_BUFFER_STRIDE_OFFSET 28
#define D12_DESC_TEXTURE_METADATA0_OFFSET 24
#define D12_DESC_TEXTURE_METADATA1_OFFSET 28
#define D12_SPARSE_BUFFER_TILE_SIZE 65536u

/* Writable image operations do not consume AGX TEXTURE descriptors.  Stores
 * and atomics take a PBE descriptor with a different binary layout, so they
 * have a parallel same-slot heap and a distinct uniform base. */
#define D12_UNIFORM_PBES            60
#define D12_UNIFORM_PBE_REGS        4

/* Parallel AGX-native arrays for shader-visible D3D12 heaps.
 * D12_UNIFORM_HEAP remains the 40-byte metadata heap. Dynamic texture and
 * image handles convert that heap index to the 24-byte hardware stride and
 * use independent bases; bindless samplers use their compact 8-byte heap. */
#define D12_UNIFORM_HEAP_TEXTURES   64
#define D12_UNIFORM_HEAP_TEXTURE_REGS 4
#define D12_UNIFORM_HEAP_PBES       68
#define D12_UNIFORM_HEAP_PBE_REGS   4
#define D12_UNIFORM_HEAP_SAMPLERS   72
#define D12_UNIFORM_HEAP_SAMPLER_REGS 4

/* Per-stage custom-border side table. Bound samplers are still ordinary AGX
 * sampler states; this table supplies the arbitrary colour that the hardware
 * sampler descriptor cannot encode. */
#define D12_UNIFORM_CUSTOM_BORDERS  76
#define D12_UNIFORM_CUSTOM_BORDER_REGS 4

/*
 * The root binding upload contains resource, counter, BUFINFO and sparse
 * buffer residency arrays back to back plus an optional sampler-index plane.
 * The metadata header also publishes the coherent bindless residency array.
 * Loading pointers first avoids AGX's global-load immediate limit.
 */
#define D12_ROOT_RESOURCE_BASE      0
#define D12_ROOT_COUNTER_BASE       D12_MAX_RESOURCE_BINDINGS
#define D12_ROOT_BUFINFO_BASE       (2 * D12_MAX_RESOURCE_BINDINGS)
#define D12_ROOT_RESIDENCY_BASE     (3 * D12_MAX_RESOURCE_BINDINGS)
#define D12_ROOT_SAMPLER_BASE       (4 * D12_MAX_RESOURCE_BINDINGS)
#define D12_ROOT_TABLE_ENTRIES      (4 * D12_MAX_RESOURCE_BINDINGS)

#define D12_ROOT_HEADER_SIZE          32
#define D12_ROOT_METADATA_HEADER_SIZE 56

/*
 * SM 6.8 extended command info: StartVertexLocation then
 * StartInstanceLocation, two u32 published per draw. Unlike the tables
 * above these registers hold the VALUES, not an address.
 */
#define D12_UNIFORM_DRAW            20
#define D12_UNIFORM_DRAW_REGS       4

/*
 * Parameter buffers for the geometry/tessellation compute emulation.
 * src/poly's lowering emits loads of these and leaves them "for the
 * driver"; Honeykrisp answers them from its root descriptor, we answer
 * them from reserved uniforms like everything else. Each is a 64-bit
 * address, so four 16-bit registers apiece.
 */
#define D12_UNIFORM_GEOM_PARAMS     24
#define D12_UNIFORM_TESS_PARAMS     28
#define D12_UNIFORM_VS_OUTPUTS      32
#define D12_UNIFORM_VERTEX_PARAMS   36
#define D12_UNIFORM_PARAM_REGS      4

/*
 * Mesh and amplification parameter buffer. AGX has no mesh hardware and
 * Honeykrisp exposes none, so unlike geometry and tessellation there is
 * no in-tree emulation to borrow: a mesh shader is translated straight
 * into a compute shader that writes its outputs to driver buffers. This
 * uniform points at a struct of addresses describing where:
 *
 *    +0   u64  counts      { u32 vertices, u32 primitives }
 *    +8   u64  vertices    per-vertex outputs, D12_MESH_SLOT_STRIDE each
 *    +16  u64  primitives  per-primitive outputs, same stride
 *    +24  u64  indices     3 x u32 per primitive
 *    +32  u64  payload     amplification payload (0 when there is none)
 *    +40  u64  dispatch    { u32 x, y, z } written by dispatchMesh
 *    +48  u64  draw        the indirect draw descriptor ms_setup fills
 *    +56  u64  grid        6-word indirect-local grid for the mesh
 *                          dispatch, filled by as_setup
 *
 * The last two are driver-side only -- the translator never reads them.
 * They are separate slots because an AMPLIFICATION shader writes the
 * dispatch grid at +40 itself, so the draw descriptor cannot share it.
 */
#define D12_UNIFORM_MESH_PARAMS     40
#define D12_UNIFORM_MESH_REGS       4

#define D12_MESH_OFF_COUNTS         0
#define D12_MESH_OFF_VERTICES       8
#define D12_MESH_OFF_PRIMITIVES     16
#define D12_MESH_OFF_INDICES        24
#define D12_MESH_OFF_PAYLOAD        32
#define D12_MESH_OFF_DISPATCH       40
#define D12_MESH_OFF_DRAW           48
#define D12_MESH_OFF_GRID           56

/* One 16-byte vec4 per signature slot, indexed by slot then component. */
#define D12_MESH_SLOT_STRIDE        16
#define D12_MESH_MAX_SLOTS          32

/* Worst-case mesh workgroups an amplification shader may ask for. Its
 * DispatchMesh grid is GPU-written, so the output buffers must be sized
 * before it is known. Exceeding this overruns them, so the mesh shader
 * is dispatched indirectly but the ALLOCATION is bounded here.
 */
#define D12_MESH_MAX_AMP_GROUPS     32

/*
 * Work-graph node parameter block. SM 6.8 work graphs are GPU-scheduled:
 * a node reads input records, allocates output records for successor
 * nodes, and the scheduler runs the graph. There is no scheduler here,
 * so this describes only the DATA the shader touches; dispatching the
 * graph is runtime that does not exist yet.
 *
 *    +0   u64  input records   (this node's input record array)
 *    +8   u64  output records  (successor record arena)
 *    +16  u64  counts          { u32 input count, u32 output cursor }
 *    +24  u32  record stride   bytes per record
 */
#define D12_UNIFORM_NODE_PARAMS     44
#define D12_UNIFORM_NODE_REGS       4

/* Pipeline-statistics counters the geometry emulation increments, and
 * the input topology its count pass reads.
 */
#define D12_UNIFORM_STAT_QUERY      48
#define D12_UNIFORM_STAT_REGS       4

#define D12_NODE_OFF_INPUT          0
#define D12_NODE_OFF_OUTPUT         8
#define D12_NODE_OFF_COUNTS         16
#define D12_NODE_OFF_STRIDE         24

/*
 * Vertex attribute fetch. Points at a struct d12_vb_params, which the
 * draw path fills from the bound D3D12_VERTEX_BUFFER_VIEWs and the PSO's
 * input layout.
 *
 * This exists because AGX's own vertex-fetch ABI cannot be used here.
 * agx_nir_vs_prolog answers load_vbo_base_agx from AGX_ABI_VUNI_VBO_BASE,
 * which is uniform register 4*n starting at ZERO and growing with the
 * attribute count -- it would run straight through D12_UNIFORM_ROOT at 4
 * and D12_UNIFORM_TEXTURES at 16. Honeykrisp has the same problem and
 * solves it the same way, answering the three intrinsics from its root
 * table instead (hk_nir_lower_descriptors.c).
 */
#define D12_UNIFORM_VB_PARAMS       52
#define D12_UNIFORM_VB_REGS         4

#define D12_UNIFORM_RESERVED                                                 \
   (D12_UNIFORM_CUSTOM_BORDERS + D12_UNIFORM_CUSTOM_BORDER_REGS)

/*
 * The vertex attribute parameter block, indexed by DXIL input signature
 * element id -- which is what agx_nir_lower_vbo uses as its buffer handle,
 * and what our loadInput lowering already puts in nir_intrinsic_base.
 *
 *   base      per-ATTRIBUTE address, with AlignedByteOffset folded in;
 *             AGX_ZERO_PAGE_ADDRESS when the attribute has no buffer
 *   clamp     highest in-bounds element index (agx_calculate_vbo_clamp)
 *   stride_el stride in INTERCHANGE ELEMENTS, not bytes: the shader
 *             multiplies it by the element index directly
 */
#define D12_MAX_VB_ATTRIBS          16
/* D3D12 exposes 32 independently bindable vertex-buffer slots. AGX still
 * consumes at most 16 live attributes, but those attributes may reference
 * any of the API slots, so the command-list binding state must not collapse
 * the two distinct limits. */
#define D12_MAX_VB_SLOTS            32
#define D12_VB_OFF_BASE             0
#define D12_VB_OFF_CLAMP            (8 * D12_MAX_VB_ATTRIBS)
#define D12_VB_OFF_STRIDE                                                    \
   (D12_VB_OFF_CLAMP + 4 * D12_MAX_VB_ATTRIBS)
#define D12_VB_PARAMS_SIZE                                                   \
   (D12_VB_OFF_STRIDE + 4 * D12_MAX_VB_ATTRIBS)

struct d12_vb_params {
   uint64_t base[D12_MAX_VB_ATTRIBS];
   uint32_t clamp[D12_MAX_VB_ATTRIBS];
   uint32_t stride_el[D12_MAX_VB_ATTRIBS];
};

static_assert(sizeof(struct d12_vb_params) == D12_VB_PARAMS_SIZE,
              "vertex attribute block must match the offsets the shader bakes");
static_assert(offsetof(struct d12_vb_params, clamp) == D12_VB_OFF_CLAMP,
              "clamp offset must match D12_VB_OFF_CLAMP");
static_assert(offsetof(struct d12_vb_params, stride_el) == D12_VB_OFF_STRIDE,
              "stride offset must match D12_VB_OFF_STRIDE");

#define D12_SAMPLER_DESC_SIZE       8

/*
 * Root-table layout.
 *
 * A resource is named by three things -- class, register space and shader
 * register -- and all three matter: t0, u0 and b0 are different resources,
 * and so are t0 space0 and t0 space1. A formula over those would need a
 * slot for every combination that could exist, most of them empty, and
 * uploaded on every draw.
 *
 * Instead the slots are assigned DENSELY, from the root signature, which
 * enumerates exactly the bindings a pipeline can have. Both sides read the
 * same assignment: the shader is compiled against it, and the command list
 * fills the table through it. Spaces cost nothing because an unused one
 * takes no slot.
 */
#define D12_MAX_RESOURCE_BINDINGS 256

struct d12_binding_map {
   struct {
      uint32_t klass;
      uint32_t space;
      uint32_t reg;
      /* Resource descriptors and sampler states are different hardware
       * files. Keep a dense slot in the appropriate file instead of making
       * unrelated t/u/b bindings consume scarce sampler indices. */
      uint32_t slot;
      /* One descriptor range/root descriptor in the API root signature.
       * Adjacent shader registers are not necessarily one range: a sized
       * shader array may not spill from one root range into the next.
       */
      uint32_t root_range;
      /* ALL or the one graphics stage allowed to see this root-signature
       * record. Disjoint stages may legally reuse the same class/space/reg
       * and therefore need distinct runtime slots. */
      uint32_t visibility;
      /* This entry names the lower bound of a rooted descriptor range.
       * The slot contains a GPU pointer to descriptor-array element zero;
       * shader indexing happens in that array rather than across root slots.
       */
      bool unbounded;
      /* Descriptor count of a rooted range, or 0 when the API range really
       * is unbounded.  A range the root signature sized explicitly is rooted
       * for slot economy, not because it extends forever: without its extent
       * a lookup for a register PAST its end resolves to this base and reads
       * the wrong descriptor out of the heap.
       */
      uint32_t range_count;
   } entries[D12_MAX_RESOURCE_BINDINGS];
   unsigned count;
   unsigned resource_count;
   unsigned sampler_count;
   /* Set when a binding could not be assigned a dense slot.  The map is then
    * incomplete and must not be used to compile or bind a pipeline. */
   bool overflow;
   /* UINT32_MAX for bind-time/root-range lookup, otherwise the stage whose
    * shader is being translated. */
   uint32_t visibility_filter;
};

/* Entry for a binding, or ~0u when the map does not name it. */
static inline unsigned
d12_binding_map_find_entry(const struct d12_binding_map *map, unsigned klass,
                           unsigned space, unsigned reg)
{
   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].klass == klass && map->entries[i].space == space &&
          map->entries[i].reg == reg &&
          (map->visibility_filter == UINT32_MAX ||
           map->entries[i].visibility == 0 ||
           map->entries[i].visibility == map->visibility_filter))
         return i;
   }
   return ~0u;
}

/* Bind-time lookup is keyed by the root-signature record, not merely by
 * register. That distinguishes legal VS/PS records which both name b0. */
static inline unsigned
d12_binding_map_find_root_range(const struct d12_binding_map *map,
                                unsigned root_range, unsigned reg)
{
   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].root_range == root_range &&
          map->entries[i].reg == reg)
         return map->entries[i].slot;
   }
   return ~0u;
}

static inline bool
d12_binding_map_root_range_unbounded(const struct d12_binding_map *map,
                                     unsigned root_range, unsigned reg)
{
   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].root_range == root_range &&
          map->entries[i].reg == reg)
         return map->entries[i].unbounded;
   }
   return false;
}

/* Slot in the resource root table or sampler-state file. */
static inline unsigned
d12_binding_map_find(const struct d12_binding_map *map, unsigned klass,
                     unsigned space, unsigned reg)
{
   unsigned entry = d12_binding_map_find_entry(map, klass, space, reg);
   return entry == ~0u ? ~0u : map->entries[entry].slot;
}

static inline unsigned
d12_binding_map_find_visible(const struct d12_binding_map *map,
                             unsigned klass, unsigned space, unsigned reg,
                             uint32_t visibility)
{
   struct d12_binding_map filtered = *map;
   filtered.visibility_filter = visibility;
   return d12_binding_map_find(&filtered, klass, space, reg);
}

static inline unsigned
d12_binding_map_root_range(const struct d12_binding_map *map, unsigned klass,
                           unsigned space, unsigned reg)
{
   unsigned entry = d12_binding_map_find_entry(map, klass, space, reg);
   return entry == ~0u ? ~0u : map->entries[entry].root_range;
}

static inline bool
d12_binding_map_unbounded(const struct d12_binding_map *map, unsigned klass,
                          unsigned space, unsigned reg)
{
   unsigned entry = d12_binding_map_find_entry(map, klass, space, reg);
   return entry != ~0u && map->entries[entry].unbounded;
}

/* Append a binding with its root-signature range identity. */
static inline unsigned
d12_binding_map_add_range_visible(struct d12_binding_map *map, unsigned klass,
                                  unsigned space, unsigned reg,
                                  unsigned root_range,
                                  uint32_t visibility)
{
   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].klass == klass && map->entries[i].space == space &&
          map->entries[i].reg == reg &&
          map->entries[i].visibility == visibility) {
         if (map->entries[i].root_range != root_range)
            map->entries[i].root_range = ~0u;
         return map->entries[i].slot;
      }
   }

   if (map->count >= D12_MAX_RESOURCE_BINDINGS) {
      /* Dropping the binding silently makes the shader fail later with
       * "the root signature does not bind this resource", which names the
       * wrong party.  Record the real cause so the caller can report it. */
      map->overflow = true;
      return ~0u;
   }

   unsigned entry = map->count++;
   unsigned slot = klass == D12_RES_SAMPLER ? map->sampler_count++
                                             : map->resource_count++;
   map->entries[entry].klass = klass;
   map->entries[entry].space = space;
   map->entries[entry].reg = reg;
   map->entries[entry].slot = slot;
   map->entries[entry].root_range = root_range;
   map->entries[entry].visibility = visibility;
   map->entries[entry].unbounded = false;
   map->entries[entry].range_count = 0;
   return slot;
}

static inline unsigned
d12_binding_map_add_range(struct d12_binding_map *map, unsigned klass,
                          unsigned space, unsigned reg,
                          unsigned root_range)
{
   return d12_binding_map_add_range_visible(
      map, klass, space, reg, root_range, 0);
}

static inline unsigned
d12_binding_map_add_unbounded(struct d12_binding_map *map, unsigned klass,
                              unsigned space, unsigned reg,
                              unsigned root_range)
{
   unsigned slot = d12_binding_map_add_range(map, klass, space, reg,
                                             root_range);
   unsigned entry = d12_binding_map_find_entry(map, klass, space, reg);
   if (entry != ~0u) {
      map->entries[entry].unbounded = true;
      map->entries[entry].range_count = 0;
   }
   return slot;
}

/* range_count is the API range's descriptor count, or 0 when the range is
 * genuinely unbounded. */
static inline unsigned
d12_binding_map_add_rooted_visible(
   struct d12_binding_map *map, unsigned klass, unsigned space, unsigned reg,
   unsigned root_range, uint32_t visibility, uint32_t range_count)
{
   unsigned slot = d12_binding_map_add_range_visible(
      map, klass, space, reg, root_range, visibility);
   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].root_range == root_range &&
          map->entries[i].reg == reg) {
         map->entries[i].unbounded = true;
         map->entries[i].range_count = range_count;
         break;
      }
   }
   return slot;
}

static inline unsigned
d12_binding_map_add_unbounded_visible(
   struct d12_binding_map *map, unsigned klass, unsigned space, unsigned reg,
   unsigned root_range, uint32_t visibility)
{
   return d12_binding_map_add_rooted_visible(map, klass, space, reg,
                                             root_range, visibility, 0);
}

/*
 * Resolve a shader register to the rooted descriptor-array base that
 * actually contains it.
 *
 * An API range has one map entry at its lower bound, while the shader may
 * split that same heap into several declared ranges beginning at t4, t32 and
 * so on, so exact register lookup cannot answer whether a register lives in
 * a rooted array.  Take the nearest visible rooted base at or below the
 * register -- but only if the register is genuinely inside it.  A range the
 * root signature sized explicitly ends, and a register past its end belongs
 * to some other root parameter; resolving it here would silently read the
 * wrong descriptor out of the heap.
 */
static inline bool
d12_binding_map_rooted_base(const struct d12_binding_map *map, unsigned klass,
                            unsigned space, unsigned reg, unsigned *out_slot,
                            unsigned *out_reg)
{
   bool found = false;
   unsigned best_reg = 0;

   for (unsigned i = 0; i < map->count; i++) {
      if (map->entries[i].klass != klass ||
          map->entries[i].space != space || !map->entries[i].unbounded ||
          map->entries[i].reg > reg ||
          (map->visibility_filter != UINT32_MAX &&
           map->entries[i].visibility != 0 &&
           map->entries[i].visibility != map->visibility_filter))
         continue;

      /* A sized range covers [reg, reg + range_count). Zero means the API
       * range is unbounded and every higher register is inside it. */
      if (map->entries[i].range_count &&
          reg - map->entries[i].reg >= map->entries[i].range_count)
         continue;

      if (!found || map->entries[i].reg > best_reg) {
         found = true;
         best_reg = map->entries[i].reg;
         *out_slot = map->entries[i].slot;
         *out_reg = map->entries[i].reg;
      }
   }

   return found;
}

/* Shader-derived maps have no API range topology. */
static inline unsigned
d12_binding_map_add(struct d12_binding_map *map, unsigned klass,
                    unsigned space, unsigned reg)
{
   return d12_binding_map_add_range(map, klass, space, reg, ~0u);
}

/*
 * Build a map from a module's own declarations. Used when there is no root
 * signature to derive one from -- the standalone compiler tool -- and never
 * by the driver, where compile time and bind time have to agree on an
 * assignment neither derives from the other.
 */
void d12_binding_map_from_module(const struct d12_module *mod,
                                 struct d12_binding_map *out);
bool d12_dxil_collect_texture_shapes(const struct d12_module *mod,
                                     const struct d12_binding_map *map,
                                     uint8_t *texture_shapes);

/* How many registers a declared range actually covers, clamped so an
 * unbounded array cannot be walked. Returns 0 for unbounded.
 */
static inline unsigned
d12_range_count(const struct d12_resource_binding *rb)
{
   if (rb->range_size == D12_RANGE_UNBOUNDED)
      return 0;
   return MAX2(rb->range_size, 1u);
}

/*
 * Translate a parsed module's entry point to NIR for the AGX backend.
 * Returns NULL with *error set on failure. Caller owns the shader.
 */
struct nir_shader *d12_dxil_to_nir(const struct d12_module *mod,
                                   mesa_shader_stage stage,
                                   const struct d12_binding_map *map,
                                   uint32_t descriptor_generation,
                                   const char **error);

/*
 * Replace the root-table placeholder with a real uniform read. MUST run
 * after agx_preprocess_nir and before agx_compile_shader_nir.
 */
bool d12_dxil_lower_root(struct nir_shader *nir);
bool d12_dxil_lower_root_mask(struct nir_shader *nir, uint16_t sample_mask);
bool d12_dxil_lower_sample_positions(struct nir_shader *nir,
                                     uint32_t packed_positions);

/* Vertex-shader sysvals for the SOFTWARE vertex shader -- the compute
 * program that runs when a geometry or tessellation shader follows it.
 * Shared so the standalone tool lowers exactly what the driver lowers.
 */
bool d12_dxil_lower_sw_vs(struct nir_shader *nir);
bool d12_dxil_mark_sw_vs_vertex_id(struct nir_shader *nir);
bool d12_dxil_lower_sw_vs_vertex_id(struct nir_shader *nir);
/*
 * Answer agx_nir_lower_vbo's three intrinsics from the driver's vertex
 * attribute parameter block. Run AFTER agx_nir_lower_vbo.
 */
bool d12_dxil_lower_vbo_params(struct nir_shader *nir);

/* DXBC (Shader Model 4/5) front end -- see d12_dxbc.c. */
bool d12_dxbc_find_shader(const void *data, size_t size_B,
                          const uint8_t **out, size_t *out_size_B,
                          uint32_t *out_version);
bool d12_dxbc_is_dxbc_shader(const void *data, size_t size_B);
bool d12_dxbc_container_stage(const void *data, size_t size_B,
                              mesa_shader_stage *stage);
struct nir_shader *d12_dxbc_to_nir(const void *data, size_t size_B,
                                   mesa_shader_stage stage,
                                   const struct d12_binding_map *map,
                                   uint32_t descriptor_generation,
                                   uint8_t *texture_shapes, const char **error);

/*
 * Read a shader's INPUT signature out of a DXIL container, without
 * translating it. The pipeline needs the semantic names and indices to
 * resolve a D3D12_INPUT_LAYOUT_DESC against the shader's own inputs, and
 * that happens before -- and independently of -- compilation.
 *
 * The name is COPIED: struct d12_sig_element points into the module's
 * string arena, which d12_dxil_module_finish frees.
 *
 * Returns the number of elements written, up to max.
 */
struct d12_input_sig_elem {
   char name[32];
   uint32_t semantic_index;
   uint32_t id;
   uint32_t register_index;
   uint32_t start_component;
   uint32_t component_count;
   uint32_t rows;
   uint32_t component_type;
   uint32_t min_precision;
   uint32_t system_value;
};

unsigned d12_dxbc_input_signature(const void *data, size_t size_B,
                                  struct d12_input_sig_elem *out,
                                  unsigned max);
unsigned d12_dxil_input_signature(const void *dxil, size_t size_B,
                                  struct d12_input_sig_elem *out,
                                  unsigned max);
unsigned d12_dxbc_output_signature(const void *data, size_t size_B,
                                   struct d12_input_sig_elem *out,
                                   unsigned max);
unsigned d12_dxil_output_signature(const void *dxil, size_t size_B,
                                   struct d12_input_sig_elem *out,
                                   unsigned max);
unsigned d12_dxbc_patch_signature(const void *data, size_t size_B,
                                  struct d12_input_sig_elem *out,
                                  unsigned max);
unsigned d12_dxil_patch_signature(const void *dxil, size_t size_B,
                                  struct d12_input_sig_elem *out,
                                  unsigned max);

bool d12_dxil_lower_uvs_index(struct nir_shader *nir,
                              const unsigned *slots);
