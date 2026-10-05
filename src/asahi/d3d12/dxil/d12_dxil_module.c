/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DXIL module decoder: turns the bitstream record tree into type,
 * constant, function and metadata tables — everything the NIR translator
 * (G4c) needs, with no SPIR-V and no external compiler in the path.
 *
 * Record numbering follows LLVM 3.7 (what DXIL froze on), cross-checked
 * against Mesa's own MIT-licensed DXIL writer in src/microsoft/compiler.
 */

#include "d12_debug.h"
#include "d12_dxil.h"

#include <stdlib.h>
#include <string.h>

#define MESA_LOG_TAG "d3d12agx"
#include "util/log.h"
#include "util/macros.h"

static void parse_entry_signatures(struct d12_module *mod);
static void parse_entry_properties(struct d12_module *mod);
static void parse_resources(struct d12_module *mod);

struct parse_ctx {
   struct d12_module *mod;

   /* Type table state */
   uint32_t pending_struct_name; /* index into strings, ~0 if none */

   /* Constant block state: CST_SETTYPE latches the type for what follows */
   uint32_t const_type;

   /* Function parsing state */
   struct d12_function *cur_func;
   uint32_t cur_block; /* basic block the next instruction lands in */

   /* Number of MODULE-level values, latched before the first function
    * body. LLVM discards a function's local values when the function
    * ends, so every function body numbers its locals starting from this
    * same base -- but we keep them all in one global array, so relative
    * operands have to be decoded against the logical number and then
    * mapped back to the global slot. Getting this wrong is invisible in
    * a module with ONE body and off-by-the-previous-function's-locals in
    * every one after it, which is every hull shader.
    */
   uint32_t module_values;
   bool module_values_latched;
   unsigned next_func_to_define; /* module order of the next FUNCTION block */

   /* Metadata naming: MD_NAME is followed by MD_NAMED_NODE */
   const char *pending_md_name;
};

static void
fail(struct d12_module *mod, const char *msg)
{
   if (!mod->error)
      mod->error = msg;
}

/* Copy a record's char-array operands into the module string arena. */
static const char *
intern_chars(struct d12_module *mod, const uint64_t *ops, unsigned n)
{
   /* Start a new chunk when this string does not fit. Chunks are never
    * grown or moved: the pointers handed out here outlive this call and
    * are held for the life of the module.
    */
   if (!mod->strings || mod->chunk_used + n + 1 > mod->chunk_size) {
      size_t want = MAX2(4096, n + 1);
      struct d12_str_chunk *chunk = malloc(sizeof(*chunk) + want);
      if (!chunk) {
         fail(mod, "out of memory interning a string");
         return NULL;
      }

      chunk->next = mod->strings;
      mod->strings = chunk;
      mod->chunk_size = want;
      mod->chunk_used = 0;
   }

   char *out = mod->strings->data + mod->chunk_used;
   for (unsigned i = 0; i < n; i++)
      out[i] = (char)ops[i];
   out[n] = '\0';
   mod->chunk_used += n + 1;
   return out;
}

static struct d12_value *
push_value(struct d12_module *mod, enum d12_value_kind kind, uint32_t type)
{
   if (mod->num_values >= D12_MAX_VALUES) {
      fail(mod, "too many values in module");
      return NULL;
   }

   struct d12_value *v = &mod->values[mod->num_values++];
   memset(v, 0, sizeof(*v));
   v->kind = kind;
   v->type = type;
   return v;
}

/*
 * Types
 */
static bool
parse_type_record(struct parse_ctx *ctx, const struct d12_record *rec)
{
   struct d12_module *mod = ctx->mod;

   if (rec->code == D12_TYPE_NUMENTRY)
      return true; /* capacity hint only */

   if (rec->code == D12_TYPE_STRUCT_NAME) {
      const char *name = intern_chars(mod, rec->ops, rec->num_ops);
      ctx->pending_struct_name = 0;
      /* Stash for the STRUCT_NAMED that follows */
      if (mod->num_types < D12_MAX_TYPES)
         mod->types[mod->num_types].name = name;
      return true;
   }

   if (mod->num_types >= D12_MAX_TYPES) {
      fail(mod, "too many types");
      return false;
   }

   struct d12_type *t = &mod->types[mod->num_types];
   const char *carried_name = t->name; /* set by a preceding STRUCT_NAME */
   memset(t, 0, sizeof(*t));
   t->kind = rec->code;

   switch (rec->code) {
   case D12_TYPE_VOID:
   case D12_TYPE_FLOAT:
   case D12_TYPE_DOUBLE:
   case D12_TYPE_HALF:
   case D12_TYPE_LABEL:
   case D12_TYPE_METADATA:
   case D12_TYPE_OPAQUE:
      break;

   case D12_TYPE_INTEGER:
      if (rec->num_ops < 1)
         return true;
      t->int_bits = (unsigned)rec->ops[0];
      break;

   case D12_TYPE_POINTER:
      if (rec->num_ops < 1)
         return true;
      t->ptr.pointee = (uint32_t)rec->ops[0];
      t->ptr.addr_space = rec->num_ops > 1 ? (uint32_t)rec->ops[1] : 0;
      break;

   case D12_TYPE_ARRAY:
   case D12_TYPE_VECTOR:
      if (rec->num_ops < 2)
         return true;
      t->seq.count = rec->ops[0];
      t->seq.elem = (uint32_t)rec->ops[1];
      break;

   case D12_TYPE_FUNCTION:
      /* [vararg, retty, paramty...] */
      if (rec->num_ops < 2)
         return true;
      t->func.ret = (uint32_t)rec->ops[1];
      for (unsigned i = 2; i < rec->num_ops && t->func.num_params < 16; i++)
         t->func.params[t->func.num_params++] = (uint32_t)rec->ops[i];
      break;

   case D12_TYPE_STRUCT_ANON:
   case D12_TYPE_STRUCT_NAMED:
      /* [ispacked, eltty...] */
      for (unsigned i = 1; i < rec->num_ops && t->strct.num_members < 16; i++)
         t->strct.members[t->strct.num_members++] = (uint32_t)rec->ops[i];
      if (rec->code == D12_TYPE_STRUCT_NAMED)
         t->name = carried_name;
      break;

   default:
      /* Unknown type kinds still occupy a slot so indices stay aligned. */
      break;
   }

   mod->num_types++;
   return true;
}

/*
 * Constants. LLVM signs integers with a rotate: bit 0 is the sign.
 */
static uint64_t
decode_signed_vbr(uint64_t v)
{
   if (v & 1)
      return v == 1 ? (uint64_t)INT64_MIN : (uint64_t)(-(int64_t)(v >> 1));
   return v >> 1;
}

static bool
parse_const_record(struct parse_ctx *ctx, const struct d12_record *rec)
{
   struct d12_module *mod = ctx->mod;

   if (rec->code == D12_CST_SETTYPE) {
      if (rec->num_ops >= 1)
         ctx->const_type = (uint32_t)rec->ops[0];
      return true;
   }

   struct d12_value *v = push_value(mod, D12_VALUE_CONSTANT, ctx->const_type);
   if (!v)
      return false;

   switch (rec->code) {
   case D12_CST_NULL:
      v->is_null = true;
      break;
   case D12_CST_UNDEF:
      v->is_undef = true;
      break;
   case D12_CST_INTEGER:
      if (rec->num_ops >= 1)
         v->const_int = decode_signed_vbr(rec->ops[0]);
      break;
   case D12_CST_FLOAT:
      if (rec->num_ops >= 1)
         v->const_float = rec->ops[0];
      break;
   case D12_CST_AGGREGATE:
      /* Element value ids. SM 6.6's createHandleFromBinding names its
       * resource with one of these (a ResBind struct), so the class and
       * register cannot be recovered without them.
       */
      for (unsigned i = 0; i < rec->num_ops && i < ARRAY_SIZE(v->agg_elems);
           i++)
         v->agg_elems[v->num_agg_elems++] = (uint32_t)rec->ops[i];
      break;

   case D12_CST_DATA:
      /* Raw element values of a constant array or vector -- how DXC
       * encodes an HLSL `static const` lookup table. The elements are
       * literals, not value references.
       */
      if (rec->num_ops) {
         v->data_elems = calloc(rec->num_ops, sizeof(*v->data_elems));
         if (!v->data_elems) {
            fail(mod, "out of memory for constant data");
            return false;
         }
         for (unsigned i = 0; i < rec->num_ops; i++)
            v->data_elems[i] = rec->ops[i];
         v->num_data_elems = rec->num_ops;
      }
      break;

   case D12_CST_CE_GEP:
   case D12_CST_CE_INBOUNDS_GEP: {
      /* [opt pointee-type, (type, value) pairs]: a pointer computed at
       * constant time -- DXC's address of a static array element. The
       * value ids are module-level and therefore absolute. The value
       * slot was already pushed above -- pushing another here shifts
       * every subsequent value id in the module.
       */
      unsigned start = (rec->num_ops & 1) ? 1 : 0;
      v->is_ce_gep = true;
      for (unsigned i = start;
           i + 1 < rec->num_ops && v->num_agg_elems < 8; i += 2)
         v->agg_elems[v->num_agg_elems++] = (uint32_t)rec->ops[i + 1];
      break;
   }
   default:
      /* Strings and const-exprs: slot reserved, contents parsed when a
       * shader is found that needs them.
       */
      break;
   }

   return true;
}

/*
 * Metadata
 */
static bool
parse_metadata_record(struct parse_ctx *ctx, const struct d12_record *rec)
{
   struct d12_module *mod = ctx->mod;

   if (rec->code == D12_MD_NAME) {
      ctx->pending_md_name = intern_chars(mod, rec->ops, rec->num_ops);
      return true;
   }

   if (rec->code == D12_MD_NAMED_NODE) {
      if (mod->num_named_md >= D12_MAX_NAMED_MD)
         return true;

      struct d12_named_md *nmd = &mod->named_md[mod->num_named_md++];
      memset(nmd, 0, sizeof(*nmd));
      nmd->name = ctx->pending_md_name;
      for (unsigned i = 0; i < rec->num_ops && nmd->num_nodes < 16; i++)
         nmd->nodes[nmd->num_nodes++] = rec->ops[i];
      ctx->pending_md_name = NULL;
      return true;
   }

   if (mod->num_metadata >= D12_MAX_METADATA) {
      fail(mod, "too many metadata nodes");
      return false;
   }

   struct d12_metadata *md = &mod->metadata[mod->num_metadata++];
   memset(md, 0, sizeof(*md));
   md->kind = rec->code;

   switch (rec->code) {
   case D12_MD_STRING:
      md->string = intern_chars(mod, rec->ops, rec->num_ops);
      break;

   case D12_MD_VALUE:
      /* [type, value] */
      if (rec->num_ops >= 2) {
         md->value_type = (uint32_t)rec->ops[0];
         md->value_id = rec->ops[1];
      }
      break;

   case D12_MD_NODE:
      /* Operands are metadata ids biased by 1; 0 means null. */
      for (unsigned i = 0; i < rec->num_ops && md->node.num_ops < 64; i++) {
         unsigned slot = md->node.num_ops++;
         if (rec->ops[i] == 0) {
            md->node.op_is_null[slot] = true;
            md->node.ops[slot] = 0;
         } else {
            md->node.ops[slot] = rec->ops[i] - 1;
         }
      }
      break;

   default:
      break;
   }

   return true;
}

/*
 * Module-level records
 */
static bool
parse_module_record(struct parse_ctx *ctx, const struct d12_record *rec)
{
   struct d12_module *mod = ctx->mod;

   switch (rec->code) {
   case D12_MODULE_GLOBALVAR: {
      /*
       * [type, flags, initid, linkage, alignment, section, ...]
       *
       * The documented layout calls operand 0 a POINTER type, and for
       * pre-3.7 bitcode it is. DXC's LLVM 3.7 does not write one: dumping
       * `groupshared uint scratch[64]` gives operand 0 = [64 x i32], an
       * ARRAY, with the address space folded into operand 1 instead --
       *
       *    flags = addrspace << 2 | explicit-type | isconst
       *
       * and `explicit-type` set is what says operand 0 is the value type.
       * Reading the address space off operand 0's type therefore finds
       * nothing to read, and every groupshared array looks like address
       * space 0. Both encodings are handled because only the flag bit
       * distinguishes them.
       */
      uint32_t type = rec->num_ops >= 1 ? (uint32_t)rec->ops[0] : 0;
      uint64_t flags = rec->num_ops >= 2 ? rec->ops[1] : 0;
      uint32_t addr_space = D12_ADDR_SPACE_DEFAULT;

      if (flags & 2) {
         addr_space = (uint32_t)(flags >> 2);
      } else if (type < mod->num_types &&
                 mod->types[type].kind == D12_TYPE_POINTER) {
         addr_space = mod->types[type].ptr.addr_space;
         type = mod->types[type].ptr.pointee;
      }

      struct d12_value *v = push_value(mod, D12_VALUE_GLOBAL, type);
      if (!v)
         return false;
      v->addr_space = addr_space;
      v->global_is_const = (flags & 1) != 0;
      v->global_init_id = rec->num_ops >= 3 ? (uint32_t)rec->ops[2] : 0;
      break;
   }

   case D12_MODULE_FUNCTION: {
      /* [type, callingconv, isproto, ...] */
      if (mod->num_functions >= D12_MAX_FUNCTIONS) {
         fail(mod, "too many functions");
         return false;
      }

      struct d12_function *f = &mod->functions[mod->num_functions];
      memset(f, 0, sizeof(*f));
      f->type = rec->num_ops >= 1 ? (uint32_t)rec->ops[0] : 0;
      f->is_declaration = rec->num_ops >= 3 && rec->ops[2] != 0;
      f->value_index = mod->num_values;

      struct d12_value *v =
         push_value(mod, D12_VALUE_FUNCTION, f->type);
      if (!v)
         return false;
      v->func_index = mod->num_functions;

      mod->num_functions++;
      break;
   }

   default:
      break;
   }

   return true;
}

/*
 * Function bodies. LLVM 3.2+ encodes value operands as RELATIVE ids:
 * (current value number - operand). Resolve them here so downstream code
 * sees absolute indices.
 */
static uint32_t
rel_to_abs(struct parse_ctx *ctx, uint64_t rel)
{
   uint32_t mod_vals =
      ctx->module_values_latched ? ctx->module_values : ctx->mod->num_values;
   uint32_t base = ctx->cur_func ? ctx->cur_func->first_value : mod_vals;

   /* Where LLVM thinks we are: module values plus this function's own
    * locals so far, NOT the global array's length.
    */
   uint32_t locals = ctx->mod->num_values - base;
   uint32_t logical = mod_vals + locals;

   if (rel == 0 || rel > logical)
      return ~0u; /* forward reference or invalid; G4c resolves phis */

   uint32_t v = logical - (uint32_t)rel;
   if (v < mod_vals)
      return v; /* a module-level value: same slot either way */

   return base + (v - mod_vals); /* a local of the function being parsed */
}

static bool
parse_function_record(struct parse_ctx *ctx, const struct d12_record *rec)
{
   struct d12_module *mod = ctx->mod;
   struct d12_function *f = ctx->cur_func;

   if (!f)
      return true;

   if (rec->code == D12_FUNC_DECLAREBLOCKS) {
      f->num_blocks = rec->num_ops >= 1 ? (unsigned)rec->ops[0] : 0;
      return true;
   }

   /* Debug location records carry no dataflow. */
   if (rec->code == D12_FUNC_DEBUG_LOC || rec->code == D12_FUNC_DEBUG_LOC_AGAIN)
      return true;

   if (f->num_insts >= D12_MAX_INSTS) {
      fail(mod, "too many instructions in function");
      return false;
   }

   struct d12_inst *inst = &f->insts[f->num_insts];
   memset(inst, 0, sizeof(*inst));
   inst->code = rec->code;
   inst->result_type = ~0u;
   inst->result_value = ~0u;

   bool defines_value = false;

   switch (rec->code) {
   case D12_FUNC_INST_CALL: {
      /* [paramattr, cc, fnty?, fnid, args...]. Bit 15 of the cc field
       * (CALL_EXPLICIT_TYPE) says whether the function type is spelled
       * out before the callee. Everything from the callee on is a
       * relative value id.
       */
      if (rec->num_ops < 3) {
         fail(mod, "malformed call record");
         return false;
      }

      unsigned opnum = 2;
      if ((rec->ops[1] >> 15) & 1) {
         inst->extra[inst->num_extra++] = rec->ops[opnum]; /* fnty */
         opnum++;
      }

      /* ops[0] is the callee; ops[1..] are the arguments. */
      for (; opnum < rec->num_ops && inst->num_ops < D12_MAX_INST_OPS; opnum++)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[opnum]);

      /* A call defines a value only if the callee does not return void.
       * Getting this wrong shifts every later value index in the
       * function, so resolve it from the callee's type here.
       */
      defines_value = false;
      if (inst->num_ops >= 1 && inst->ops[0] < mod->num_values) {
         uint32_t fty = mod->values[inst->ops[0]].type;

         /* Function values are typed as pointer-to-function in LLVM. */
         if (fty < mod->num_types &&
             mod->types[fty].kind == D12_TYPE_POINTER)
            fty = mod->types[fty].ptr.pointee;

         if (fty < mod->num_types &&
             mod->types[fty].kind == D12_TYPE_FUNCTION) {
            uint32_t ret = mod->types[fty].func.ret;
            bool returns_void = ret < mod->num_types &&
                                mod->types[ret].kind == D12_TYPE_VOID;
            if (!returns_void) {
               defines_value = true;
               inst->result_type = ret;
            }
         }
      }
      break;
   }

   case D12_FUNC_INST_BINOP:
      /* [lhs, rhs, opcode, (flags)] */
      if (rec->num_ops >= 3) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[1]);
         inst->extra[inst->num_extra++] = rec->ops[2];

         /* A binop's type is its operands' type, and it must be recorded
          * so a CHAIN of them stays typed: `a*b + c` is a float add only
          * if the multiply that feeds it was known to be float.
          */
         if (inst->ops[0] < mod->num_values)
            inst->result_type = mod->values[inst->ops[0]].type;
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_CAST:
      /* [value, desttype, castopcode] */
      if (rec->num_ops >= 3) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->result_type = (uint32_t)rec->ops[1];
         inst->extra[inst->num_extra++] = rec->ops[2];
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_CMP:
   case D12_FUNC_INST_CMP2:
      if (rec->num_ops >= 3) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[1]);
         inst->extra[inst->num_extra++] = rec->ops[2];
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_SELECT:
   case D12_FUNC_INST_VSELECT:
      for (unsigned i = 0; i < rec->num_ops && inst->num_ops < 3; i++)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[i]);

      /* LLVM does not spell out a select's result type: it is the type of
       * either selected value (operands 0 and 1).  Preserve it here so a
       * floating binop fed by a select remains floating.  Leaving this at
       * ~0u silently turned e.g. `select ...; mul float` into integer
       * multiplication on the float bit patterns. */
      if (inst->num_ops >= 1 && inst->ops[0] < mod->num_values)
         inst->result_type = mod->values[inst->ops[0]].type;
      defines_value = true;
      break;

   case D12_FUNC_INST_EXTRACTVAL:
      if (rec->num_ops >= 2) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         for (unsigned i = 1; i < rec->num_ops && inst->num_extra < 4; i++)
            inst->extra[inst->num_extra++] = rec->ops[i];

         /*
          * LLVM does not store the extracted type -- it is the aggregate's
          * member at the given index. Deriving it matters because DXIL
          * binops are polymorphic: `mul` is integer or floating point
          * purely by operand TYPE. Leaving this unset made every
          * arithmetic on a struct-returning dx.op result (a texture
          * sample, a buffer load) compile to INTEGER ops on float bits.
          */
         uint32_t agg = inst->ops[0];
         if (agg < mod->num_values && inst->num_extra >= 1) {
            uint32_t t = mod->values[agg].type;
            if (t < mod->num_types) {
               const struct d12_type *at = &mod->types[t];
               unsigned idx = (unsigned)inst->extra[0];
               if ((at->kind == D12_TYPE_STRUCT_ANON ||
                    at->kind == D12_TYPE_STRUCT_NAMED) &&
                   idx < at->strct.num_members)
                  inst->result_type = at->strct.members[idx];
               else if ((at->kind == D12_TYPE_ARRAY ||
                         at->kind == D12_TYPE_VECTOR))
                  inst->result_type = at->seq.elem;
            }
         }
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_RET:
      if (rec->num_ops >= 1)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
      break;

   case D12_FUNC_INST_BR:
      /* [bb#] or [true bb#, false bb#, cond] */
      for (unsigned i = 0; i < rec->num_ops && inst->num_extra < 4; i++)
         inst->extra[inst->num_extra++] = rec->ops[i];
      if (rec->num_ops >= 3)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[2]);
      break;

   case D12_FUNC_INST_STORE:
      /* [ptr, val, align, isvolatile] */
      if (rec->num_ops >= 2) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[1]);
      }
      break;

   case D12_FUNC_INST_LOAD:
      /* [ptr, ty, align, isvolatile] */
      if (rec->num_ops >= 2) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->result_type = (uint32_t)rec->ops[1];
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_GEP:
   case D12_FUNC_INST_GEP_OLD:
      for (unsigned i = (rec->code == D12_FUNC_INST_GEP ? 2 : 0);
           i < rec->num_ops && inst->num_ops < D12_MAX_INST_OPS; i++)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[i]);
      defines_value = true;
      break;

   case D12_FUNC_INST_PHI:
      /* [ty, (val, bb)...]. The values are signed-relative to this
       * instruction's own value number, which is how a phi can name a
       * value defined later in the module -- a loop back edge.
       */
      if (rec->num_ops >= 1)
         inst->result_type = (uint32_t)rec->ops[0];

      for (unsigned i = 1; i + 1 < rec->num_ops &&
                           inst->num_phi_srcs < D12_MAX_PHI_SOURCES;
           i += 2) {
         struct d12_phi_src *src = &inst->phi_srcs[inst->num_phi_srcs++];
         int64_t rel = (int64_t)decode_signed_vbr(rec->ops[i]);
         int64_t abs = (int64_t)ctx->mod->num_values - rel;

         src->value = (abs >= 0 && abs < (int64_t)D12_MAX_VALUES)
                         ? (uint32_t)abs
                         : ~0u;
         src->block = (uint32_t)rec->ops[i + 1];
      }

      defines_value = true;
      break;

   case D12_FUNC_INST_ATOMICRMW:
      /* [ptr, val, operation, vol, ordering, ssid]. Only the first two
       * are value references; the operation is a LITERAL, and running it
       * through rel_to_abs silently turned it into a bogus value index.
       */
      if (rec->num_ops >= 3) {
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[0]);
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[1]);
         inst->extra[inst->num_extra++] = rec->ops[2];
      }
      defines_value = true;
      break;

   case D12_FUNC_INST_CMPXCHG:
      /* [ptr, cmp, new, vol, successordering, ssid, failureordering,
       * weak]: three value references, the rest literals.
       */
      for (unsigned i = 0; i < rec->num_ops && inst->num_ops < 3; i++)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[i]);
      defines_value = true;
      break;

   case D12_FUNC_INST_EXTRACTELT:
   case D12_FUNC_INST_INSERTELT:
   case D12_FUNC_INST_SHUFFLEVEC:
      for (unsigned i = 0; i < rec->num_ops && inst->num_ops < 3; i++)
         inst->ops[inst->num_ops++] = rel_to_abs(ctx, rec->ops[i]);
      defines_value = true;
      break;

   case D12_FUNC_INST_ALLOCA:
      if (rec->num_ops >= 1)
         inst->result_type = (uint32_t)rec->ops[0];
      defines_value = true;
      break;

   default:
      /* Unhandled instruction: recorded so the dump can report it, but it
       * must not silently define a value and shift every later index.
       * When one of these DOES define a value, every later operand in
       * the function resolves one off -- so say which code it was.
       */
      mesa_logw("dxil: unnumbered instruction record code %u (%u ops)",
                rec->code, rec->num_ops);
      break;
   }

   /* A call to a void function defines nothing. We cannot know that until
    * the callee is resolved, so CALL provisionally defines a value and G4c
    * corrects using the callee type. For everything else the rule above is
    * exact.
    */
   if (defines_value) {
      struct d12_value *v =
         push_value(mod, D12_VALUE_INSTRUCTION, inst->result_type);
      if (!v)
         return false;
      inst->result_value = mod->num_values - 1;
   }

   /* Basic block boundaries are implicit in LLVM bitcode: a block ends at
    * a terminator and the next instruction begins the next one. Recording
    * the block per instruction here means the translator does not have to
    * rediscover it.
    */
   inst->block = ctx->cur_block;
   if (d12_inst_is_terminator(inst))
      ctx->cur_block++;

   f->num_insts++;
   return true;
}

/*
 * Visitor plumbing
 */
static bool
on_enter_block(void *user, unsigned block_id, unsigned depth)
{
   struct parse_ctx *ctx = user;
   struct d12_module *mod = ctx->mod;

   if (block_id == D12_BLOCK_CONSTANTS)
      ctx->const_type = 0;

   if (block_id == D12_BLOCK_FUNCTION) {
      /* Bodies appear in module order, skipping declarations. */
      while (ctx->next_func_to_define < mod->num_functions &&
             mod->functions[ctx->next_func_to_define].is_declaration)
         ctx->next_func_to_define++;

      if (ctx->next_func_to_define >= mod->num_functions) {
         fail(mod, "function body without a matching declaration");
         return false;
      }

      struct d12_function *f = &mod->functions[ctx->next_func_to_define++];
      f->insts = calloc(D12_MAX_INSTS, sizeof(*f->insts));
      if (!f->insts) {
         fail(mod, "out of memory for function body");
         return false;
      }

      if (!ctx->module_values_latched) {
         ctx->module_values = mod->num_values;
         ctx->module_values_latched = true;
      }

      /* Arguments occupy value slots before any instruction. */
      f->first_value = mod->num_values;
      if (f->type < mod->num_types) {
         const struct d12_type *ft = &mod->types[f->type];
         if (ft->kind == D12_TYPE_FUNCTION) {
            for (unsigned i = 0; i < ft->func.num_params; i++) {
               if (!push_value(mod, D12_VALUE_ARGUMENT, ft->func.params[i]))
                  return false;
            }
         }
      }

      ctx->cur_func = f;
      ctx->cur_block = 0;
   }

   return true;
}

static bool
on_exit_block(void *user, unsigned block_id, unsigned depth)
{
   struct parse_ctx *ctx = user;

   if (block_id == D12_BLOCK_FUNCTION)
      ctx->cur_func = NULL;

   return true;
}

static bool
on_record(void *user, unsigned block_id, unsigned depth,
          const struct d12_record *rec, unsigned abbrev_id)
{
   struct parse_ctx *ctx = user;

   switch (block_id) {
   case D12_BLOCK_MODULE:
      return parse_module_record(ctx, rec);
   case D12_BLOCK_TYPE:
      return parse_type_record(ctx, rec);
   case D12_BLOCK_CONSTANTS:
      return parse_const_record(ctx, rec);
   case D12_BLOCK_METADATA:
      return parse_metadata_record(ctx, rec);
   case D12_BLOCK_FUNCTION:
      return parse_function_record(ctx, rec);
   case D12_BLOCK_VALUE_SYMTAB:
      /* [valueid, chars...] — names for globals, functions, and locals */
      if (rec->code == D12_VST_ENTRY && rec->num_ops >= 2) {
         uint64_t id = rec->ops[0];
         const char *name =
            intern_chars(ctx->mod, rec->ops + 1, rec->num_ops - 1);
         if (id < ctx->mod->num_values)
            ctx->mod->values[id].name = name;
      }
      return true;
   default:
      return true;
   }
}

bool
d12_dxil_module_parse(struct d12_module *mod, const uint8_t *bitcode,
                      size_t size_B)
{
   memset(mod, 0, sizeof(*mod));

   struct d12_bitstream bs;
   d12_bitstream_init(&bs, bitcode, size_B);

   struct parse_ctx ctx = {
      .mod = mod,
      .pending_struct_name = ~0u,
      .next_func_to_define = 0,
   };

   struct d12_block_visitor visitor = {
      .user = &ctx,
      .enter_block = on_enter_block,
      .record = on_record,
      .exit_block = on_exit_block,
   };

   if (!d12_dxil_walk(&bs, &visitor)) {
      if (!mod->error)
         mod->error = bs.error_msg ? bs.error_msg : "bitstream walk failed";
      return false;
   }

   /* Names arrive in VALUE_SYMTAB after the function table is built; copy
    * them onto the functions for convenience.
    */
   for (unsigned i = 0; i < mod->num_functions; i++) {
      struct d12_function *f = &mod->functions[i];
      if (f->value_index < mod->num_values)
         f->name = mod->values[f->value_index].name;
   }

   /* Signatures and properties need the finished value/metadata tables. */
   parse_entry_signatures(mod);
   parse_entry_properties(mod);
   parse_resources(mod);

   return mod->error == NULL;
}

void
d12_dxil_module_finish(struct d12_module *mod)
{
   for (unsigned i = 0; i < mod->num_functions; i++)
      free(mod->functions[i].insts);
   for (unsigned i = 0; i < mod->num_values; i++)
      free(mod->values[i].data_elems);
   for (struct d12_str_chunk *c = mod->strings; c;) {
      struct d12_str_chunk *next = c->next;
      free(c);
      c = next;
   }
   free(mod->resources);
   memset(mod, 0, sizeof(*mod));
}

/*
 * Signature metadata. Resolve a metadata operand that is an MD_VALUE
 * wrapping an integer constant.
 */
static bool
md_const_uint(const struct d12_module *mod, uint64_t md_id, uint64_t *out)
{
   if (md_id >= mod->num_metadata)
      return false;

   const struct d12_metadata *md = &mod->metadata[md_id];
   if (md->kind != D12_MD_VALUE || md->value_id >= mod->num_values)
      return false;

   const struct d12_value *v = &mod->values[md->value_id];
   if (v->kind != D12_VALUE_CONSTANT)
      return false;

   *out = v->is_null ? 0 : v->const_int;
   return true;
}

static const char *
md_string(const struct d12_module *mod, uint64_t md_id)
{
   if (md_id >= mod->num_metadata)
      return NULL;

   const struct d12_metadata *md = &mod->metadata[md_id];
   return md->kind == D12_MD_STRING ? md->string : NULL;
}

/* Parse one signature list node into an element array. */
static void
parse_signature(struct d12_module *mod, uint64_t list_id,
                struct d12_sig_element *out, unsigned *out_count)
{
   *out_count = 0;

   if (list_id >= mod->num_metadata)
      return;

   const struct d12_metadata *list = &mod->metadata[list_id];
   if (list->kind != D12_MD_NODE)
      return;

   for (unsigned i = 0;
        i < list->node.num_ops && *out_count < D12_MAX_SIG_ELEMENTS; i++) {
      if (list->node.op_is_null[i])
         continue;

      uint64_t elem_id = list->node.ops[i];
      if (elem_id >= mod->num_metadata)
         continue;

      const struct d12_metadata *elem = &mod->metadata[elem_id];
      if (elem->kind != D12_MD_NODE || elem->node.num_ops < 10)
         continue;

      struct d12_sig_element *se = &out[(*out_count)++];
      memset(se, 0, sizeof(*se));

      uint64_t v;
      if (md_const_uint(mod, elem->node.ops[0], &v))
         se->id = (uint32_t)v;

      se->name = md_string(mod, elem->node.ops[1]);

      if (md_const_uint(mod, elem->node.ops[2], &v))
         se->component_type = (uint32_t)v;
      if (md_const_uint(mod, elem->node.ops[3], &v))
         se->system_value = (uint32_t)v;

      /* Operand 4 is the semantic INDEX vector -- one entry per row. A
       * D3D12 input layout binds by (SemanticName, SemanticIndex), so
       * without this TEXCOORD0 and TEXCOORD1 are indistinguishable.
       */
      if (!elem->node.op_is_null[4] && elem->node.ops[4] < mod->num_metadata) {
         const struct d12_metadata *iv = &mod->metadata[elem->node.ops[4]];
         if (iv->kind == D12_MD_NODE && iv->node.num_ops >= 1 &&
             md_const_uint(mod, iv->node.ops[0], &v))
            se->semantic_index = (uint32_t)v;
      }

      if (md_const_uint(mod, elem->node.ops[5], &v))
         se->interpolation = (uint32_t)v;

      if (md_const_uint(mod, elem->node.ops[6], &v))
         se->rows = (uint32_t)v;
      if (md_const_uint(mod, elem->node.ops[7], &v))
         se->cols = (uint32_t)v;
      if (md_const_uint(mod, elem->node.ops[8], &v))
         se->start_row = (uint32_t)v;
      if (md_const_uint(mod, elem->node.ops[9], &v))
         se->start_col = (uint32_t)v;
   }
}

static void
parse_entry_signatures(struct d12_module *mod)
{
   const struct d12_named_md *eps = d12_dxil_find_named_md(mod,
                                                           "dx.entryPoints");
   if (!eps || !eps->num_nodes)
      return;

   uint64_t ep_id = eps->nodes[0];
   if (ep_id >= mod->num_metadata)
      return;

   /* Entry node: {function, name, signatures, resources, properties} */
   const struct d12_metadata *ep = &mod->metadata[ep_id];
   if (ep->kind != D12_MD_NODE || ep->node.num_ops < 3 ||
       ep->node.op_is_null[2])
      return;

   uint64_t sigs_id = ep->node.ops[2];
   if (sigs_id >= mod->num_metadata)
      return;

   /* Signatures node: {input, output, patch-constant} */
   const struct d12_metadata *sigs = &mod->metadata[sigs_id];
   if (sigs->kind != D12_MD_NODE)
      return;

   if (sigs->node.num_ops >= 1 && !sigs->node.op_is_null[0]) {
      parse_signature(mod, sigs->node.ops[0], mod->input_sig,
                      &mod->num_input_sig);
   }

   if (sigs->node.num_ops >= 2 && !sigs->node.op_is_null[1]) {
      parse_signature(mod, sigs->node.ops[1], mod->output_sig,
                      &mod->num_output_sig);
   }

   if (sigs->node.num_ops >= 3 && !sigs->node.op_is_null[2]) {
      parse_signature(mod, sigs->node.ops[2], mod->patch_sig,
                      &mod->num_patch_sig);
   }
}

/*
 * Declared resources, from the dx.resources named metadata.
 *
 * dx.resources is a node of four lists — SRVs, UAVs, CBVs, samplers — and
 * each entry shares a common prefix:
 *
 *   0 id   1 variable   2 name   3 space   4 lowerBound   5 rangeSize
 *
 * after which the classes diverge. What we need past the prefix is the
 * structured-buffer element stride, which lives in a trailing tag list as
 * (tag, value) pairs with tag 1 = element stride, and the CBV byte size.
 *
 * Field order verified against dxc (SDK 26100) output for resources placed
 * at t2/u3/b1 with a 12-byte structured element, so that every field is
 * distinct and a transposition cannot go unnoticed.
 */
#define D12_RES_TAG_ELEMENT_STRIDE 1

/* Offset of the tag list within each class's record. */
#define D12_SRV_TAGS_OP 8
#define D12_UAV_TAGS_OP 10
#define D12_CBV_SIZE_OP 6

static uint32_t
parse_resource_stride(const struct d12_module *mod, uint64_t tags_id)
{
   if (tags_id >= mod->num_metadata)
      return 0;

   const struct d12_metadata *tags = &mod->metadata[tags_id];
   if (tags->kind != D12_MD_NODE)
      return 0;

   for (unsigned i = 0; i + 1 < tags->node.num_ops; i += 2) {
      uint64_t tag, value;
      if (!md_const_uint(mod, tags->node.ops[i], &tag))
         continue;
      if (!md_const_uint(mod, tags->node.ops[i + 1], &value))
         continue;

      if (tag == D12_RES_TAG_ELEMENT_STRIDE)
         return (uint32_t)value;
   }

   return 0;
}

static void
parse_resource_list(struct d12_module *mod, uint64_t list_id, unsigned klass)
{
   if (list_id >= mod->num_metadata)
      return;

   const struct d12_metadata *list = &mod->metadata[list_id];
   if (list->kind != D12_MD_NODE)
      return;

   for (unsigned i = 0; i < list->node.num_ops; i++) {
      if (list->node.op_is_null[i])
         continue;

      uint64_t rec_id = list->node.ops[i];
      if (rec_id >= mod->num_metadata)
         continue;

      const struct d12_metadata *rec = &mod->metadata[rec_id];
      if (rec->kind != D12_MD_NODE || rec->node.num_ops < 6)
         continue;

      if (mod->num_resources == mod->resource_capacity) {
         if (mod->resource_capacity >= D12_MAX_METADATA) {
            fail(mod, "too many DXIL resource declarations");
            return;
         }

         unsigned capacity = mod->resource_capacity
                                ? mod->resource_capacity * 2
                                : 32;
         capacity = MIN2(capacity, D12_MAX_METADATA);
         if (capacity < mod->resource_capacity ||
             (size_t)capacity >
                SIZE_MAX / sizeof(struct d12_resource_binding)) {
            fail(mod, "DXIL resource declaration size overflow");
            return;
         }

         void *resources = realloc(
            mod->resources,
            (size_t)capacity * sizeof(struct d12_resource_binding));
         if (!resources) {
            fail(mod, "out of memory growing DXIL resource declarations");
            return;
         }
         mod->resources = resources;
         mod->resource_capacity = capacity;
      }

      struct d12_resource_binding *rb = &mod->resources[mod->num_resources];
      memset(rb, 0, sizeof(*rb));
      rb->klass = klass;
      rb->global_value = ~0u;

      /* Operand 1 names the global variable standing for this resource. */
      if (rec->node.ops[1] < mod->num_metadata) {
         const struct d12_metadata *var = &mod->metadata[rec->node.ops[1]];
         if (var->kind == D12_MD_VALUE)
            rb->global_value = (uint32_t)var->value_id;
      }

      /* Operand 2 is the source-level name. */
      rb->name = md_string(mod, rec->node.ops[2]);

      uint64_t v;
      if (md_const_uint(mod, rec->node.ops[0], &v))
         rb->id = (uint32_t)v;
      if (md_const_uint(mod, rec->node.ops[3], &v))
         rb->space = (uint32_t)v;
      if (md_const_uint(mod, rec->node.ops[4], &v))
         rb->lower_bound = (uint32_t)v;
      if (md_const_uint(mod, rec->node.ops[5], &v))
         rb->range_size = (uint32_t)v;

      /* Operand 6 is the shape for SRVs and UAVs; for CBVs it is the byte
       * size instead, handled below.
       */
      if (klass != D12_RES_CBV && rec->node.num_ops > 6 &&
          md_const_uint(mod, rec->node.ops[6], &v))
         rb->kind = (uint32_t)v;

      unsigned tags_op = 0;
      switch (klass) {
      case D12_RES_SRV:
         tags_op = D12_SRV_TAGS_OP;
         break;
      case D12_RES_UAV:
         tags_op = D12_UAV_TAGS_OP;
         break;
      case D12_RES_CBV:
         if (rec->node.num_ops > D12_CBV_SIZE_OP &&
             md_const_uint(mod, rec->node.ops[D12_CBV_SIZE_OP], &v))
            rb->cb_size_B = (uint32_t)v;
         break;
      default:
         break;
      }

      if (tags_op && rec->node.num_ops > tags_op &&
          !rec->node.op_is_null[tags_op]) {
         rb->stride_B =
            parse_resource_stride(mod, rec->node.ops[tags_op]);
      }

      mod->num_resources++;
   }
}

static void
parse_resources(struct d12_module *mod)
{
   const struct d12_named_md *res =
      d12_dxil_find_named_md(mod, "dx.resources");
   if (!res || !res->num_nodes)
      return;

   uint64_t root_id = res->nodes[0];
   if (root_id >= mod->num_metadata)
      return;

   const struct d12_metadata *root = &mod->metadata[root_id];
   if (root->kind != D12_MD_NODE)
      return;

   /* Four lists in class order, any of which may be null. */
   for (unsigned klass = 0;
        klass < D12_RES_CLASS_COUNT && klass < root->node.num_ops; klass++) {
      if (root->node.op_is_null[klass])
         continue;
      parse_resource_list(mod, root->node.ops[klass], klass);
      if (mod->error)
         return;
   }
}

/*
 * Entry-point properties are a flat list of (tag, value) pairs. Tag 4 is
 * kNumThreads, whose value is a node of three integer constants — the
 * [numthreads(x,y,z)] attribute. Without it the launch grid collapses to
 * zero threads and the dispatch silently does nothing.
 */
/*
 * Entry-point property tags. Measured against real DXC output, not
 * assumed: a first pass had these one lower each, which silently left
 * every geometry shader with vertices_out = 0 -- and let two of them
 * compile to a bare `stop`. The hull path looked correct only because
 * its default control-point count happened to match the source.
 *
 *   1  GSState  {inputPrim, maxVertexCount, streamMask, outputTopology,
 *                instanceCount}
 *   2  DSState  {domain, inputControlPoints}
 *   3  HSState  {patchConstantFn, inCP, outCP, domain, partitioning,
 *                outputPrimitive, maxTessFactor}
 *   4  NumThreads {x, y, z}
 */
#define D12_ENTRY_PROP_GS_STATE     1
#define D12_ENTRY_PROP_DS_STATE     2
#define D12_ENTRY_PROP_HS_STATE     3
#define D12_ENTRY_PROP_NUM_THREADS  4
/* MSState / ASState. Operand 0 is the numthreads NODE -- mesh and
 * amplification shaders do not use the compute NumThreads tag.
 */
#define D12_ENTRY_PROP_MS_STATE     9
#define D12_ENTRY_PROP_AS_STATE     10

static void
parse_entry_properties(struct d12_module *mod)
{
   const struct d12_named_md *eps =
      d12_dxil_find_named_md(mod, "dx.entryPoints");
   if (!eps || !eps->num_nodes || eps->nodes[0] >= mod->num_metadata)
      return;

   const struct d12_metadata *ep = &mod->metadata[eps->nodes[0]];
   if (ep->kind != D12_MD_NODE || ep->node.num_ops < 5 ||
       ep->node.op_is_null[4])
      return;

   uint64_t props_id = ep->node.ops[4];
   if (props_id >= mod->num_metadata)
      return;

   const struct d12_metadata *props = &mod->metadata[props_id];
   if (props->kind != D12_MD_NODE)
      return;

   for (unsigned i = 0; i + 1 < props->node.num_ops; i += 2) {
      if (props->node.op_is_null[i] || props->node.op_is_null[i + 1])
         continue;

      uint64_t tag;
      if (!md_const_uint(mod, props->node.ops[i], &tag))
         continue;

      uint64_t val_id = props->node.ops[i + 1];

      if (val_id >= mod->num_metadata)
         continue;

      const struct d12_metadata *val = &mod->metadata[val_id];

      if (d12_debug()->dump_entry_props) {
         fprintf(stderr, "entry prop tag %u:", (unsigned)tag);
         if (val->kind == D12_MD_NODE) {
            for (unsigned k = 0; k < val->node.num_ops; k++) {
               uint64_t cv;
               if (md_const_uint(mod, val->node.ops[k], &cv))
                  fprintf(stderr, " %llu", (unsigned long long)cv);
               else
                  fprintf(stderr, " ?");
            }
         }
         fprintf(stderr, "\n");
      }

      if (tag == D12_ENTRY_PROP_NUM_THREADS) {
         if (val->kind != D12_MD_NODE || val->node.num_ops < 3)
            continue;
         for (unsigned c = 0; c < 3; c++) {
            uint64_t v = 1;
            if (md_const_uint(mod, val->node.ops[c], &v))
               mod->num_threads[c] = (uint32_t)v;
         }
      } else if (tag == D12_ENTRY_PROP_MS_STATE ||
                 tag == D12_ENTRY_PROP_AS_STATE) {
         /* MSState { numThreads, maxVertexCount, maxPrimitiveCount,
          *           outputTopology, payloadSizeInBytes }
          * ASState { numThreads, payloadSizeInBytes }
          */
         if (val->kind != D12_MD_NODE || val->node.num_ops < 2)
            continue;

         uint64_t nt_id = val->node.ops[0];
         if (nt_id < mod->num_metadata) {
            const struct d12_metadata *nt = &mod->metadata[nt_id];
            if (nt->kind == D12_MD_NODE && nt->node.num_ops >= 3) {
               for (unsigned c = 0; c < 3; c++) {
                  uint64_t v = 1;
                  if (md_const_uint(mod, nt->node.ops[c], &v))
                     mod->num_threads[c] = (uint32_t)v;
               }
            }
         }

         uint64_t v;
         if (tag == D12_ENTRY_PROP_MS_STATE && val->node.num_ops >= 5) {
            if (md_const_uint(mod, val->node.ops[1], &v))
               mod->mesh_max_vertices = (uint32_t)v;
            if (md_const_uint(mod, val->node.ops[2], &v))
               mod->mesh_max_primitives = (uint32_t)v;
            if (md_const_uint(mod, val->node.ops[3], &v))
               mod->mesh_topology = (uint32_t)v;
            if (md_const_uint(mod, val->node.ops[4], &v))
               mod->mesh_payload_size = (uint32_t)v;
         } else if (md_const_uint(mod, val->node.ops[1], &v)) {
            mod->mesh_payload_size = (uint32_t)v;
         }
      } else if (tag == D12_ENTRY_PROP_HS_STATE) {
         /* {patchConstantFn, inputCPs, outputCPs, domain, partitioning,
          *  outputPrimitive, maxTessFactor}
          */
         if (val->kind != D12_MD_NODE || val->node.num_ops < 6)
            continue;
         uint64_t v;

         /* Operand 0 names the patch constant function, as a metadata
          * value wrapping a function reference. Resolve it to an index.
          */
         if (val->node.ops[0] < mod->num_metadata) {
            const struct d12_metadata *fn = &mod->metadata[val->node.ops[0]];
            if (fn->kind == D12_MD_VALUE) {
               for (unsigned f = 0; f < mod->num_functions; f++) {
                  if (mod->functions[f].value_index == fn->value_id) {
                     mod->patch_constant_func = f;
                     mod->has_patch_constant = true;
                     break;
                  }
               }
            }
         }

         if (md_const_uint(mod, val->node.ops[2], &v))
            mod->tess_control_points = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[3], &v))
            mod->tess_domain = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[4], &v))
            mod->tess_partitioning = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[5], &v))
            mod->tess_output_prim = (uint32_t)v;
      } else if (tag == D12_ENTRY_PROP_DS_STATE) {
         /* {domain, inputControlPoints} */
         if (val->kind != D12_MD_NODE || val->node.num_ops < 2)
            continue;
         uint64_t v;
         if (md_const_uint(mod, val->node.ops[0], &v))
            mod->tess_domain = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[1], &v))
            mod->tess_control_points = (uint32_t)v;
      } else if (tag == D12_ENTRY_PROP_GS_STATE) {
         /* {inputPrimitive, maxVertexCount, streamMask, outputTopology,
          *  instanceCount}
          */
         if (val->kind != D12_MD_NODE || val->node.num_ops < 5)
            continue;
         uint64_t v;
         if (md_const_uint(mod, val->node.ops[0], &v))
            mod->gs_input_prim = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[1], &v))
            mod->gs_max_vertices = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[3], &v))
            mod->gs_output_topology = (uint32_t)v;
         if (md_const_uint(mod, val->node.ops[4], &v))
            mod->gs_instance_count = (uint32_t)v;
      }
   }
}

bool
d12_dxil_collect_texture_shapes(const struct d12_module *mod,
                                const struct d12_binding_map *map,
                                uint8_t *texture_shapes)
{
   if (!map)
      return true;

   for (unsigned i = 0; i < mod->num_resources; i++) {
      const struct d12_resource_binding *rb = &mod->resources[i];
      enum d12_texture_shape shape = d12_texture_shape_from_kind(rb->kind);
      if (shape == D12_TEXTURE_SHAPE_NONE)
         continue;

      uint64_t upper = rb->range_size == D12_RANGE_UNBOUNDED
                          ? UINT64_MAX
                          : (uint64_t)rb->lower_bound + rb->range_size;
      for (unsigned e = 0; e < map->count; e++) {
         if (map->entries[e].klass != rb->klass ||
             map->entries[e].space != rb->space ||
             map->entries[e].reg < rb->lower_bound ||
             map->entries[e].reg >= upper)
            continue;

         unsigned slot = map->entries[e].slot;
         if (texture_shapes[slot] != D12_TEXTURE_SHAPE_NONE &&
             texture_shapes[slot] != shape)
            return false;
         texture_shapes[slot] = shape;
      }
   }

   return true;
}

const struct d12_sig_element *
d12_dxil_find_sig(const struct d12_sig_element *sig, unsigned count,
                  uint32_t id)
{
   for (unsigned i = 0; i < count; i++) {
      if (sig[i].id == id)
         return &sig[i];
   }
   return NULL;
}

const struct d12_named_md *
d12_dxil_find_named_md(const struct d12_module *mod, const char *name)
{
   for (unsigned i = 0; i < mod->num_named_md; i++) {
      if (mod->named_md[i].name && !strcmp(mod->named_md[i].name, name))
         return &mod->named_md[i];
   }
   return NULL;
}

/*
 * Build a binding map from the module's own declarations, in declaration
 * order. Only for the standalone compiler tool, which has no root
 * signature: in the driver, compile time and bind time must agree on an
 * assignment that neither derives from the other, so both read the one on
 * the root signature.
 */
void
d12_binding_map_from_module(const struct d12_module *mod,
                            struct d12_binding_map *out)
{
   memset(out, 0, sizeof(*out));

   for (unsigned i = 0; i < mod->num_resources; i++) {
      const struct d12_resource_binding *rb = &mod->resources[i];

      /* An unbounded array cannot be enumerated, but it still needs a
       * BASE slot for the dynamic-handle path to index from -- one slot,
       * carrying the range's stride and kind. The runtime bound is the
       * descriptor heap's, not the shader's.
       */
      unsigned count = d12_range_count(rb);
      if (!count)
         count = 1;

      for (unsigned r = 0; r < count; r++) {
         if (d12_binding_map_add(out, rb->klass, rb->space,
                                 rb->lower_bound + r) == ~0u) {
            /* Full. Later RANGES still deserve their base slots --
             * losing them fails whole shaders for registers they never
             * touch -- so stop enumerating this range but keep going.
             */
            mesa_logw("dxil: binding map full at class %u space %u "
                      "register %u; higher registers unmapped",
                      rb->klass, rb->space, rb->lower_bound + r);
            break;
         }
      }
   }
}
