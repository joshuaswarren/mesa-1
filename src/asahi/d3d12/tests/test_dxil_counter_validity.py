#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Execute production counter lowering with a scalar NIR evaluator.

An invalid dynamic handle must neither change a valid descriptor's counter nor
return its value. Check increment/decrement and root/heap address paths without
opening a GPU or importing external shader bytes.
"""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
source = (root / 'src/asahi/d3d12/dxil/d12_dxil_nir.c').read_text()
a = source.index('   case D12_DXOP_BUFFER_UPDATE_COUNTER: {')
a = source.index('{', a)
b = source.index('   case D12_DXOP_CHECK_ACCESS_FULLY_MAPPED:', a)
body = source[a:b]
harness = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct nir_def { uint64_t value; unsigned bit_size; };
typedef struct nir_def nir_def;
typedef struct { nir_def pool[100]; unsigned count; } nir_builder;
struct handle_ref { nir_def *slot, *valid; bool heap; };
struct trans_ctx {
 nir_builder builder; struct handle_ref handle;
 uint64_t descriptor[2], slots[1], increment;
 uint32_t counter, discard, result;
};
static nir_def *make(nir_builder *b, uint64_t v) {
 assert(b->count < 100); b->pool[b->count].value=v; b->pool[b->count].bit_size=64;
 return &b->pool[b->count++];
}
static nir_def *make32(nir_builder *b,uint64_t v) {nir_def *d=make(b,(uint32_t)v);d->bit_size=32;return d;}
static nir_def *nir_imm_int(nir_builder *b, int v) { return make32(b,v); }
static nir_def *nir_iadd_imm(nir_builder *b,nir_def *v,int i) {return v->bit_size==32?make32(b,v->value+i):make(b,v->value+i);}
static nir_def *nir_iadd(nir_builder *b,nir_def *a,nir_def *c) {return make(b,a->value+c->value);}
static nir_def *nir_imul_imm(nir_builder *b,nir_def *v,int i) {return make(b,v->value*i);}
static nir_def *nir_u2u64(nir_builder *b,nir_def *v) {return make(b,v->value);}
static nir_def *nir_bcsel(nir_builder *b,nir_def *c,nir_def *t,nir_def *f) {return t->bit_size==32?make32(b,c->value?t->value:f->value):make(b,c->value?t->value:f->value);}
static nir_def *load(nir_builder *b, unsigned n,unsigned bits,nir_def *addr) {
 assert(n==1 && bits==64);return make(b,*(uint64_t *)(uintptr_t)addr->value);
}
static nir_def *atomic(nir_builder *b,unsigned bits,nir_def *addr,nir_def *add) {
 assert(bits==32);uint32_t *p=(uint32_t *)(uintptr_t)addr->value;
 uint32_t before=*p;*p+=add->value;return make32(b,before);
}
#define nir_load_global(b,n,bits,addr,...) load(b,n,bits,addr)
#define nir_global_atomic(b,bits,addr,add,...) atomic(b,bits,addr,add)
#define NEED(n) assert((n)<=2)
#define D12_DESC_COUNTER_OFFSET 8
#define D12_ROOT_PLANE_COUNTER 0
#define D12_ROOT_PLANE_DISCARD 1
static bool handle_info(struct trans_ctx *c,uint32_t v,struct handle_ref *h) {(void)v;*h=c->handle;return true;}
static bool const_uint(struct trans_ctx *c,uint32_t v,uint64_t *out) {(void)v;*out=c->increment;return true;}
static void trans_fail(struct trans_ctx *c,const char *s) {(void)c;(void)s;assert(false);}
static nir_def *descriptor_address(struct trans_ctx *c,const struct handle_ref *h) {(void)h;return make(&c->builder,(uintptr_t)c->descriptor);}
static nir_def *root_plane(struct trans_ctx *c,int plane) {return make(&c->builder,plane?(uintptr_t)&c->discard:(uintptr_t)c->slots);}
static void def_result(struct trans_ctx *c,const void *inst,nir_def *v) {(void)inst;c->result=v->value;}
static bool translate(struct trans_ctx *ctx) {
 nir_builder *b=&ctx->builder;uint32_t p[2]={0,1};const void *inst=NULL;
 /* PRODUCTION_COUNTER_CASE */
}
int main(void) {
 unsigned checks=0;
 for(unsigned heap=0;heap<2;heap++)for(unsigned validity=0;validity<3;validity++)for(unsigned dec=0;dec<2;dec++) {
  struct trans_ctx c;memset(&c,0,sizeof(c));c.counter=25;c.discard=99;
  c.increment=dec?255:1;c.slots[0]=(uintptr_t)&c.counter;c.descriptor[1]=c.slots[0];
  c.handle.slot=make(&c.builder,0);c.handle.heap=heap;
  c.handle.valid=validity==2?NULL:make(&c.builder,validity);
  assert(translate(&c));
  bool valid=validity!=0;
  assert(c.counter==(valid?(dec?24:26):25));
  assert(c.result==(valid?(dec?24:25):0));checks++;
 }
 printf("DXIL COUNTER VALIDITY: %u root/heap and increment/decrement checks PASS\n",checks);
}
'''
with tempfile.TemporaryDirectory(prefix='d12-counter-validity-') as tmp:
    c = Path(tmp) / 'test.c'
    c.write_text(harness.replace('/* PRODUCTION_COUNTER_CASE */', body))
    binary = Path(tmp) / 'test'
    subprocess.run([os.environ.get('HOST_CC', 'clang'), '-std=c11', '-O2',
                    '-Wall', '-Wextra', '-Werror', str(c), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
