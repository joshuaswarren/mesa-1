#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Compile the production argument conversion plus actual legacy/new kernel
address helpers as CPU C. No DRM node, GPU, DXIL rewrite or device is accessed.
"""
from pathlib import Path
import os,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[4]
cmd=(ROOT/'src/asahi/d3d12/d12_cmd.c').read_text()
cl=(ROOT/'src/asahi/libagx/copy.cl').read_text()
def function(name):
 start=cl.index(name+'(');start=cl.rfind('static uint',0,start)
 at=cl.index('{',start);depth,end=1,at+1
 while depth:depth+=(cl[end]=='{')-(cl[end]=='}');end+=1
 return cl[start:end]
a=cmd.index('   /* libagx_copy_image now');b=cmd.index('   /* The proven copy kernel',a)
conversion=cmd[a:b]
assert 'agx_2d(plan.width_el, plan.height_el)' in cmd[b:b+950]
helpers='\n'.join(function(name)for name in ['libagx_space_bits','libagx_space_bits_7','libagx_copy_image_offset','libagx_copy_image_offset_pow2'])
harness=r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
typedef uint32_t uint;
#define ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))
#define D12_LIST_ERROR 1
#define mesa_loge(...) ((void)0)
static bool util_is_power_of_two_nonzero(uint x){return x && !(x&(x-1));}
static uint util_logbase2(uint x){assert(x);return 31u-(uint)__builtin_clz(x);}
struct d12_cmd_list{int state;};
struct d12_copy_surface{uint pitch_B,stride_el,tile_width_el,tile_height_el,blocksize_B;bool tiled;};
struct plan{uint width_el,height_el,destination_x_el,destination_y_el,source_x_el,source_y_el;};
struct libagx_copy_image_args{uint width_el,height_el,dest_x_el,dest_y_el,src_x_el,src_y_el,dest_pitch_B,src_pitch_B,dest_tiles_per_row,src_tiles_per_row,dest_tile_width_log2,dest_tile_height_log2,src_tile_width_log2,src_tile_height_log2,blocksize_B,dest_tiled,src_tiled;};
static void convert(struct d12_cmd_list *list,struct d12_copy_surface d,struct d12_copy_surface s,struct plan plan,struct libagx_copy_image_args*out){
/* PRODUCTION_CONVERSION */
*out=args;
}
/* ACTUAL_KERNEL_HELPERS */
int main(void){
 unsigned long long checks=0;
 uint dims[]={1,8,32,64,128,256,512};
 for(unsigned wi=0;wi<ARRAY_SIZE(dims);wi++)for(unsigned hi=0;hi<ARRAY_SIZE(dims);hi++){
  uint w=dims[wi],h=dims[hi],stride=2*w+1;
  struct d12_copy_surface tiled={.stride_el=stride,.tile_width_el=w,.tile_height_el=h,.blocksize_B=4,.tiled=true};
  struct d12_copy_surface linear={.pitch_B=1232,.blocksize_B=4};
  struct plan plan={37,11,5,3,7,2};
  for(unsigned mode=0;mode<4;mode++){
   struct d12_copy_surface d=mode&1?tiled:linear,s=mode&2?tiled:linear;
   struct d12_cmd_list list={0};struct libagx_copy_image_args a={0};convert(&list,d,s,plan,&a);assert(!list.state);
   assert(a.width_el==37 && a.height_el==11 && a.dest_x_el==5 && a.src_y_el==2);
   assert(a.dest_pitch_B==d.pitch_B && a.src_pitch_B==s.pitch_B);
   for(uint y=0;y<2*h+1;y++)for(uint x=0;x<3*w+1;x++){
    uint old=libagx_copy_image_offset(x,y,s.pitch_B,s.stride_el,s.tile_width_el,s.tile_height_el,s.blocksize_B,s.tiled);
    uint now=libagx_copy_image_offset_pow2(x,y,a.src_pitch_B,a.src_tiles_per_row,a.src_tile_width_log2,a.src_tile_height_log2,a.blocksize_B,a.src_tiled);
    assert(old==now);checks++;
    old=libagx_copy_image_offset(x,y,d.pitch_B,d.stride_el,d.tile_width_el,d.tile_height_el,d.blocksize_B,d.tiled);
    now=libagx_copy_image_offset_pow2(x,y,a.dest_pitch_B,a.dest_tiles_per_row,a.dest_tile_width_log2,a.dest_tile_height_log2,a.blocksize_B,a.dest_tiled);
    assert(old==now);checks++;
   }
  }
 }
 for(unsigned bad=0;bad<4;bad++){
  struct d12_copy_surface s={.stride_el=128,.tile_width_el=32,.tile_height_el=32,.blocksize_B=4,.tiled=true};
  if(bad==0)s.stride_el=0;else if(bad==1)s.tile_width_el=3;else if(bad==2)s.tile_height_el=0;else s.tile_height_el=32768;
  for(unsigned endpoint=0;endpoint<2;endpoint++){
   struct d12_copy_surface good={.pitch_B=512,.blocksize_B=4};struct d12_cmd_list list={0};struct libagx_copy_image_args args={0};
   convert(&list,endpoint?s:good,endpoint?good:s,(struct plan){1,1,0,0,0,0},&args);assert(list.state==D12_LIST_ERROR);checks++;
  }
 }
 struct d12_copy_surface huge={.stride_el=UINT32_MAX,.tile_width_el=128,.tile_height_el=32,.blocksize_B=4,.tiled=true};
 struct d12_cmd_list list={0};struct libagx_copy_image_args args={0};convert(&list,huge,huge,(struct plan){1,1,0,0,0,0},&args);
 assert(!list.state && args.src_tiles_per_row==33554432u);checks++;
 printf("COPY API PORT: %llu address/conversion checks PASS; invalid geometry rejected; 2D call verified\n",checks);
}
'''
with tempfile.TemporaryDirectory(prefix='d12-copy-api-port-')as tmp:
 c=Path(tmp)/'test.c';c.write_text(harness.replace('/* PRODUCTION_CONVERSION */',conversion).replace('/* ACTUAL_KERNEL_HELPERS */',helpers));binary=Path(tmp)/'test'
 subprocess.run([os.environ.get('HOST_CC','clang'),'-std=c11','-O2','-Wall','-Wextra','-Werror',str(c),'-o',str(binary)],check=True)
 subprocess.run([str(binary)],check=True)
