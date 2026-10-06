/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "../d12_g17p_txf_prefix_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(c) do {++checks; if(!(c)){fprintf(stderr,"FAIL %u %s\n",(unsigned)__LINE__,#c);exit(1);}} while(0)
static void test(struct d12_g17p_txf_prefix_sources s, bool wanted)
{
   struct d12_g17p_txf_range out[4], before[4];
   memset(out,0xa5,sizeof(out));memcpy(before,out,sizeof(out));
   CHECK(d12_g17p_txf_prefix_ranges(&s,out)==wanted);
   if (!wanted) {CHECK(!memcmp(out,before,sizeof(out)));return;}
   const unsigned start[4]={0,4,8,16};
   const uint64_t address[4]={s.texture_cell,s.sampler_cell,s.root_cell,s.texture_cell};
   for(unsigned i=0;i<4;i++){CHECK(out[i].start_halfs==start[i]);CHECK(out[i].size_halfs==4);CHECK(out[i].address==address[i]);}
   /* Every physical half has one owner; query16 aliases source cell only. */
   for(unsigned i=0;i<4;i++)for(unsigned j=i+1;j<4;j++)
      CHECK(out[i].start_halfs+out[i].size_halfs<=out[j].start_halfs);
}
int main(void)
{
   struct d12_g17p_txf_prefix_sources good={.root_cell=0x10000,.texture_cell=0x10008,.sampler_cell=0x10010,.texture_table=0x20000,.texture_table_size_B=0x2000,.sampler_table=0x30000};
   test(good,true);
   for(unsigned i=0;i<64;i++){struct d12_g17p_txf_prefix_sources s=good;s.root_cell=0x10000+i;test(s,i%8==0&&s.root_cell!=s.texture_cell&&s.root_cell!=s.sampler_cell);s=good;s.texture_table=0x20000+i;test(s,i%32==0);s=good;s.sampler_table=0x30000+i;test(s,i%32==0);}
   for(unsigned size=0;size<1024;size++){struct d12_g17p_txf_prefix_sources s=good;s.texture_table_size_B=size;test(s,size&&size%32==0);}
   const uint64_t limit=UINT64_C(1)<<48;
   for(unsigned i=0;i<96;i++){struct d12_g17p_txf_prefix_sources s=good;s.root_cell=limit-i;test(s,i>=8&&i%8==0);s=good;s.texture_table=limit-i;s.texture_table_size_B=32;test(s,i>=32&&i%32==0);s=good;s.sampler_table=limit-i;test(s,i>=32&&i%32==0);}
   struct d12_g17p_txf_prefix_sources s=good;s.root_cell=0;test(s,false);s=good;s.texture_cell=s.root_cell;test(s,false);s=good;s.sampler_cell=s.texture_cell;test(s,false);s=good;s.texture_table_size_B=UINT64_MAX;test(s,false);s=good;s.texture_table=UINT64_MAX-31;test(s,false);
   struct d12_g17p_txf_range out[4];memset(out,0xa5,sizeof(out));struct d12_g17p_txf_range before[4];memcpy(before,out,sizeof(out));CHECK(!d12_g17p_txf_prefix_ranges(NULL,out));CHECK(!memcmp(out,before,sizeof(out)));CHECK(!d12_g17p_txf_prefix_ranges(&good,NULL));
   printf("PASS %u actual prefix/range/boundary/transaction checks; NO_GPU\n",checks);return 0;
}
