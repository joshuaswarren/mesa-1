/* Compile the SCS forms used by the retained native M5 query runtime. */
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_m5_ray_query.h"
#include "agx3_scs_core.h"
#include "agx3_sr_core.h"
#include "agx3_pixel_core.h"
#include "agx3_graphics_program.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_file(const char *name,const void *data,size_t size)
{
 FILE *f=fopen(name,"wb");assert(f);assert(fwrite(data,1,size,f)==size);assert(!fclose(f));
}

/* mode0 direct, mode1 uniform indexed, mode2 register base, mode3 status. */
static void compile(unsigned mode,unsigned count,unsigned width,unsigned flags,
                    const char *dir)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"M5 query SCS forms");
 b.shader->info.workgroup_size[0]=64;
 b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
 nir_def *id=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
 nir_def *address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001cd00ull);
 nir_def *input=nir_load_global(&b,1,32,address,.align_mul=4);
 nir_def *expected=nir_iadd_imm(&b,nir_imul_imm(&b,input,3),5);
 nir_def *base=mode==1?nir_load_preamble(&b,1,64,.base=122):nir_ishl_imm(&b,id,4);
 enum pipe_format format=width==8?PIPE_FORMAT_R8_UINT:width==16?PIPE_FORMAT_R16_UINT:PIPE_FORMAT_R32_UINT;
 const unsigned result_bits=(width==32||(flags&2))?32:16;
 nir_def *components[4];
 for(unsigned c=0;c<count;c++) {
  nir_def *v=nir_iadd_imm(&b,expected,17*c);
  components[c]=width==32?v:nir_u2u16(&b,v);
 }
 nir_def *data=nir_vec(&b,components,count);
 if(mode==0)nir_push_if(&b,nir_ieq_imm(&b,id,0));
 nir_def *values;
 if(mode==0) {
  nir_store_scs_agx(&b,data,.base=0x6000);
  values=nir_load_scs_agx(&b,count,result_bits,.base=0x6000);
 } else if(mode==1) {
  nir_def *index=nir_ishl_imm(&b,id,4);
  nir_store_scs_uniform_agx(&b,data,base,index,.base=0x6000,.format=format,.component=0);
  values=nir_load_scs_uniform_agx(&b,count,result_bits,base,index,.base=0x6000,.format=format,.component=0,.flags=flags);
 } else {
  nir_store_scs_base_agx(&b,data,base,.base=0x6000,.format=format,.component=4);
  values=nir_load_scs_base_agx(&b,count,result_bits,base,.base=0x6000,.format=format,.component=4,.flags=flags);
 }
 if(getenv("M5_QUERY_RAW") && mode==1 && (flags&2)) {
  nir_def *raw_address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001ed00ull);
  nir_store_global(&b,nir_u2u32(&b,values),raw_address,.align_mul=4,.write_mask=1);
 }
 nir_def *valid=nir_imm_true(&b);
 for(unsigned c=0;c<count;c++) {
  nir_def *v=nir_u2u32(&b,nir_channel(&b,values,c));
  nir_def *e=nir_u2u32(&b,components[c]);
  if(width<32) {
   unsigned mask=(1u<<width)-1;
   e=nir_iand_imm(&b,e,mask);
   if(!(flags&2))v=nir_iand_imm(&b,v,mask);
  }
  valid=nir_iand(&b,valid,nir_ieq(&b,v,e));
 }
 nir_def *result=nir_bcsel(&b,valid,expected,nir_imm_int(&b,0xbad0bad0));
 if(mode==0) {
  nir_push_else(&b,NULL);
  nir_pop_if(&b,NULL);
  result=nir_if_phi(&b,result,expected);
 }
 address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001dd00ull);
 nir_store_global(&b,result,address,.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};
 bool ok=agx_compile_shader_nir(b.shader,&key,&part);
 printf("mode=%u count=%u width=%u flags=%u %s %s\n",mode,count,width,flags,ok?"PASS":"FAIL",ok?"":agx_last_compile_refusal());
 assert(ok&&part.info.uses_ray_traversal&&!part.info.ray_traversal_depth_nonzero);
 unsigned stores=0,loads=0;
 for(size_t off=part.info.main_offset;off<part.info.main_offset+part.info.main_size;off++)
  for(unsigned n=2;n<=16&&off+n<=part.info.main_offset+part.info.main_size;n+=2) {
   struct agx3_scs_desc d;bool load=agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d);
   bool store=!load&&agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d);
   if(!load&&!store)continue;
   assert(d.address_kind==(mode==0?AGX3_SCS_ADDRESS_DIRECT:mode==1?AGX3_SCS_ADDRESS_UNIFORM_INDEXED:AGX3_SCS_ADDRESS_REGISTER_BASE));
   assert(d.component_mask==((1u<<count)-1)&&d.byte_offset==0x6000);
   assert(d.width==(width==8?AGX3_MEM_W8:width==16?AGX3_MEM_W16:AGX3_MEM_W32));
   if(load)assert(d.zero_extend==!!(flags&2)&&d.c_load_marker==!!(flags&1)&&d.c_status_load==!!(flags&4));
   stores+=store;loads+=load;
  }
 assert(stores==1&&loads==1);
 if(dir) {
  char name[512];snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.bin",dir,mode,count,width,flags);
  write_file(name,part.binary+part.info.main_offset,part.info.main_size);
  struct agx3_ioto_uniform_range range={.start_halfs=122,.size_halfs=4,.address=0x10000090000ull};
  struct agx3_g17p_graphics_program_desc desc={.program_va=0x10000000000ull,.shader_va=0x10000003000ull,
   .uniform_ranges=mode==1?&range:NULL,.uniform_range_count=mode==1,
   .shader_binary_size=part.info.main_size,.nr_gprs=part.info.nr_gprs,
   .scratch_size=MAX2(part.info.scratch_size,part.info.agx3_spectpr_scratch_size),
   .compiler_d_valid=part.info.agx3_staging_known,.compiler_d=(uint32_t)part.info.agx3_staging_halfs*2};
  struct agx3_g17p_graphics_program program;assert(agx3_g17g_compute_program_generate(AGX_CHIP_G17S,&desc,&program));
  snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.program",dir,mode,count,width,flags);write_file(name,program.bytes,program.length);
  struct agx3_g17p_native_compute_program_desc ld={.program_va=desc.program_va,.shader_va=desc.shader_va};
  assert(agx3_cdm_direct_template_sources_compute(false,true,false,&ld.launch_sources));
  struct agx3_compute_program launch;assert(agx3_m5_native_compute_program_bind(AGX_CHIP_G17S,&ld,&launch));
  uint8_t cdm[48]={0};struct agx3_compute_dispatch_sink sink={.data=cdm,.capacity=44};
  const struct agx3_compute_direct_grid grid={.source=AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,.size={1024,1,1},.local_size={64,1,1},.dimensions=1};
  assert(agx3_m5_native_compute_dispatch_append(&launch,&grid,0,&sink));assert(sink.length==44);cdm[47]=0x40;
  snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.cdm",dir,mode,count,width,flags);write_file(name,cdm,48);
  printf("  bytes=%u gprs=%u loader=%u\n",part.info.main_size,part.info.nr_gprs,program.length);
 }
 free(part.binary);ralloc_free(b.shader);
}
/* Check the native candidate record layout across all ten records per lane.
 * This validates SCS transport, not hardware candidate production. */
static void candidate_packet(const char *dir)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"M5 candidate record transport");
 b.shader->info.workgroup_size[0]=64;
 b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
 nir_def *id=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
 nir_def *resident=nir_load_scs_thread_index_agx(&b);
 nir_def *lane=nir_iand_imm(&b,resident,31);
 nir_def *driver=nir_iadd_imm(&b,nir_imul_imm(&b,nir_iand_imm(&b,resident,~31u),272),0x5d8000);
 nir_def *input=nir_load_global(&b,1,32,nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001cd00ull),.align_mul=4);
 nir_def *expected=nir_iadd_imm(&b,nir_imul_imm(&b,input,3),5);
 nir_def *address=nir_iadd_imm(&b,nir_iadd(&b,driver,nir_imul_imm(&b,lane,200)),1856);
 nir_def *queue=nir_iadd_imm(&b,nir_iadd(&b,driver,nir_ishl_imm(&b,lane,2)),1728);
 nir_store_scs_base_agx(&b,nir_imm_int(&b,(10u<<21)|0x1fffff),queue,.base=0,.format=PIPE_FORMAT_R32_UINT,.component=4);
 for(unsigned i=0;i<10;i++) {
  nir_def *v=nir_iadd_imm(&b,nir_imul_imm(&b,id,17),i);
  nir_def *flags=nir_iand_imm(&b,nir_ixor(&b,nir_iadd_imm(&b,id,i),nir_imm_int(&b,0x5a00)),0xffff);
  nir_def *header=nir_vec2(&b,v,nir_ior_imm(&b,nir_ishl_imm(&b,flags,16),0xa123));
  nir_store_scs_base_agx(&b,header,address,.base=20*i,.format=PIPE_FORMAT_R32_UINT,.component=4);
  nir_store_scs_base_agx(&b,nir_vec2(&b,nir_iadd_imm(&b,v,100),nir_iadd_imm(&b,v,200)),address,.base=20*i+8,.format=PIPE_FORMAT_R32_UINT,.component=4);
  nir_store_scs_base_agx(&b,nir_iadd_imm(&b,v,300),address,.base=20*i+16,.format=PIPE_FORMAT_R32_UINT,.component=4);
 }
 nir_def *valid=nir_ieq_imm(&b,agx_nir_m5_candidate_count(&b,driver,lane),10);
 for(unsigned i=0;i<10;i++) {
  struct agx_m5_ray_candidate c=agx_nir_m5_load_candidate(&b,driver,lane,nir_imm_int(&b,i));
  nir_def *v=nir_iadd_imm(&b,nir_imul_imm(&b,id,17),i);
  nir_def *pointer=nir_pack_64_2x32_split(&b,v,nir_imm_int(&b,0xa123));
  nir_def *flags=nir_iand_imm(&b,nir_ixor(&b,nir_iadd_imm(&b,id,i),nir_imm_int(&b,0x5a00)),0xffff);
  valid=nir_iand(&b,valid,nir_ieq(&b,c.compressed_address,pointer));
  valid=nir_iand(&b,valid,nir_ieq(&b,c.flags,flags));
  valid=nir_iand(&b,valid,nir_ieq(&b,nir_channel(&b,c.barycentrics,0),nir_iadd_imm(&b,v,100)));
  valid=nir_iand(&b,valid,nir_ieq(&b,nir_channel(&b,c.barycentrics,1),nir_iadd_imm(&b,v,200)));
  valid=nir_iand(&b,valid,nir_ieq(&b,c.distance,nir_iadd_imm(&b,v,300)));
 }
 nir_def *result=nir_bcsel(&b,valid,expected,nir_imm_int(&b,0xbad0bad0));
 nir_store_global(&b,result,nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001dd00ull),.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};
 bool ok=agx_compile_shader_nir(b.shader,&key,&part);
 printf("M5_CANDIDATE_PACKET %s %s\n",ok?"PASS":"FAIL",ok?"":agx_last_compile_refusal());
 assert(ok&&part.info.uses_ray_traversal&&!part.info.ray_traversal_depth_nonzero);
 if(dir) {
  char name[512];snprintf(name,sizeof(name),"%s/candidate-packet.bin",dir);
  write_file(name,part.binary+part.info.main_offset,part.info.main_size);
  struct agx3_g17p_graphics_program_desc desc={.program_va=0x10000000000ull,.shader_va=0x10000002000ull,
   .uniform_ranges=NULL,.uniform_range_count=0,
   .shader_binary_size=part.info.main_size,.nr_gprs=part.info.nr_gprs,
   .scratch_size=MAX2(part.info.scratch_size,part.info.agx3_spectpr_scratch_size),
   .compiler_d_valid=part.info.agx3_staging_known,.compiler_d=(uint32_t)part.info.agx3_staging_halfs*2};
  struct agx3_g17p_graphics_program program;assert(agx3_g17g_compute_program_generate(AGX_CHIP_G17S,&desc,&program));
  snprintf(name,sizeof(name),"%s/candidate-packet.program",dir);write_file(name,program.bytes,program.length);
  struct agx3_g17p_native_compute_program_desc ld={.program_va=desc.program_va,.shader_va=desc.shader_va};
  assert(agx3_cdm_direct_template_sources_compute(false,true,false,&ld.launch_sources));
  struct agx3_compute_program launch;assert(agx3_m5_native_compute_program_bind(AGX_CHIP_G17S,&ld,&launch));
  uint8_t cdm[48]={0};struct agx3_compute_dispatch_sink sink={.data=cdm,.capacity=44};
  const struct agx3_compute_direct_grid grid={.source=AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,.size={1024,1,1},.local_size={64,1,1},.dimensions=1};
  assert(agx3_m5_native_compute_dispatch_append(&launch,&grid,0,&sink));assert(sink.length==44);cdm[47]=0x40;
  snprintf(name,sizeof(name),"%s/candidate-packet.cdm",dir);write_file(name,cdm,48);
  printf("  bytes=%u gprs=%u loader=%u\n",part.info.main_size,part.info.nr_gprs,program.length);
 }
 free(part.binary);ralloc_free(b.shader);
}
static void refused(unsigned format,unsigned flags,unsigned selector,int offset)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"invalid SCS form");
 nir_def *v=nir_load_scs_base_agx(&b,1,16,nir_imm_int(&b,0),.base=offset,
                                .format=format,.flags=flags,.component=selector);
 nir_store_global(&b,v,nir_imm_int64(&b,0x10000000000ull),.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};
 assert(!agx_compile_shader_nir(b.shader,&key,&part)&&!part.binary);
 ralloc_free(b.shader);
}

static void
standard_ray_query(unsigned count, bool array, mesa_shader_stage stage)
{
   const bool kernel = stage != MESA_SHADER_COMPUTE;
   const char *value = getenv("ASAHI_M5_RQ_CANDIDATE_API");
   char *saved = value ? strdup(value) : NULL;
   if (kernel)
      setenv("ASAHI_M5_RQ_CANDIDATE_API", "1", 1);
   nir_builder b = nir_builder_init_simple_shader(
      stage, &agx_nir_options, "M5 standard ray query");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = count;
   nir_variable *queries[16];
   nir_def *rqs[16];
   assert(count && count <= 16);
   nir_variable *query_array = array ? nir_local_variable_create(
      b.impl, glsl_array_type(glsl_uint64_t_type(), count, 0), "queries") : NULL;
   if (query_array)
      query_array->data.ray_query = true;
   for (unsigned i = 0; i < count; ++i) {
      nir_variable *query = array ? query_array :
         nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
      query->data.ray_query = true;
      queries[i] = query;
      nir_deref_instr *deref = nir_build_deref_var(&b, query);
      if (array) {
         nir_def *index = nir_iand_imm(&b,
            nir_iadd_imm(&b, nir_load_local_invocation_index(&b), i), count - 1);
         deref = nir_build_deref_array(&b, deref, index);
      }
      nir_def *rq = &deref->def;
      rqs[i] = rq;
      nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10002000000ull),
                        nir_imm_int(&b, 1), nir_imm_int(&b, 255),
                        nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
                        nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   }
   for (unsigned i = count; i-- > 0;) {
      nir_push_loop(&b);
      nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rqs[i])));
      nir_pop_loop(&b, NULL);
   }
   nir_def *rq = rqs[0];
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type, .committed = true);
   nir_def *distance = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_t, .committed = true);
   nir_store_global(&b, nir_vec2(&b, type, distance),
                     nir_imm_int64(&b, 0x10003010000ull),
                     .align_mul = 8, .write_mask = 3);
   const struct agx_m5_ray_query_layout layout = {
      .payload_uniform = kernel ? 134 : 122,
      .bvh_uniform = kernel ? 100 : 96, .status_uniform = 134,
      .launch_header = 0x10000000,
      .context_uniform = kernel ? 160 : 0,
      .as_table_uniform = kernel ? 164 : 0,
      .instance_traversal = kernel,
   };
   struct agx_m5_ray_query_layout bad_layout = layout;
   bad_layout.payload_uniform++;
   assert(!agx_nir_lower_m5_ray_query(b.shader, &bad_layout));
   assert(queries[0]->data.ray_query && b.shader->info.ray_queries == count);
   struct agx_m5_ray_query_layout kernel_layout = layout;
   if (stage != MESA_SHADER_COMPUTE) {
      assert(!agx_nir_lower_m5_ray_query(b.shader, &layout));
      assert(queries[0]->data.ray_query && b.shader->info.ray_queries == count);
      kernel_layout.compute_kernel = true;
   }
   assert(agx_nir_lower_m5_ray_query(b.shader, &kernel_layout));
   assert(!b.shader->info.ray_queries);
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic)
            assert(!nir_intrinsic_is_ray_query(nir_instr_as_intrinsic(instr)->intrinsic));
      }
   }
   b.shader->info.stage = stage == MESA_SHADER_FRAGMENT ? stage : MESA_SHADER_COMPUTE;
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17S, .encoding = AGX_ISA_ENCODING_AGX3C},
      .no_preamble = true,
      .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled) fprintf(stderr, "Query compile refusal: %s\n", agx_last_compile_refusal());
   assert(compiled);
   assert(part.info.uses_ray_traversal && part.info.main_size);
   if (stage == MESA_SHADER_FRAGMENT) {
      const uint8_t acquire[] = {0x27, 0x00, 0x00, 0x05, 0x00, 0x00};
      unsigned acquires = 0;
      for (size_t i = part.info.main_offset;
           i + sizeof(acquire) <= part.info.main_offset + part.info.main_size; ++i)
         acquires += !memcmp(part.binary + i, acquire, sizeof(acquire));
      assert(acquires == 1);
      assert(!part.info.agx3_colour_link.rt_query_write_acquired);
      const uint8_t release[] = {0x27, 0x00, 0x00, 0x04, 0x00, 0x00};
      unsigned releases = 0;
      for (size_t i = part.info.main_offset;
           i + sizeof(release) <= part.info.main_offset + part.info.main_size; ++i)
         releases += !memcmp(part.binary + i, release, sizeof(release));
      assert(releases == 1);
   }
   free(part.binary);
   ralloc_free(b.shader);
   if (saved) {
      setenv("ASAHI_M5_RQ_CANDIDATE_API", saved, 1);
      free(saved);
   } else {
      unsetenv("ASAHI_M5_RQ_CANDIDATE_API");
   }
   printf("M5_STANDARD_RAY_QUERY_PASS count=%u array=%u stage=%u\n", count, array, stage);
}

static unsigned
native_query_ops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_enter_query_agx:
            case nir_intrinsic_intersect_ray_agx:
            case nir_intrinsic_continue_ray_agx:
            case nir_intrinsic_load_scs_indexed_agx:
            case nir_intrinsic_store_scs_indexed_agx:
            case nir_intrinsic_load_scs_uniform_agx:
            case nir_intrinsic_store_scs_uniform_agx:
            case nir_intrinsic_load_scs_base_agx:
            case nir_intrinsic_store_scs_base_agx:
            case nir_intrinsic_load_scs_register_agx:
            case nir_intrinsic_store_scs_agx:
               n++;
               break;
            default:
               break;
            }
         }
      }
   }
   return n;
}

/* Geometry lowering clones a lowered query into kernels whose API stores it
 * strips. Such a query is removed; one whose result is stored is kept. */
static void
dead_ray_query(bool live)
{
   const char *value = getenv("ASAHI_M5_RQ_CANDIDATE_API");
   char *saved = value ? strdup(value) : NULL;
   setenv("ASAHI_M5_RQ_CANDIDATE_API", "1", 1);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "M5 dead ray query");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = 1;
   nir_variable *query =
      nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   /* A live ID-dependent value before the query, as poly's index write. */
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10002000000ull),
                     nir_imm_int(&b, 1), nir_imm_int(&b, 255),
                     nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
                     nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_pop_loop(&b, NULL);
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type, .committed = true);
   nir_intrinsic_instr *result = nir_store_global(&b, type,
      nir_imm_int64(&b, 0x10003010000ull), .align_mul = 4, .write_mask = 1);
   nir_store_global(&b, id, nir_imm_int64(&b, 0x10003020000ull),
                    .align_mul = 4, .write_mask = 1);
   const struct agx_m5_ray_query_layout layout = {
      .payload_uniform = 134, .bvh_uniform = 100, .status_uniform = 134,
      .launch_header = 0x10000000, .context_uniform = 160,
      .as_table_uniform = 164, .instance_traversal = true,
      .compute_kernel = true,
   };
   assert(agx_nir_lower_m5_ray_query(b.shader, &layout));
   assert(native_query_ops(b.shader));
   if (!live)
      nir_instr_remove(&result->instr);
   const unsigned before = native_query_ops(b.shader);
   const bool progress = agx_nir_m5_remove_dead_ray_queries(b.shader);
   assert(progress || live);
   assert(live ? native_query_ops(b.shader) == before
               : !native_query_ops(b.shader));
   unsigned stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         stores += instr->type == nir_instr_type_intrinsic &&
            nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global;
      }
   }
   assert(stores == (live ? 2 : 1));
   b.shader->info.stage = MESA_SHADER_COMPUTE;
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17S, .encoding = AGX_ISA_ENCODING_AGX3C},
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled) fprintf(stderr, "Dead query compile refusal: %s\n", agx_last_compile_refusal());
   assert(compiled);
   assert(part.info.uses_ray_traversal == live && part.info.main_size);
   free(part.binary);
   ralloc_free(b.shader);
   if (saved) {
      setenv("ASAHI_M5_RQ_CANDIDATE_API", saved, 1);
      free(saved);
   } else {
      unsetenv("ASAHI_M5_RQ_CANDIDATE_API");
   }
   printf("M5_DEAD_RAY_QUERY_PASS live=%u\n", live);
}

/* Which break left a query-selected loop is a query result: the phis after
 * the loop merge one value per break, here constants. A store of such a phi
 * keeps the query, as rayQueryProceedEXT's "found" flag after the traversal
 * loop does in ray_query.builtin.getworldrayorigin.geom. */
static void
dead_ray_query_loop_exit(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "M5 query loop exit");
   nir_variable *found =
      nir_local_variable_create(b.impl, glsl_bool_type(), "found");
   /* Defined before the loop, as the lowering's hoisted constants are. */
   nir_def *yes = nir_imm_true(&b), *no = nir_imm_false(&b);
   nir_push_loop(&b);
   {
      nir_def *q = nir_load_scs_thread_index_agx(&b);
      nir_push_if(&b, nir_ieq_imm(&b, q, 0));
      {
         nir_store_var(&b, found, yes, 1);
         nir_jump(&b, nir_jump_break);
      }
      nir_pop_if(&b, NULL);
      nir_store_var(&b, found, no, 1);
      nir_jump(&b, nir_jump_break);
   }
   nir_pop_loop(&b, NULL);
   nir_store_global(&b, nir_b2i32(&b, nir_load_var(&b, found)),
                    nir_imm_int64(&b, 0x10003010000ull), .align_mul = 4,
                    .write_mask = 1);
   nir_lower_vars_to_ssa(b.shader);
   agx_nir_m5_remove_dead_ray_queries(b.shader);
   unsigned queries = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         queries += instr->type == nir_instr_type_intrinsic &&
            nir_instr_as_intrinsic(instr)->intrinsic ==
               nir_intrinsic_load_scs_thread_index_agx;
      }
   }
   ralloc_free(b.shader);
   if (queries != 1) {
      fprintf(stderr, "M5 query feeding a loop-exit phi was removed\n");
      exit(1);
   }
   printf("M5_DEAD_RAY_QUERY_LOOP_EXIT_PASS\n");
}

/* Native instance completion is a byte read, not a word read and shift. */
static void
instance_completion_byte(void)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"instance completion byte");
 nir_def *base=nir_load_global(&b,1,32,nir_imm_int64(&b,0x10000),.align_mul=4);
 nir_def *lane=nir_load_global(&b,1,32,nir_imm_int64(&b,0x10004),.align_mul=4);
 nir_def *v=nir_load_scs_register_agx(&b,1,16,base,lane,
    .base=263,.format=PIPE_FORMAT_R8_UINT,.component=3,.flags=9);
 nir_store_global(&b,nir_u2u32(&b,v),nir_imm_int64(&b,0x20000),.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};assert(agx_compile_shader_nir(b.shader,&key,&part));
 unsigned found=0;
 for(size_t off=part.info.main_offset;off<part.info.main_offset+part.info.main_size;off++)
  for(unsigned n=2;n<=16&&off+n<=part.info.main_offset+part.info.main_size;n+=2){
   struct agx3_scs_desc d;if(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d))continue;
   assert(d.address_kind==AGX3_SCS_ADDRESS_REGISTER_INDEXED&&d.width==AGX3_MEM_W8&&
          d.byte_offset==263&&d.address_selector==3&&d.component_mask==1&&d.c_load_marker&&(d.wait_mask&0x80));
   found++;
  }
 assert(found==1);free(part.binary);ralloc_free(b.shader);
 puts("M5_INSTANCE_COMPLETION_BYTE_PASS");
}

/* Directly reproduce the native entry's SR59/W16 read (rld h28). */
static void
scs_thread_identity(void)
{
 struct agx3_sr_desc d={.dest=28,.sr=59,.width=AGX3_SR_W16,.slot=3};
 const uint8_t native[]={0xe4,0xea,0x10,0x46};uint8_t out[16],len=0;
 assert(agx3_sr_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
 assert(len==sizeof(native)&&!memcmp(out,native,len));
 assert(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
 puts("M5_NATIVE_SCS_THREAD_ID_BYTES_PASS");
}

/* A completed singleton batch owns its fields until continuation. Check the
 * lowering's actual NIR guards, resets and completion operations, rather than
 * using fabricated SCS state. The hardware controls supply real candidates. */
static bool
inside_batch_completion_guard(nir_instr *instr)
{
   for (nir_cf_node *node = instr->block->cf_node.parent; node; node = node->parent) {
      /* The query activation guard outside the batch loop is not a latch. */
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type != nir_cf_node_if)
         continue;
      nir_def *cond = nir_cf_node_as_if(node)->condition.ssa;
      if (nir_def_instr(cond)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
      if (alu->op == nir_op_inot && alu->src[0].src.ssa->bit_size == 1) {
         nir_def *input = alu->src[0].src.ssa;
         while (nir_def_instr(input)->type == nir_instr_type_alu &&
                nir_instr_as_alu(nir_def_instr(input))->op == nir_op_mov)
            input = nir_instr_as_alu(nir_def_instr(input))->src[0].src.ssa;
         if (nir_def_instr(input)->type == nir_instr_type_phi)
            return true;
      }
   }
   return false;
}

static void
batch_completion_latch(unsigned count)
{
   const char *option = getenv("ASAHI_M5_RQ_CANDIDATE_API");
   char *saved = option ? strdup(option) : NULL;
   setenv("ASAHI_M5_RQ_CANDIDATE_API", "1", 1);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "M5 completed batch ownership");
   b.shader->info.ray_queries = count;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_variable *queries[2];
   for (unsigned q = 0; q < count; ++q) {
      queries[q] = nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
      queries[q]->data.ray_query = true;
      for (unsigned epoch = 0; epoch < 2; ++epoch) {
         nir_def *rq = &nir_build_deref_var(&b, queries[q])->def;
         nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10002000000ull),
            nir_imm_int(&b, epoch ? 2 : 1), nir_imm_int(&b, 255),
            nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0.001),
            nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
         nir_push_loop(&b);
         nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
         nir_rq_confirm_intersection(&b, rq);
         nir_pop_loop(&b, NULL);
      }
   }
   const struct agx_m5_ray_query_layout layout = {
      .payload_uniform = 152, .bvh_uniform = 96, .status_uniform = 134,
      .launch_header = 0x10000000, .instance_traversal = true,
   };
   assert(agx_nir_lower_m5_ray_query(b.shader, &layout));
   unsigned waited = 0, guarded_waits = 0, raw_status = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_scs_base_agx &&
             nir_intrinsic_base(intr) == 7) {
            if (nir_intrinsic_flags(intr) == 8) {
               guarded_waits += inside_batch_completion_guard(instr);
               waited++;
            } else if (nir_intrinsic_flags(intr) == 0) raw_status++;
         }
      }
   }
   fprintf(stderr, "BATCH_COUNTS count=%u waited=%u guarded=%u raw=%u\n", count, waited, guarded_waits, raw_status);
   assert((count == 1 ? waited == 2 : waited >= 4 * count) &&
          guarded_waits == (count == 1 ? 2 : 0) &&
          raw_status == (count == 1 ? 2 : 0));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17S, .encoding = AGX_ISA_ENCODING_AGX3C},
      .no_preamble = true,
      .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(b.shader, &key, &part));
   assert(part.info.uses_ray_traversal && part.info.main_size);
   free(part.binary);
   ralloc_free(b.shader);
   if (saved) { setenv("ASAHI_M5_RQ_CANDIDATE_API", saved, 1); free(saved); }
   else unsetenv("ASAHI_M5_RQ_CANDIDATE_API");
   printf("M5_BATCH_COMPLETION_LATCH_PASS count=%u\n", count);
}

int main(int argc,char **argv)
{
 setbuf(stdout,NULL);glsl_type_singleton_init_or_ref();const char *out=argc>1?argv[1]:NULL;
 candidate_packet(out);
 scs_thread_identity();
 instance_completion_byte();
 batch_completion_latch(1);
 batch_completion_latch(2);
 standard_ray_query(1, false, MESA_SHADER_COMPUTE);
 standard_ray_query(1, false, MESA_SHADER_TESS_CTRL);
 dead_ray_query(false);
 dead_ray_query(true);
 dead_ray_query_loop_exit();
 standard_ray_query(1, false, MESA_SHADER_TESS_EVAL);
 standard_ray_query(1, false, MESA_SHADER_GEOMETRY);
 standard_ray_query(1, false, MESA_SHADER_VERTEX);
 standard_ray_query(1, false, MESA_SHADER_FRAGMENT);
 standard_ray_query(2, false, MESA_SHADER_COMPUTE);
 standard_ray_query(8, false, MESA_SHADER_COMPUTE);
 standard_ray_query(4, true, MESA_SHADER_COMPUTE);
 standard_ray_query(8, true, MESA_SHADER_COMPUTE);
 /* Candidate-API path (kernel stages): more logical queries than native
  * slots, and dynamically selected query-array elements. */
 standard_ray_query(9, false, MESA_SHADER_TESS_CTRL);
 standard_ray_query(12, false, MESA_SHADER_TESS_CTRL);
 standard_ray_query(2, true, MESA_SHADER_TESS_CTRL);
 standard_ray_query(8, true, MESA_SHADER_TESS_CTRL);
 standard_ray_query(16, true, MESA_SHADER_TESS_CTRL);
 for(unsigned mode=0;mode<3;mode++)for(unsigned n=1;n<=4;n++)compile(mode,n,32,0,out);
 compile(0,1,16,0,out);
 for(unsigned mode=1;mode<=2;mode++)for(unsigned width=8;width<=16;width+=8)compile(mode,1,width,0,out);
 for(unsigned width=8;width<=16;width+=8)compile(1,1,width,2,out);
 /* The status conversion is unknown: exact encoding only, never run using
  * fabricated query state. */
 compile(3,1,8,5,NULL);
 refused(PIPE_FORMAT_R8_UINT,4,4,0);  /* status requires marker */
 refused(PIPE_FORMAT_R16_UINT,5,4,0); /* status is byte-only */
 refused(PIPE_FORMAT_R16_UINT,0,5,0); /* invalid selector */
 refused(PIPE_FORMAT_R16_UINT,0,4,1); /* unaligned access */
 refused(PIPE_FORMAT_R32_FLOAT,0,4,0); /* unsupported memory format */
 printf("Invalid SCS forms rejected transactionally\n");
 glsl_type_singleton_decref();return 0;
}
