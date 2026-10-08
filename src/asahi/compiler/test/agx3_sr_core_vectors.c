/* GENERATED -- do not edit. Clean SR-read rld witnesses (f23==7) from the
 * srops/memops/lsw corpora + their exact bytes; the C encoder must reproduce
 * each. Witnesses carrying the unattributed hint/marker fields (f11 on c ep1,
 * f40/f41 on b, f44/f45 on c) are excluded -- the driver emits those as 0. */
#include <stdint.h>
#include <stddef.h>

struct sr_vec {
  int enc, dest, width, sr, slot; unsigned wait;
  uint8_t nbytes; const uint8_t *bytes;
};

static const uint8_t sb0[]={4,168,16,6};
static const uint8_t sb1[]={20,169,16,38};
static const uint8_t sb2[]={36,170,16,70};
static const uint8_t sb3[]={4,130,16,6};
static const uint8_t sb4[]={28,160,16,6};
static const uint8_t sb5[]={60,161,16,6};
static const uint8_t sb6[]={44,162,16,6};
static const uint8_t sb7[]={4,130,16,6};
static const uint8_t sb8[]={20,133,16,38};
static const uint8_t sb9[]={4,130,16,6};
static const uint8_t sb10[]={4,133,16,6};
static const uint8_t sb11[]={36,152,16,6};
static const uint8_t sb12[]={4,153,16,6};
static const uint8_t sb13[]={20,154,16,38};
static const uint8_t sb14[]={100,168,16,6};
static const uint8_t sb15[]={52,169,16,38};
static const uint8_t sb16[]={4,170,16,6};
static const uint8_t sb17[]={12,156,16,6};
static const uint8_t sb18[]={28,157,16,6};
static const uint8_t sb19[]={44,158,16,6};
static const uint8_t sb20[]={4,167,16,6};
static const uint8_t sb21[]={12,160,16,6};
static const uint8_t sb22[]={28,161,16,6};
static const uint8_t sb23[]={44,162,16,6};
static const uint8_t sb24[]={4,164,16,6};
static const uint8_t sb25[]={20,165,16,38};
static const uint8_t sb26[]={36,166,16,70};
static const uint8_t sb27[]={4,152,16,6};
static const uint8_t sb28[]={20,153,16,38};
static const uint8_t sb29[]={36,154,16,70};
static const uint8_t sb30[]={4,168,16,6};
static const uint8_t sb31[]={20,169,16,6};
static const uint8_t sb32[]={36,170,16,6};
static const uint8_t sb33[]={4,130,16,6};
static const uint8_t sb34[]={28,160,16,6};
static const uint8_t sb35[]={60,161,16,6};
static const uint8_t sb36[]={44,162,16,6};
static const uint8_t sb37[]={4,130,16,6};
static const uint8_t sb38[]={20,133,16,38};
static const uint8_t sb39[]={4,130,16,6};
static const uint8_t sb40[]={4,133,16,6};
static const uint8_t sb41[]={4,152,20,6};
static const uint8_t sb42[]={4,153,16,6};
static const uint8_t sb43[]={20,154,16,38};
static const uint8_t sb44[]={100,168,16,6};
static const uint8_t sb45[]={52,169,16,38};
static const uint8_t sb46[]={4,170,16,70};
static const uint8_t sb47[]={12,156,16,6};
static const uint8_t sb48[]={28,157,16,6};
static const uint8_t sb49[]={44,158,16,6};
static const uint8_t sb50[]={4,167,16,6};
static const uint8_t sb51[]={12,160,16,6};
static const uint8_t sb52[]={28,161,16,6};
static const uint8_t sb53[]={44,162,16,6};
static const uint8_t sb54[]={4,164,16,6};
static const uint8_t sb55[]={20,165,16,6};
static const uint8_t sb56[]={36,166,16,6};
static const uint8_t sb57[]={4,152,16,6};
static const uint8_t sb58[]={20,153,16,6};
static const uint8_t sb59[]={36,154,16,6};
static const uint8_t sb60[]={28,160,16,6};
static const uint8_t sb61[]={28,160,16,6};
static const uint8_t sb62[]={28,160,16,6};
static const uint8_t sb63[]={44,160,16,6};
static const uint8_t sb64[]={44,160,16,6};
static const uint8_t sb65[]={28,160,16,6};
static const uint8_t sb66[]={28,160,16,6};
static const uint8_t sb67[]={28,160,16,6};
static const uint8_t sb68[]={28,160,16,6};
static const uint8_t sb69[]={4,167,16,6};
static const uint8_t sb70[]={28,160,16,6};
static const uint8_t sb71[]={44,160,16,6};
static const uint8_t sb72[]={28,160,16,6};
static const uint8_t sb73[]={12,160,16,6};
static const uint8_t sb74[]={12,160,16,6};
static const uint8_t sb75[]={140,160,16,6};
static const uint8_t sb76[]={12,160,16,6};
static const uint8_t sb77[]={44,160,16,6};
static const uint8_t sb78[]={28,160,16,6};
static const uint8_t sb79[]={28,160,16,6};
static const uint8_t sb80[]={12,160,16,6};
static const uint8_t sb81[]={28,160,16,6};
static const uint8_t sb82[]={44,160,16,6};
static const uint8_t sb83[]={12,160,16,6};
static const uint8_t sb84[]={28,160,16,6};
static const uint8_t sb85[]={44,160,16,6};
static const uint8_t sb86[]={60,160,16,6};
static const uint8_t sb87[]={76,160,16,6};
static const uint8_t sb88[]={76,160,16,6};
static const uint8_t sb89[]={92,160,16,6};
static const uint8_t sb90[]={28,160,16,6};
static const uint8_t sb91[]={28,160,16,6};
static const uint8_t sb92[]={36,167,20,6};
static const uint8_t sb93[]={12,160,16,6};
static const uint8_t sb94[]={36,167,16,6};
static const uint8_t sb95[]={28,160,16,6};
static const uint8_t sb96[]={28,160,16,6};
static const uint8_t sb97[]={36,149,16,6};
static const uint8_t sb98[]={4,151,16,6};
static const uint8_t sb99[]={92,160,16,38};
static const uint8_t sb100[]={4,149,16,6};
static const uint8_t sb101[]={4,151,16,6};
static const uint8_t sb102[]={12,160,17,6,0,32,0,0};
static const uint8_t sb103[]={28,160,16,6};
static const uint8_t sb104[]={36,164,16,6};
static const uint8_t sb105[]={28,160,16,6};
static const uint8_t sb106[]={36,164,16,6};
static const uint8_t sb107[]={28,160,16,6};
static const uint8_t sb108[]={36,164,16,6};
static const uint8_t sb109[]={28,160,16,6};
static const uint8_t sb110[]={36,164,16,6};
static const uint8_t sb111[]={28,160,16,6};
static const uint8_t sb112[]={36,164,16,6};
static const uint8_t sb113[]={28,160,16,6};
static const uint8_t sb114[]={36,164,16,6};
static const uint8_t sb115[]={60,160,16,6};
static const uint8_t sb116[]={28,160,16,6};
static const uint8_t sb117[]={60,160,16,6};
static const uint8_t sb118[]={140,160,81,6,0,0,0,16};
static const uint8_t sb119[]={140,160,144,6};
static const uint8_t sb120[]={44,160,16,6};
static const uint8_t sb121[]={12,161,16,38};
static const uint8_t sb122[]={44,160,16,6};
static const uint8_t sb123[]={12,161,16,38};
static const uint8_t sb124[]={44,160,16,6};
static const uint8_t sb125[]={28,161,16,6};
static const uint8_t sb126[]={28,160,16,6};
static const uint8_t sb127[]={68,164,16,38};
static const uint8_t sb128[]={4,164,16,6};
static const uint8_t sb129[]={12,160,16,6};
static const uint8_t sb130[]={68,149,16,6};
static const uint8_t sb131[]={4,164,16,6};
static const uint8_t sb132[]={12,160,16,6};
static const uint8_t sb133[]={36,149,16,6};
static const uint8_t sb134[]={28,160,16,6};
static const uint8_t sb135[]={68,164,16,38};
static const uint8_t sb136[]={28,160,16,6};
static const uint8_t sb137[]={116,164,16,166};
static const uint8_t sb138[]={28,160,16,6};
static const uint8_t sb139[]={52,164,20,38};
static const uint8_t sb140[]={44,160,16,6};
static const uint8_t sb141[]={84,164,16,6};
static const uint8_t sb142[]={12,160,16,6};
static const uint8_t sb143[]={36,164,20,6};
static const uint8_t sb144[]={28,160,16,6};
static const uint8_t sb145[]={36,164,20,6};
static const uint8_t sb146[]={76,160,16,6};
static const uint8_t sb147[]={84,164,20,6};
static const uint8_t sb148[]={28,160,16,6};
static const uint8_t sb149[]={68,164,16,6};
static const uint8_t sb150[]={28,160,16,6};
static const uint8_t sb151[]={28,160,16,6};
static const uint8_t sb152[]={28,160,16,6};
static const uint8_t sb153[]={28,160,16,6};
static const uint8_t sb154[]={12,160,16,6};
static const uint8_t sb155[]={28,160,16,6};
static const uint8_t sb156[]={28,160,16,6};
static const uint8_t sb157[]={12,160,16,6};
static const uint8_t sb158[]={28,160,16,6};
static const uint8_t sb159[]={28,160,16,6};
static const uint8_t sb160[]={28,160,16,6};
static const uint8_t sb161[]={28,160,16,6};
static const uint8_t sb162[]={28,160,16,6};
static const uint8_t sb163[]={28,160,16,6};
static const uint8_t sb164[]={28,160,16,6};
static const uint8_t sb165[]={28,160,16,6};
static const uint8_t sb166[]={28,160,16,6};
static const uint8_t sb167[]={28,160,16,6};
static const uint8_t sb168[]={28,160,16,6};
static const uint8_t sb169[]={28,160,16,6};
static const uint8_t sb170[]={28,160,16,6};
static const uint8_t sb171[]={12,160,16,6};
static const uint8_t sb172[]={28,160,16,6};
static const uint8_t sb173[]={28,160,16,6};
static const uint8_t sb174[]={4,168,16,6};
static const uint8_t sb175[]={20,169,16,6};
static const uint8_t sb176[]={28,160,16,6};
static const uint8_t sb177[]={60,161,16,6};
static const uint8_t sb178[]={44,162,16,6};
static const uint8_t sb179[]={4,130,16,6};
static const uint8_t sb180[]={20,133,16,38};
static const uint8_t sb181[]={4,130,16,6};
static const uint8_t sb182[]={4,152,20,6};
static const uint8_t sb183[]={4,153,16,6};
static const uint8_t sb184[]={20,154,16,38};
static const uint8_t sb185[]={12,156,16,6};
static const uint8_t sb186[]={28,157,16,6};
static const uint8_t sb187[]={44,158,16,6};
static const uint8_t sb188[]={12,160,16,6};
static const uint8_t sb189[]={28,161,16,6};
static const uint8_t sb190[]={44,162,16,6};
static const uint8_t sb191[]={4,164,16,6};
static const uint8_t sb192[]={20,165,16,6};
static const uint8_t sb193[]={4,152,16,6};
static const uint8_t sb194[]={20,153,16,6};
static const uint8_t sb195[]={28,160,16,6};
static const uint8_t sb196[]={28,160,16,6};
static const uint8_t sb197[]={28,160,16,6};
static const uint8_t sb198[]={44,160,16,6};
static const uint8_t sb199[]={44,160,16,6};
static const uint8_t sb200[]={28,160,16,6};
static const uint8_t sb201[]={28,160,16,6};
static const uint8_t sb202[]={28,160,16,6};
static const uint8_t sb203[]={28,160,16,6};
static const uint8_t sb204[]={28,160,16,6};
static const uint8_t sb205[]={44,160,16,6};
static const uint8_t sb206[]={28,160,16,6};
static const uint8_t sb207[]={12,160,16,6};
static const uint8_t sb208[]={12,160,16,6};
static const uint8_t sb209[]={140,160,16,6};
static const uint8_t sb210[]={12,160,16,6};
static const uint8_t sb211[]={44,160,16,6};
static const uint8_t sb212[]={28,160,16,6};
static const uint8_t sb213[]={28,160,16,6};
static const uint8_t sb214[]={12,160,16,6};
static const uint8_t sb215[]={28,160,16,6};
static const uint8_t sb216[]={44,160,16,6};
static const uint8_t sb217[]={12,160,16,6};
static const uint8_t sb218[]={28,160,16,6};
static const uint8_t sb219[]={44,160,16,6};
static const uint8_t sb220[]={60,160,16,6};
static const uint8_t sb221[]={76,160,16,6};
static const uint8_t sb222[]={76,160,16,6};
static const uint8_t sb223[]={92,160,16,6};
static const uint8_t sb224[]={28,160,16,6};
static const uint8_t sb225[]={28,160,16,6};
static const uint8_t sb226[]={36,167,20,6};
static const uint8_t sb227[]={12,160,16,6};
static const uint8_t sb228[]={36,167,16,6};
static const uint8_t sb229[]={28,160,16,6};
static const uint8_t sb230[]={28,160,16,6};
static const uint8_t sb231[]={36,149,16,6};
static const uint8_t sb232[]={92,160,16,38};
static const uint8_t sb233[]={4,149,16,6};
static const uint8_t sb234[]={36,132,1,26,32,0,160,2};
static const uint8_t sb235[]={28,160,16,6};
static const uint8_t sb236[]={36,164,16,6};
static const uint8_t sb237[]={28,160,16,6};
static const uint8_t sb238[]={36,164,16,6};
static const uint8_t sb239[]={28,160,16,6};
static const uint8_t sb240[]={36,164,16,6};
static const uint8_t sb241[]={28,160,16,6};
static const uint8_t sb242[]={36,164,16,6};
static const uint8_t sb243[]={28,160,16,6};
static const uint8_t sb244[]={36,164,16,6};
static const uint8_t sb245[]={28,160,16,6};
static const uint8_t sb246[]={36,164,16,6};
static const uint8_t sb247[]={12,160,16,6};
static const uint8_t sb248[]={12,160,16,6};
static const uint8_t sb249[]={28,160,16,6};
static const uint8_t sb250[]={60,160,16,6};
static const uint8_t sb251[]={164,128,9,26,32,0,160,34};
static const uint8_t sb252[]={140,160,144,6};
static const uint8_t sb253[]={44,160,16,6};
static const uint8_t sb254[]={12,161,16,38};
static const uint8_t sb255[]={44,160,16,6};
static const uint8_t sb256[]={12,161,16,38};
static const uint8_t sb257[]={44,160,16,6};
static const uint8_t sb258[]={28,161,16,6};
static const uint8_t sb259[]={28,160,16,6};
static const uint8_t sb260[]={68,164,16,38};
static const uint8_t sb261[]={12,160,16,6};
static const uint8_t sb262[]={68,149,16,6};
static const uint8_t sb263[]={4,164,16,6};
static const uint8_t sb264[]={12,160,16,6};
static const uint8_t sb265[]={36,149,16,6};
static const uint8_t sb266[]={28,160,16,6};
static const uint8_t sb267[]={68,164,16,38};
static const uint8_t sb268[]={28,160,16,6};
static const uint8_t sb269[]={116,164,16,230};
static const uint8_t sb270[]={28,160,16,6};
static const uint8_t sb271[]={52,164,20,38};
static const uint8_t sb272[]={44,160,16,6};
static const uint8_t sb273[]={84,164,16,6};
static const uint8_t sb274[]={12,160,16,6};
static const uint8_t sb275[]={36,164,20,6};
static const uint8_t sb276[]={28,160,16,6};
static const uint8_t sb277[]={36,164,20,6};
static const uint8_t sb278[]={76,160,16,6};
static const uint8_t sb279[]={84,164,20,6};
static const uint8_t sb280[]={28,160,16,6};
static const uint8_t sb281[]={68,164,16,6};
static const uint8_t sb282[]={28,160,16,6};
static const uint8_t sb283[]={28,160,16,6};
static const uint8_t sb284[]={28,160,16,6};
static const uint8_t sb285[]={28,160,16,6};
static const uint8_t sb286[]={12,160,16,6};
static const uint8_t sb287[]={28,160,16,6};
static const uint8_t sb288[]={28,160,16,6};
static const uint8_t sb289[]={12,160,16,6};
static const uint8_t sb290[]={28,160,16,6};
static const uint8_t sb291[]={28,160,16,6};
static const uint8_t sb292[]={28,160,16,6};
static const uint8_t sb293[]={28,160,16,6};
static const uint8_t sb294[]={28,160,16,6};
static const uint8_t sb295[]={28,160,16,6};
static const uint8_t sb296[]={28,160,16,6};
static const uint8_t sb297[]={28,160,16,6};
static const uint8_t sb298[]={28,160,16,6};
static const uint8_t sb299[]={28,160,16,6};
static const uint8_t sb300[]={28,160,16,6};
static const uint8_t sb301[]={28,160,16,6};
static const uint8_t sb302[]={28,160,16,6};
static const uint8_t sb303[]={12,160,16,6};
static const uint8_t sb304[]={28,160,16,6};
static const uint8_t sb305[]={28,160,16,6};

static const struct sr_vec sr_vecs[] = {
  {0,0,1,20,1,0u,4,sb0},
  {0,2,1,21,2,0u,4,sb1},
  {0,4,1,22,3,0u,4,sb2},
  {0,0,1,7,1,0u,4,sb3},
  {0,2,2,23,1,0u,4,sb4},
  {0,6,2,24,1,0u,4,sb5},
  {0,4,2,25,1,0u,4,sb6},
  {0,0,1,7,1,0u,4,sb7},
  {0,2,1,6,2,0u,4,sb8},
  {0,0,1,7,1,0u,4,sb9},
  {0,0,1,6,1,0u,4,sb10},
  {0,4,1,17,1,0u,4,sb11},
  {0,0,1,18,1,0u,4,sb12},
  {0,2,1,19,2,0u,4,sb13},
  {0,12,1,20,1,0u,4,sb14},
  {0,6,1,21,2,0u,4,sb15},
  {0,0,1,22,1,0u,4,sb16},
  {0,0,2,26,1,0u,4,sb17},
  {0,2,2,27,1,0u,4,sb18},
  {0,4,2,28,1,0u,4,sb19},
  {0,0,1,13,1,0u,4,sb20},
  {0,0,2,23,1,0u,4,sb21},
  {0,2,2,24,1,0u,4,sb22},
  {0,4,2,25,1,0u,4,sb23},
  {0,0,1,9,1,0u,4,sb24},
  {0,2,1,10,2,0u,4,sb25},
  {0,4,1,11,3,0u,4,sb26},
  {0,0,1,17,1,0u,4,sb27},
  {0,2,1,18,2,0u,4,sb28},
  {0,4,1,19,3,0u,4,sb29},
  {0,0,1,20,1,0u,4,sb30},
  {0,2,1,21,1,0u,4,sb31},
  {0,4,1,22,1,0u,4,sb32},
  {0,0,1,7,1,0u,4,sb33},
  {0,2,2,23,1,0u,4,sb34},
  {0,6,2,24,1,0u,4,sb35},
  {0,4,2,25,1,0u,4,sb36},
  {0,0,1,7,1,0u,4,sb37},
  {0,2,1,6,2,0u,4,sb38},
  {0,0,1,7,1,0u,4,sb39},
  {0,0,1,6,1,0u,4,sb40},
  {0,1,1,17,1,0u,4,sb41},
  {0,0,1,18,1,0u,4,sb42},
  {0,2,1,19,2,0u,4,sb43},
  {0,12,1,20,1,0u,4,sb44},
  {0,6,1,21,2,0u,4,sb45},
  {0,0,1,22,3,0u,4,sb46},
  {0,0,2,26,1,0u,4,sb47},
  {0,2,2,27,1,0u,4,sb48},
  {0,4,2,28,1,0u,4,sb49},
  {0,0,1,13,1,0u,4,sb50},
  {0,0,2,23,1,0u,4,sb51},
  {0,2,2,24,1,0u,4,sb52},
  {0,4,2,25,1,0u,4,sb53},
  {0,0,1,9,1,0u,4,sb54},
  {0,2,1,10,1,0u,4,sb55},
  {0,4,1,11,1,0u,4,sb56},
  {0,0,1,17,1,0u,4,sb57},
  {0,2,1,18,1,0u,4,sb58},
  {0,4,1,19,1,0u,4,sb59},
  {0,2,2,23,1,0u,4,sb60},
  {0,2,2,23,1,0u,4,sb61},
  {0,2,2,23,1,0u,4,sb62},
  {0,4,2,23,1,0u,4,sb63},
  {0,4,2,23,1,0u,4,sb64},
  {0,2,2,23,1,0u,4,sb65},
  {0,2,2,23,1,0u,4,sb66},
  {0,2,2,23,1,0u,4,sb67},
  {0,2,2,23,1,0u,4,sb68},
  {0,0,1,13,1,0u,4,sb69},
  {0,2,2,23,1,0u,4,sb70},
  {0,4,2,23,1,0u,4,sb71},
  {0,2,2,23,1,0u,4,sb72},
  {0,0,2,23,1,0u,4,sb73},
  {0,0,2,23,1,0u,4,sb74},
  {0,16,2,23,1,0u,4,sb75},
  {0,0,2,23,1,0u,4,sb76},
  {0,4,2,23,1,0u,4,sb77},
  {0,2,2,23,1,0u,4,sb78},
  {0,2,2,23,1,0u,4,sb79},
  {0,0,2,23,1,0u,4,sb80},
  {0,2,2,23,1,0u,4,sb81},
  {0,4,2,23,1,0u,4,sb82},
  {0,0,2,23,1,0u,4,sb83},
  {0,2,2,23,1,0u,4,sb84},
  {0,4,2,23,1,0u,4,sb85},
  {0,6,2,23,1,0u,4,sb86},
  {0,8,2,23,1,0u,4,sb87},
  {0,8,2,23,1,0u,4,sb88},
  {0,10,2,23,1,0u,4,sb89},
  {0,2,2,23,1,0u,4,sb90},
  {0,2,2,23,1,0u,4,sb91},
  {0,5,1,13,1,0u,4,sb92},
  {0,0,2,23,1,0u,4,sb93},
  {0,4,1,13,1,0u,4,sb94},
  {0,2,2,23,1,0u,4,sb95},
  {0,2,2,23,1,0u,4,sb96},
  {0,4,1,34,1,0u,4,sb97},
  {0,0,1,36,1,0u,4,sb98},
  {0,10,2,23,2,0u,4,sb99},
  {0,0,1,34,1,0u,4,sb100},
  {0,0,1,36,1,0u,4,sb101},
  {0,0,2,23,1,1u,8,sb102},
  {0,2,2,23,1,0u,4,sb103},
  {0,4,1,9,1,0u,4,sb104},
  {0,2,2,23,1,0u,4,sb105},
  {0,4,1,9,1,0u,4,sb106},
  {0,2,2,23,1,0u,4,sb107},
  {0,4,1,9,1,0u,4,sb108},
  {0,2,2,23,1,0u,4,sb109},
  {0,4,1,9,1,0u,4,sb110},
  {0,2,2,23,1,0u,4,sb111},
  {0,4,1,9,1,0u,4,sb112},
  {0,2,2,23,1,0u,4,sb113},
  {0,4,1,9,1,0u,4,sb114},
  {0,6,2,23,1,0u,4,sb115},
  {0,2,2,23,1,0u,4,sb116},
  {0,6,2,23,1,0u,4,sb117},
  {0,176,2,23,1,0u,8,sb118},
  {0,80,2,23,1,0u,4,sb119},
  {0,4,2,23,1,0u,4,sb120},
  {0,0,2,24,2,0u,4,sb121},
  {0,4,2,23,1,0u,4,sb122},
  {0,0,2,24,2,0u,4,sb123},
  {0,4,2,23,1,0u,4,sb124},
  {0,2,2,24,1,0u,4,sb125},
  {0,2,2,23,1,0u,4,sb126},
  {0,8,1,9,2,0u,4,sb127},
  {0,0,1,9,1,0u,4,sb128},
  {0,0,2,23,1,0u,4,sb129},
  {0,8,1,34,1,0u,4,sb130},
  {0,0,1,9,1,0u,4,sb131},
  {0,0,2,23,1,0u,4,sb132},
  {0,4,1,34,1,0u,4,sb133},
  {0,2,2,23,1,0u,4,sb134},
  {0,8,1,9,2,0u,4,sb135},
  {0,2,2,23,1,0u,4,sb136},
  {0,14,1,9,6,0u,4,sb137},
  {0,2,2,23,1,0u,4,sb138},
  {0,7,1,9,2,0u,4,sb139},
  {0,4,2,23,1,0u,4,sb140},
  {0,10,1,9,1,0u,4,sb141},
  {0,0,2,23,1,0u,4,sb142},
  {0,5,1,9,1,0u,4,sb143},
  {0,2,2,23,1,0u,4,sb144},
  {0,5,1,9,1,0u,4,sb145},
  {0,8,2,23,1,0u,4,sb146},
  {0,11,1,9,1,0u,4,sb147},
  {0,2,2,23,1,0u,4,sb148},
  {0,8,1,9,1,0u,4,sb149},
  {0,2,2,23,1,0u,4,sb150},
  {0,2,2,23,1,0u,4,sb151},
  {0,2,2,23,1,0u,4,sb152},
  {0,2,2,23,1,0u,4,sb153},
  {0,0,2,23,1,0u,4,sb154},
  {0,2,2,23,1,0u,4,sb155},
  {0,2,2,23,1,0u,4,sb156},
  {0,0,2,23,1,0u,4,sb157},
  {0,2,2,23,1,0u,4,sb158},
  {0,2,2,23,1,0u,4,sb159},
  {0,2,2,23,1,0u,4,sb160},
  {0,2,2,23,1,0u,4,sb161},
  {0,2,2,23,1,0u,4,sb162},
  {0,2,2,23,1,0u,4,sb163},
  {0,2,2,23,1,0u,4,sb164},
  {0,2,2,23,1,0u,4,sb165},
  {0,2,2,23,1,0u,4,sb166},
  {0,2,2,23,1,0u,4,sb167},
  {0,2,2,23,1,0u,4,sb168},
  {0,2,2,23,1,0u,4,sb169},
  {0,2,2,23,1,0u,4,sb170},
  {0,0,2,23,1,0u,4,sb171},
  {0,2,2,23,1,0u,4,sb172},
  {0,2,2,23,1,0u,4,sb173},
  {1,0,1,20,1,0u,4,sb174},
  {1,2,1,21,1,0u,4,sb175},
  {1,2,2,23,1,0u,4,sb176},
  {1,6,2,24,1,0u,4,sb177},
  {1,4,2,25,1,0u,4,sb178},
  {1,0,1,7,1,0u,4,sb179},
  {1,2,1,6,2,0u,4,sb180},
  {1,0,1,7,1,0u,4,sb181},
  {1,1,1,17,1,0u,4,sb182},
  {1,0,1,18,1,0u,4,sb183},
  {1,2,1,19,2,0u,4,sb184},
  {1,0,2,26,1,0u,4,sb185},
  {1,2,2,27,1,0u,4,sb186},
  {1,4,2,28,1,0u,4,sb187},
  {1,0,2,23,1,0u,4,sb188},
  {1,2,2,24,1,0u,4,sb189},
  {1,4,2,25,1,0u,4,sb190},
  {1,0,1,9,1,0u,4,sb191},
  {1,2,1,10,1,0u,4,sb192},
  {1,0,1,17,1,0u,4,sb193},
  {1,2,1,18,1,0u,4,sb194},
  {1,2,2,23,1,0u,4,sb195},
  {1,2,2,23,1,0u,4,sb196},
  {1,2,2,23,1,0u,4,sb197},
  {1,4,2,23,1,0u,4,sb198},
  {1,4,2,23,1,0u,4,sb199},
  {1,2,2,23,1,0u,4,sb200},
  {1,2,2,23,1,0u,4,sb201},
  {1,2,2,23,1,0u,4,sb202},
  {1,2,2,23,1,0u,4,sb203},
  {1,2,2,23,1,0u,4,sb204},
  {1,4,2,23,1,0u,4,sb205},
  {1,2,2,23,1,0u,4,sb206},
  {1,0,2,23,1,0u,4,sb207},
  {1,0,2,23,1,0u,4,sb208},
  {1,16,2,23,1,0u,4,sb209},
  {1,0,2,23,1,0u,4,sb210},
  {1,4,2,23,1,0u,4,sb211},
  {1,2,2,23,1,0u,4,sb212},
  {1,2,2,23,1,0u,4,sb213},
  {1,0,2,23,1,0u,4,sb214},
  {1,2,2,23,1,0u,4,sb215},
  {1,4,2,23,1,0u,4,sb216},
  {1,0,2,23,1,0u,4,sb217},
  {1,2,2,23,1,0u,4,sb218},
  {1,4,2,23,1,0u,4,sb219},
  {1,6,2,23,1,0u,4,sb220},
  {1,8,2,23,1,0u,4,sb221},
  {1,8,2,23,1,0u,4,sb222},
  {1,10,2,23,1,0u,4,sb223},
  {1,2,2,23,1,0u,4,sb224},
  {1,2,2,23,1,0u,4,sb225},
  {1,5,1,13,1,0u,4,sb226},
  {1,0,2,23,1,0u,4,sb227},
  {1,4,1,13,1,0u,4,sb228},
  {1,2,2,23,1,0u,4,sb229},
  {1,2,2,23,1,0u,4,sb230},
  {1,4,1,34,1,0u,4,sb231},
  {1,10,2,23,2,0u,4,sb232},
  {1,0,1,34,1,0u,4,sb233},
  {1,0,2,23,1,1u,8,sb234},
  {1,2,2,23,1,0u,4,sb235},
  {1,4,1,9,1,0u,4,sb236},
  {1,2,2,23,1,0u,4,sb237},
  {1,4,1,9,1,0u,4,sb238},
  {1,2,2,23,1,0u,4,sb239},
  {1,4,1,9,1,0u,4,sb240},
  {1,2,2,23,1,0u,4,sb241},
  {1,4,1,9,1,0u,4,sb242},
  {1,2,2,23,1,0u,4,sb243},
  {1,4,1,9,1,0u,4,sb244},
  {1,2,2,23,1,0u,4,sb245},
  {1,4,1,9,1,0u,4,sb246},
  {1,0,2,23,1,0u,4,sb247},
  {1,0,2,23,1,0u,4,sb248},
  {1,2,2,23,1,0u,4,sb249},
  {1,6,2,23,1,0u,4,sb250},
  {1,176,2,23,1,0u,8,sb251},
  {1,80,2,23,1,0u,4,sb252},
  {1,4,2,23,1,0u,4,sb253},
  {1,0,2,24,2,0u,4,sb254},
  {1,4,2,23,1,0u,4,sb255},
  {1,0,2,24,2,0u,4,sb256},
  {1,4,2,23,1,0u,4,sb257},
  {1,2,2,24,1,0u,4,sb258},
  {1,2,2,23,1,0u,4,sb259},
  {1,8,1,9,2,0u,4,sb260},
  {1,0,2,23,1,0u,4,sb261},
  {1,8,1,34,1,0u,4,sb262},
  {1,0,1,9,1,0u,4,sb263},
  {1,0,2,23,1,0u,4,sb264},
  {1,4,1,34,1,0u,4,sb265},
  {1,2,2,23,1,0u,4,sb266},
  {1,8,1,9,2,0u,4,sb267},
  {1,2,2,23,1,0u,4,sb268},
  {1,14,1,9,8,0u,4,sb269},
  {1,2,2,23,1,0u,4,sb270},
  {1,7,1,9,2,0u,4,sb271},
  {1,4,2,23,1,0u,4,sb272},
  {1,10,1,9,1,0u,4,sb273},
  {1,0,2,23,1,0u,4,sb274},
  {1,5,1,9,1,0u,4,sb275},
  {1,2,2,23,1,0u,4,sb276},
  {1,5,1,9,1,0u,4,sb277},
  {1,8,2,23,1,0u,4,sb278},
  {1,11,1,9,1,0u,4,sb279},
  {1,2,2,23,1,0u,4,sb280},
  {1,8,1,9,1,0u,4,sb281},
  {1,2,2,23,1,0u,4,sb282},
  {1,2,2,23,1,0u,4,sb283},
  {1,2,2,23,1,0u,4,sb284},
  {1,2,2,23,1,0u,4,sb285},
  {1,0,2,23,1,0u,4,sb286},
  {1,2,2,23,1,0u,4,sb287},
  {1,2,2,23,1,0u,4,sb288},
  {1,0,2,23,1,0u,4,sb289},
  {1,2,2,23,1,0u,4,sb290},
  {1,2,2,23,1,0u,4,sb291},
  {1,2,2,23,1,0u,4,sb292},
  {1,2,2,23,1,0u,4,sb293},
  {1,2,2,23,1,0u,4,sb294},
  {1,2,2,23,1,0u,4,sb295},
  {1,2,2,23,1,0u,4,sb296},
  {1,2,2,23,1,0u,4,sb297},
  {1,2,2,23,1,0u,4,sb298},
  {1,2,2,23,1,0u,4,sb299},
  {1,2,2,23,1,0u,4,sb300},
  {1,2,2,23,1,0u,4,sb301},
  {1,2,2,23,1,0u,4,sb302},
  {1,0,2,23,1,0u,4,sb303},
  {1,2,2,23,1,0u,4,sb304},
  {1,2,2,23,1,0u,4,sb305},
};
static const size_t sr_vec_count=306;
