/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Supplementary registered-fd certification. Never creates a command queue. */
#include "d12_private.h"
#include "asahi/lib/agx3_code_container_core.h"

extern bool asahi_noop_d3d12_graphics_qualification_allowed(int fd)
   __attribute__((weak, visibility("default")));

static unsigned checks, failures;
static void
check(bool ok, const char *what)
{
   ++checks;
   if (!ok) {
      ++failures;
      fprintf(stderr, "FAIL: %s\n", what);
   }
}

static void *
read_shader(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
   long n = ftell(f);
   if (n <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
   void *bytes = malloc(n);
   if (!bytes || fread(bytes, 1, n, f) != (size_t)n) {
      free(bytes); fclose(f); return NULL;
   }
   fclose(f);
   *size = n;
   return bytes;
}

static bool
save(const char *name, const char *suffix, const void *bytes, size_t size)
{
   const char *dir = getenv("D3D12AGX_NATIVE_STAGE_ARTIFACTS");
   char path[4096];
   if (!dir || snprintf(path, sizeof(path), "%s/%s.%s", dir, name, suffix) >=
                  (int)sizeof(path))
      return false;
   FILE *f = fopen(path, "wb");
   if (!f)
      return false;
   bool ok = fwrite(bytes, 1, size, f) == size;
   return fclose(f) == 0 && ok;
}

static void
certify(const char *name, struct agx_bo *bo, uint64_t entry,
         const struct agx_shader_info *info,
         const struct d12_native_compute_abi *abi)
{
   check(bo && bo->va && abi->certified, "retained certified native CODE");
   if (!bo || !bo->va || !abi->certified)
      return;
   check(entry == bo->va->addr + AGX3_CONTAINER_LEAD_SIZE +
                                     AGX3_CODE_BLOCK_ENTRY_OFFSET,
         "entry is CODE entry, distinct from BO base");
   check(info->stage == MESA_SHADER_COMPUTE, "software stage is compute");
   void *mapping = agx_bo_map(bo);
   check(mapping && save(name, "CODE", mapping, bo->size), "CODE artifact");

   uint64_t cells[D12_NATIVE_COMPUTE_CELLS];
   for (unsigned slot = 0; slot < ARRAY_SIZE(cells); ++slot)
      cells[slot] = UINT64_C(0x10000000) + slot * 8;
   struct agx3_ioto_uniform_range ranges[D12_NATIVE_COMPUTE_RANGES];
   unsigned count = 0;
   check(d12_native_compute_ranges(info, abi, cells, ranges, &count),
         "declared ranges only");
   struct agx3_g17p_compute_program generated;
   bool generated_ok = d12_native_compute_generate(
      info, abi, entry, bo->va->addr, cells, &generated);
   check(generated_ok, "shared native Ioto");
   if (!generated_ok)
      return;
   check(save(name, "Ioto-synthetic-cells", generated.bytes, generated.length),
         "synthetic Ioto artifact, never executed");

   char text[4096];
   unsigned used = snprintf(text, sizeof(text),
      "{\"certified\":true,\"physical_admission\":false,"
      "\"required_cells\":%u,\"code_entry\":%llu,\"rodata_va\":%llu,"
      "\"ioto_bytes\":%u,\"synthetic_source_cells\":true,\"ranges\":[",
      abi->required_cells, (unsigned long long)entry,
      (unsigned long long)abi->rodata_va, generated.length);
   for (unsigned r = 0; r < count; ++r)
      used += snprintf(text + used, sizeof(text) - used, "%s[%u,%u,%llu]",
         r ? "," : "", ranges[r].start_halfs, ranges[r].size_halfs,
         (unsigned long long)ranges[r].address);
   used += snprintf(text + used, sizeof(text) - used, "]}\n");
   check(used < sizeof(text) && save(name, "abi.json", text, used), "ABI artifact");

   /* Missing and misaligned required cells must fail before changing output.
    * DRAW and VS_OUTPUTS are still eight-byte cells even for scalar reads. */
   for (unsigned slot = 0; slot < ARRAY_SIZE(cells); ++slot) {
      if (!(abi->required_cells & BITFIELD_BIT(slot)))
         continue;
      uint64_t value = cells[slot];
      for (unsigned bad = 0; bad < 2; ++bad) {
         cells[slot] = bad ? value + 1 : 0;
         memset(ranges, 0xa5, sizeof(ranges));
         struct agx3_ioto_uniform_range before[D12_NATIVE_COMPUTE_RANGES];
         memcpy(before, ranges, sizeof(before));
         count = 0xdead;
         check(!d12_native_compute_ranges(info, abi, cells, ranges, &count) &&
                  count == 0xdead && !memcmp(before, ranges, sizeof(before)),
               "bad cell refuses transactionally");
      }
      cells[slot] = value;
   }
   struct d12_native_compute_abi bad_abi = *abi;
   bad_abi.certified = false;
   check(!d12_native_compute_generate(info, &bad_abi, entry, bo->va->addr,
                                      cells, &generated), "uncertified program refuses");
   bad_abi = *abi;
   bad_abi.required_cells ^= BITFIELD_BIT(D12_UNIFORM_GEOM_PARAMS / 4);
   check(!d12_native_compute_generate(info, &bad_abi, entry, bo->va->addr,
                                      cells, &generated), "ABI mismatch refuses");
   struct agx_shader_info bad_info = *info;
   bad_info.agx3_uniform_halfs[0] |= BITFIELD64_BIT(8);
   uint32_t mask;
   check(!d12_native_compute_uniforms(&bad_info, &mask), "undeclared uniform refuses");
   bad_info = *info;
   bad_info.agx3_staging_known = false;
   check(!d12_native_compute_generate(&bad_info, abi, entry, bo->va->addr,
                                      cells, &generated), "unknown staging refuses");
   bad_info = *info;
   bad_info.has_preamble = true;
   check(!d12_native_compute_generate(&bad_info, abi, entry, bo->va->addr,
                                      cells, &generated), "CProg requirement refuses");
   printf("CERTIFIED %s: main=%u GPR=%u WG=%ux%ux%u uniform-cells=%#x\n",
          name, info->main_size, info->nr_gprs, info->workgroup_size[0],
          info->workgroup_size[1], info->workgroup_size[2], abi->required_cells);
}

int
main(int argc, char **argv)
{
   if (argc != 4 || !getenv("D3D12AGX_NATIVE_STAGE_ARTIFACTS")) {
      fprintf(stderr, "usage under registered G17P shim: %s vs.dxil ps.dxil gs.dxil\n", argv[0]);
      return 2;
   }
   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   if (hr != S_OK || !dev)
      return 1;
   struct d12_device *device = d12_device_from_iface(dev);
   if (device->dev.chip != AGX_CHIP_G17P ||
       !asahi_noop_d3d12_graphics_qualification_allowed ||
       !asahi_noop_d3d12_graphics_qualification_allowed(device->dev.fd)) {
      fprintf(stderr, "REFUSED: actual registered G17P noop fd required\n");
      ID3D12Device_Release(dev);
      return 2;
   }
   size_t size[3] = {0};
   void *code[3] = {0};
   for (unsigned i = 0; i < 3; ++i)
      code[i] = read_shader(argv[i + 1], &size[i]);
   ID3DBlob *blob = NULL;
   ID3D12RootSignature *root = NULL;
   ID3D12PipelineState *iface = NULL;
   const D3D12_ROOT_SIGNATURE_DESC rs = {
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
   bool complete = false;
   if (!code[0] || !code[1] || !code[2])
      goto done;
   hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, NULL);
   if (hr != S_OK || !blob)
      goto done;
   hr = ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(blob),
      ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&root);
   if (hr != S_OK || !root)
      goto done;
   D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {
      .pRootSignature = root,
      .VS = {.pShaderBytecode = code[0], .BytecodeLength = size[0]},
      .PS = {.pShaderBytecode = code[1], .BytecodeLength = size[1]},
      .GS = {.pShaderBytecode = code[2], .BytecodeLength = size[2]},
      .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID,
                         .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE},
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
      .NumRenderTargets = 1, .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1}, .SampleMask = ~0u,
   };
   desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   hr = ID3D12Device_CreateGraphicsPipelineState(dev, &desc,
      &IID_ID3D12PipelineState, (void **)&iface);
   printf("API_G6_PSO_CREATION: HRESULT=%#x object=%u NOT_SUITE_PARITY\n",
          (unsigned)hr, iface != NULL);
   if (hr != S_OK || !iface) {
      printf("COMPILER_DIAGNOSTIC: API refusal retained; registered no-submit stage entry\n");
      hr = d12_pipeline_state_certify_native_gs(device,
         &((struct d12_root_signature *)root)->map, code[0], size[0], code[2], size[2], &iface);
   }
   if (hr != S_OK || !iface)
      goto done;
   struct d12_pipeline_state *pso = d12_pipeline_state_from_iface(iface);
   check(pso->has_gs && pso->gs_info.shape == POLY_GS_SHAPE_STATIC_PER_INSTANCE,
         "frozen GS static-per-instance shape");
   certify("sw-vs", pso->sw_vs_bo, pso->sw_vs_va, &pso->sw_vs_info,
           &pso->native_stages[D12_NATIVE_SW_VS]);
   certify("gs-main", pso->gs_main_bo, pso->gs_main_va, &pso->gs_main_info,
           &pso->native_stages[D12_NATIVE_GS_MAIN]);
   check(pso->gs_rast.linked.bo && pso->gs_rast.info.agx3_staging_known &&
         !pso->gs_rast.info.has_preamble, "native raster companion retained");
   if (pso->gs_rast.linked.bo) {
      struct agx_bo *rast = pso->gs_rast.linked.bo;
      void *mapping = agx_bo_map(rast);
      check(mapping && save("gs-rast", "linked", mapping, rast->size),
            "linked raster artifact");
   }
   complete = true;
done:
   if (iface) ID3D12PipelineState_Release(iface);
   if (root) ID3D12RootSignature_Release(root);
   if (blob) ID3D10Blob_Release(blob);
   ID3D12Device_Release(dev);
   for (unsigned i = 0; i < 3; ++i) free(code[i]);
   printf("NATIVE STAGE CERTIFICATION: %s checks=%u failures=%u HRESULT=%#x "
          "REGISTERED_NO_QUEUE_NO_GPU_SUBMISSION_NOT_SUITE_PARITY\n",
          complete && !failures ? "PASS" : "FAIL", checks, failures, (unsigned)hr);
   return complete && !failures ? 0 : 1;
}
