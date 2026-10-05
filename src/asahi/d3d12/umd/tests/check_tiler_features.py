#!/usr/bin/env python3
"""Static gate for the four D3D12 features AGX implements natively.

Each of these is advertised through the shared capability policy, so the
number and the implementation have to stay attached: a cap that outlives its
code is a lie an application cannot detect until it renders wrong.
"""

from pathlib import Path


d12 = Path(__file__).resolve().parents[2]
caps = (d12 / "d12_core_caps.c").read_text()
caps_h = (d12 / "d12_core_caps.h").read_text()
device = (d12 / "d12_device.c").read_text() + (d12 / "d12_core_caps_query.c").read_text()
adapter = (d12 / "umd/d12_umd_adapter.c").read_text()
command = (d12 / "d12_cmd.c").read_text()
queue = (d12 / "d12_queue.c").read_text()
private = (d12 / "d12_private.h").read_text()
resource = (d12 / "d12_resource.c").read_text()
rp_core = (d12 / "d12_render_pass_core.h").read_text()
sp_core = (d12 / "d12_sample_position_core.h").read_text()
umd_rc = (d12 / "umd/d12_umd_resource_command_funcs.c").read_text()
umd_cmd = (d12 / "umd/d12_umd_command_funcs.c").read_text()
umd_backend = (d12 / "umd/d12_umd_backend.c").read_text()


# ---------------------------------------------------------------- render pass
# The per-attachment access types have to reach the tile programs, or the
# feature is a table the driver fills in and then ignores.
assert "caps->render_passes_tier = legacy ? 1 : 0;" in caps
assert "opts->RenderPassesTier =" in device
assert "D3D12DDI_RENDER_PASS_TIER_NOT_SUPPORTED" not in adapter
assert "caps->render_passes_tier" in adapter

for token in (
    "d12_render_pass_plan_for",
    "d12_render_pass_plan_legacy",
    "D12_RP_LOAD_NONE",
    "D12_RP_STORE_NONE",
):
    assert token in rp_core, token

# Colour: the plan selects the background and end-of-tile programs, and a
# discarded attachment gets neither.
assert "seg->render_pass ? seg->rt_plan[i]" in command
assert "main_bg[i] = AGX_BG_EOT_NONE;" in command
assert "main_eot[i] = plan.store == D12_RP_STORE_PRESERVE ? AGX_EOT_STORE" in command
# A tile that spills mid-pass must still be stored and reloaded whatever the
# pass asked for at its boundaries.
assert "partial_bg[i] = AGX_BG_LOAD;" in command
assert "partial_eot[i] = AGX_EOT_STORE;" in command

# Depth-stencil: the plan drives ZLS directly.
assert ".z_load = depth_plan.load == D12_RP_LOAD_PRESERVE," in queue
assert ".z_store = depth_plan.store == D12_RP_STORE_PRESERVE," in queue
assert "zls.s_load = stencil_plan.load == D12_RP_LOAD_PRESERVE;" in queue
assert "zls.s_store = stencil_plan.store == D12_RP_STORE_PRESERVE;" in queue

# Both frontends reach the same two helpers.
assert "d12_cmd_list_begin_render_pass" in command
assert "d12_cmd_list_end_render_pass" in command
assert ".BeginRenderPass = d12_list_BeginRenderPass," in command
assert ".EndRenderPass = d12_list_EndRenderPass," in command
assert "d12_umd_fill_render_pass_table" in umd_rc
assert "D3D12DDI_TABLE_TYPE_0043_RENDER_PASS" in adapter
# Both DDI revisions of the descriptor structures are handled.
assert "D3D12DDI_RENDER_PASS_FUNCS_0043" in umd_rc
assert "D3D12DDI_RENDER_PASS_FUNCS_0053" in umd_rc
# A resolve is refused rather than silently dropped.
assert "ENDING_ACCESS_RESOLVE" in command

# ----------------------------------------------------- programmable positions
assert "caps->programmable_sample_positions_tier = legacy ? 1 : 0;" in caps
assert "caps->programmable_sample_positions_tier" in device
# The command-list table only gained pfnSetSamplePositions in DDI 0032, so a
# WDDM 2.0/2.1 build must report the tier the ABI can actually carry rather
# than what the hardware can do.
assert "D12_UMD_PROGRAMMABLE_SAMPLE_POSITIONS_TIER" in adapter
assert "#if D12_UMD_TARGETS_WDDM23" in adapter

for token in ("d12_sample_positions_pack", "d12_sample_positions_standard",
              "d12_sample_positions_tier1_request_valid"):
    assert token in sp_core, token

# The defaults must come from the same encoder as a programmed pattern, so a
# change to one cannot silently disagree with the other.
assert "d12_sample_positions_standard(nr_samples, positions)" in queue
# The old hand-tabulated switch is gone; only the comment naming the
# expected values remains.
assert "case 1: return 0x88;" not in queue
assert "case 4: return 0xeaa26e26;" not in queue
# The programmed pattern actually reaches the register.
assert "c->ppp_multisamplectl = seg->sample_positions" in queue
assert "seg->sample_positions =" in command
assert ".SetSamplePositions = d12_list_SetSamplePositions," in command
assert "table->pfnSetSamplePositions = d12_umd_set_sample_positions;" in umd_cmd
assert "d12_umd_sample_positions_unsupported" not in umd_cmd

# ----------------------------------------------------------- GPU upload heaps
assert ".gpu_upload_heap = true," in caps
assert "opts->GPUUploadHeapSupported = caps->gpu_upload_heap;" in device
# Accepted, not refused outright, and reachable from the CPU.
assert "heap_props->Type == D3D12_HEAP_TYPE_GPU_UPLOAD &&" in resource
assert "!d12_core_get_caps()->gpu_upload_heap" in resource
# GetCustomHeapProperties has to resolve the type, or applications cannot use
# it through a CUSTOM heap.
assert "d12_device_GetCustomHeapProperties" in device
assert ".GetCustomHeapProperties = d12_device_GetCustomHeapProperties," in device
assert "D3D12_HEAP_TYPE_GPU_UPLOAD:" in device

# ------------------------------------------------------- resource heap tier 2
assert "caps->resource_heap_tier = 2;" in caps
# Tier 2 does not mean the DENY flags stop being honoured: an application that
# asks for a restriction still gets it.
assert "D3D12_HEAP_FLAG_DENY_BUFFERS" in resource
assert "D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES" in resource
assert "D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES" in resource

# ------------------------------------------------------------------- plumbing
for token in ("begin_render_pass", "end_render_pass", "set_sample_positions"):
    assert token in umd_backend, f"renderer op {token} is not forwarded"
assert "render_pass_active" in private
assert "sample_positions_set" in private

print(
    "tiler-features: PASS "
    "(render passes + programmable sample positions + GPU upload heaps "
    "+ resource heap tier 2)"
)
