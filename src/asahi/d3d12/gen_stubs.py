# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Generate crash-safe COM vtable defaults for d3d12agx.
#
# Our vtables are C designated initializers, so every method we do not set is
# NULL and an application calling it jumps to NULL and dies. That is fatal for
# running an external conformance suite: the first unimplemented call kills the
# process instead of producing a failure we can record.
#
# This emits, for each interface, a correctly-typed stub per method plus a
# D12_<IFACE>_DEFAULTS macro assigning every slot to its stub. A vtable then
# starts with that macro and overrides the entries it really implements
# (later designated initializers win; the build already passes
# -Wno-override-init).
#
# Stubs are properly typed on purpose. Casting one generic function to
# arbitrary signatures would be undefined behaviour and genuinely unsafe here,
# because several D3D12 methods return structs BY VALUE on non-Windows
# (GetDesc, GetGPUVirtualAddress, GetAdapterLuid...), which uses a different
# calling convention than an HRESULT return.

import re
import sys

INTERFACES = [
    "ID3D12Device",
    # Mesh shaders arrive through ID3D12Device2::CreatePipelineState (a
    # stream descriptor, not the fixed GRAPHICS/COMPUTE descs) and are
    # launched by ID3D12GraphicsCommandList6::DispatchMesh. Both vtables
    # are supersets of the base ones, so the objects keep one vtable each
    # and QueryInterface just accepts the newer IIDs.
    "ID3D12Device2",
    # The device answers every IID up to ID3D12Device15 from one vtable that
    # starts with the ID3D12Device2 table and appends the newer slots.
    "ID3D12Device15",
    "ID3D12GraphicsCommandList",
    "ID3D12GraphicsCommandList6",
    "ID3D12CommandQueue",
    "ID3D12Resource",
    "ID3D12Fence",
    "ID3D12DescriptorHeap",
    "ID3D12PipelineState",
    "ID3D12RootSignature",
    "ID3D12CommandAllocator",
]

# Annotations that appear in the SAL-decorated headers and mean nothing to us.
SAL = re.compile(
    r"\b(_In_|_Out_|_Inout_|_In_opt_|_Out_opt_|_Inout_opt_|_COM_Outptr_|"
    r"_COM_Outptr_opt_|_Outptr_|_Outptr_opt_|_Reserved_|_Null_|"
    r"_In_reads_bytes_\([^)]*\)|_In_reads_\([^)]*\)|_Out_writes_\([^)]*\)|"
    r"_Out_writes_bytes_\([^)]*\)|_In_reads_opt_\([^)]*\)|"
    r"_Out_writes_opt_\([^)]*\)|_Inout_updates_\([^)]*\)|"
    r"_Out_writes_bytes_opt_\([^)]*\)|_In_range_\([^)]*\)|"
    r"_Field_size_full_\([^)]*\)|_Always_\([^)]*\)|_Check_return_|"
    r"__RPC__[a-z_]*|_Post_[a-z_]*)\b"
)


def clean(text):
    text = SAL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def vtable_body(header, iface, windows=False):
    """Extract <iface>Vtbl for either the Windows or non-Windows ABI."""
    m = re.search(
        r"typedef\s+struct\s+" + iface + r"Vtbl\s*\{(.*?)\}\s*" + iface + r"Vtbl;",
        header,
        re.S,
    )
    if not m:
        return None

    body = m.group(1)

    # DirectX-Headers expresses aggregate returns with an explicit RetVal
    # pointer on Windows and by value elsewhere. Keep the requested branch.
    out = []
    depth_stack = []
    keep = True
    for line in body.split("\n"):
        s = line.strip()
        if s.startswith("#if"):
            non_win = "!defined(_WIN32)" in s or "!defined( _WIN32 )" in s
            keep = not non_win if windows else non_win
            depth_stack.append(keep)
            continue
        if s.startswith("#else"):
            if depth_stack:
                keep = not depth_stack[-1]
            continue
        if s.startswith("#endif"):
            if depth_stack:
                depth_stack.pop()
            keep = True
            continue
        if keep:
            out.append(line)

    return "\n".join(out)


METHOD = re.compile(
    r"(?P<ret>[A-Za-z_][A-Za-z0-9_ \t\*]*?)"
    r"\(\s*STDMETHODCALLTYPE\s*\*\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
    r"\s*\((?P<args>[^;]*)\);",
    re.S,
)


def parse_methods(body):
    methods = []
    for m in METHOD.finditer(body):
        ret = clean(m.group("ret"))
        name = m.group("name")
        args = clean(m.group("args"))
        methods.append((ret, name, args))
    return methods


def default_return(ret):
    """A safe value to return for a stub of this return type."""
    r = ret.replace(" ", "")
    if r == "void":
        return None
    if r == "HRESULT":
        return "E_NOTIMPL"
    if r in ("ULONG", "UINT", "UINT64", "DWORD", "SIZE_T", "LONG", "INT"):
        return "0"
    if r.endswith("*"):
        return "NULL"
    # Struct returned by value (LUID, D3D12_RESOURCE_DESC, handles, ...).
    return "ZERO"


def emit_stub(lines, iface, ret, name, args):
    fn = f"d12_stub_{iface}_{name}"
    val = default_return(ret)

    lines.append(f"static inline {ret}")
    lines.append(f"{fn}({args})")
    lines.append("{")
    lines.append(f'   d12_stub_report("{iface}::{name}");')

    for outp in re.findall(r"void \*\*(\w+)", args):
        lines.append(f"   if ({outp})")
        lines.append(f"      *{outp} = NULL;")

    if val is None:
        pass
    elif val == "ZERO":
        lines.append(f"   {ret} zero;")
        lines.append("   memset(&zero, 0, sizeof(zero));")
        lines.append("   return zero;")
    elif ret.rstrip().endswith("*") and re.search(r"\bRetVal\b", args):
        lines.append("   if (RetVal)")
        lines.append("      memset(RetVal, 0, sizeof(*RetVal));")
        lines.append("   return RetVal;")
    else:
        lines.append(f"   return {val};")
    lines.append("}")
    lines.append("")


def emit(header):
    lines = []
    lines.append("/*")
    lines.append(" * Copyright 2026 Aurora Silicon")
    lines.append(" * SPDX-License-Identifier: MIT")
    lines.append(" *")
    lines.append(" * GENERATED by gen_stubs.py -- do not edit by hand.")
    lines.append(" *")
    lines.append(" * Crash-safe defaults for every COM vtable slot. Unimplemented")
    lines.append(" * methods return a failure instead of being NULL, so an external")
    lines.append(" * conformance suite records a failure rather than dying.")
    lines.append(" */")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append("/* Each stub logs once so unimplemented calls are visible. */")
    lines.append("void d12_stub_report(const char *what);")
    lines.append("")

    total = 0
    for iface in INTERFACES:
        nonwin_body = vtable_body(header, iface, windows=False)
        win_body = vtable_body(header, iface, windows=True)
        if nonwin_body is None or win_body is None:
            print(f"WARNING: no vtable for {iface}", file=sys.stderr)
            continue

        methods = parse_methods(nonwin_body)
        win_methods = {name: (ret, args) for ret, name, args in parse_methods(win_body)}
        total += len(methods)
        lines.append(f"/* ---- {iface}: {len(methods)} methods ---- */")

        for ret, name, args in methods:
            win_ret, win_args = win_methods[name]
            if (win_ret, win_args) != (ret, args):
                lines.append("#ifdef _WIN32")
                emit_stub(lines, iface, win_ret, name, win_args)
                lines.append("#else")
                emit_stub(lines, iface, ret, name, args)
                lines.append("#endif")
                lines.append("")
            else:
                emit_stub(lines, iface, ret, name, args)

        lines.append(f"#define D12_{iface}_DEFAULTS \\")
        entries = [
            f"   .{name} = d12_stub_{iface}_{name}" for _, name, _ in methods
        ]
        lines.append(", \\\n".join(entries))
        lines.append("")

    print(f"generated stubs for {total} methods", file=sys.stderr)
    return "\n".join(lines) + "\n"


def main():
    header_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(header_path) as f:
        header = f.read()
    with open(out_path, "w") as f:
        f.write(emit(header))


if __name__ == "__main__":
    main()
