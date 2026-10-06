# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Generate the D3D12AGX_DEBUG=api tracing vtables for d3d12agx.
#
# Every COM object the driver hands out owns exactly one real vtable. With
# the trace enabled, the object creation sites install a generated vtable
# instead whose methods log the call (name, scalar/pointer arguments) and,
# for HRESULT-returning methods, the result, then forward to the real
# implementation. Nothing in the driver proper changes shape: the real
# vtables stay as they are and the trace is one pointer swap at creation.
#
# Wrappers are properly typed per method, like gen_stubs.py's stubs, because
# several methods return aggregates by value on non-Windows ABIs.
#
# Usage: gen_api_trace.py <DirectX-Headers d3d12.h> <out d12_api_trace.h>

import re
import sys

from gen_stubs import clean, parse_methods, vtable_body

# The interface version each d3d12agx object exposes through its vtable.
INTERFACES = [
    ("DEVICE", "ID3D12Device15"),
    ("LIST", "ID3D12GraphicsCommandList9"),
    ("QUEUE", "ID3D12CommandQueue"),
    ("RESOURCE", "ID3D12Resource"),
    ("FENCE", "ID3D12Fence"),
    ("DESCRIPTOR_HEAP", "ID3D12DescriptorHeap"),
    ("PIPELINE_STATE", "ID3D12PipelineState"),
    ("ROOT_SIGNATURE", "ID3D12RootSignature"),
    ("COMMAND_ALLOCATOR", "ID3D12CommandAllocator"),
    ("HEAP", "ID3D12Heap"),
    ("QUERY_HEAP", "ID3D12QueryHeap"),
]

INTEGER_TYPES = {
    "UINT", "INT", "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16",
    "INT32", "INT64", "ULONG", "LONG", "DWORD", "SIZE_T", "BOOL", "LONG64",
    "ULONG64", "BYTE", "WORD", "USHORT", "SHORT", "UCHAR", "CHAR",
    "D3D12_GPU_VIRTUAL_ADDRESS", "D3D12_DESCRIPTOR_HEAP_TYPE",
}
GUID_TYPES = {"REFIID", "REFGUID", "REFCLSID"}
HANDLE_TYPES = {"D3D12_CPU_DESCRIPTOR_HANDLE", "D3D12_GPU_DESCRIPTOR_HANDLE"}


# SAL annotations the shared SAL regex misses: parenthesised ones with
# nested commas, and bare ones such as _In_z_ / _Out_writes_bytes_opt_.
SAL_CALL = re.compile(
    r"\b_[A-Z][A-Za-z_]*_\([^()]*(?:\([^()]*\)[^()]*)*\)|\b_[A-Z][A-Za-z0-9_]*_(?=\s)"
)


def split_commas(args):
    """Split on top-level commas only (SAL annotations carry commas)."""
    parts, depth, cur = [], 0, ""
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def split_params(args):
    """Yield (type, name, is_array) for each parameter of a cleaned list."""
    if not args or args == "void":
        return []
    out = []
    for raw in split_commas(args):
        raw = SAL_CALL.sub(" ", raw)
        p = clean(raw)
        is_array = False
        m = re.match(r"(.*?)\s*\[\s*\d*\s*\]$", p)
        if m:
            p = m.group(1)
            is_array = True
        m = re.match(r"^(.*?)\s*\**\s*([A-Za-z_][A-Za-z0-9_]*)$", p)
        if not m:
            raise ValueError("cannot parse parameter: " + raw)
        name = m.group(2)
        typ = clean(p[: len(p) - len(name)])
        out.append((typ, name, is_array))
    return out


def arg_format(typ, name, is_array, guid_bufs):
    """Return (fmt, expr, pre) for one parameter or None to skip it."""
    base = typ.replace("const ", "").replace("volatile ", "").strip()
    if is_array or base.endswith("*") or base in ("LPCWSTR", "LPCSTR", "HANDLE",
                                                   "HWND", "HMODULE", "LPVOID"):
        return ("%s=%p", f"(const void *){name}", None)
    if base in GUID_TYPES:
        buf = f"g{len(guid_bufs)}"
        guid_bufs.append(buf)
        return ("%s=%s", f"d12_api_trace_guid({name}, {buf})",
                f"char {buf}[D12_API_TRACE_GUID_LEN];")
    if base in HANDLE_TYPES:
        return ("%s=0x%llx", f"(unsigned long long){name}.ptr", None)
    if base in ("FLOAT", "float", "double"):
        return ("%s=%g", f"(double){name}", None)
    if base in INTEGER_TYPES:
        return ("%s=%llu", f"(unsigned long long){name}", None)
    if re.match(r"^(D3D12|DXGI|D3D)_[A-Z0-9_]+$", base):
        if base.endswith("_FLAGS"):
            return ("%s=0x%llx", f"(unsigned long long){name}", None)
        return ("%s=%llu", f"(unsigned long long){name}", None)
    return None


def emit_wrapper(lines, tag, iface, ret, name, args):
    args = clean(SAL_CALL.sub(" ", args))
    params = split_params(args)
    fn = f"d12_api_trace_{iface}_{name}"
    guid_bufs = []
    fmt_parts = []
    exprs = []
    pres = []
    for typ, pname, is_array in params:
        if pname == "This":
            continue
        f = arg_format(typ, pname, is_array, guid_bufs)
        if f is None:
            fmt_parts.append(f"{pname}=?")
            continue
        fmt, expr, pre = f
        fmt_parts.append(fmt.replace("%s=", f"{pname}=", 1))
        exprs.append(expr)
        if pre:
            pres.append(pre)
    fmt = ", ".join(fmt_parts)
    call_args = ", ".join(p[1] for p in params)
    returns_hresult = ret.replace(" ", "") == "HRESULT"
    is_void = ret.replace(" ", "") == "void"

    lines.append(f"static {ret} STDMETHODCALLTYPE")
    lines.append(f"{fn}({args})")
    lines.append("{")
    for pre in pres:
        lines.append(f"   {pre}")
    lines.append(f"   const bool trace = d12_api_trace_enabled();")
    lines.append("   if (trace)")
    extra = (", " + ", ".join(exprs)) if exprs else ""
    lines.append(f'      d12_api_trace_call("{iface}::{name}", (const void *)This, "{fmt}"{extra});')
    if is_void:
        lines.append(f"   d12_api_trace_real_{iface}->{name}({call_args});")
    else:
        lines.append(f"   {ret} d12_result = d12_api_trace_real_{iface}->{name}({call_args});")
        if returns_hresult:
            lines.append("   if (trace || (FAILED(d12_result) && d12_api_trace_fail_enabled()))")
            lines.append(f'      d12_api_trace_hresult("{iface}::{name}", (const void *)This, d12_result, trace ? NULL : "{fmt}"{extra});')
        lines.append("   return d12_result;")
    lines.append("}")
    lines.append("")


def emit(header):
    lines = []
    lines.append("/*")
    lines.append(" * Copyright 2026 Aurora Silicon")
    lines.append(" * SPDX-License-Identifier: MIT")
    lines.append(" *")
    lines.append(" * GENERATED by gen_api_trace.py -- do not edit by hand.")
    lines.append(" *")
    lines.append(" * D3D12AGX_DEBUG=api tracing vtables: one typed forwarding wrapper per")
    lines.append(" * COM method, logging the call and (for HRESULT methods) its result.")
    lines.append(" * Included only by d12_api_trace.c.")
    lines.append(" */")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")

    total = 0
    for tag, iface in INTERFACES:
        nonwin_body = vtable_body(header, iface, windows=False)
        win_body = vtable_body(header, iface, windows=True)
        if nonwin_body is None or win_body is None:
            print(f"ERROR: no vtable for {iface}", file=sys.stderr)
            sys.exit(1)

        methods = parse_methods(nonwin_body)
        win_methods = {name: (ret, args) for ret, name, args in parse_methods(win_body)}
        total += len(methods)
        lines.append(f"/* ---- {iface}: {len(methods)} methods ---- */")
        lines.append(f"static const {iface}Vtbl *d12_api_trace_real_{iface};")
        lines.append("")

        for ret, name, args in methods:
            win_ret, win_args = win_methods[name]
            if (win_ret, win_args) != (ret, args):
                lines.append("#ifdef _WIN32")
                emit_wrapper(lines, tag, iface, win_ret, name, win_args)
                lines.append("#else")
                emit_wrapper(lines, tag, iface, ret, name, args)
                lines.append("#endif")
                lines.append("")
            else:
                emit_wrapper(lines, tag, iface, ret, name, args)

        lines.append(f"static const {iface}Vtbl d12_api_trace_{iface}_vtbl = {{")
        lines.append(",\n".join(f"   .{name} = d12_api_trace_{iface}_{name}" for _, name, _ in methods))
        lines.append("};")
        lines.append("")

    lines.append("/* Interface tag -> generated vtable and its real-implementation slot. */")
    lines.append("static const struct {")
    lines.append("   const void *trace_vtbl;")
    lines.append("   const void **real;")
    lines.append("} d12_api_trace_table[D12_API_TRACE_IFACE_COUNT] = {")
    for tag, iface in INTERFACES:
        lines.append(f"   [D12_API_TRACE_{tag}] = {{&d12_api_trace_{iface}_vtbl,")
        lines.append(f"                             (const void **)&d12_api_trace_real_{iface}}},")
    lines.append("};")
    lines.append("")
    print(f"generated trace wrappers for {total} methods", file=sys.stderr)
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
