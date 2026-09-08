#!/usr/bin/env python3
"""Compile Hyprland 0.56.1's complete built-in GLES3 shader corpus on Asahi."""

import pathlib
import shutil
import subprocess
import sys
import tempfile


FRAGMENTS = (
    "quad.frag",
    "passthru.frag",
    "rgbamatte.frag",
    "ext.frag",
    "blur1.frag",
    "blur2.frag",
    "blurprepare.frag",
    "blurfinish.frag",
    "shadow.frag",
    "inner_glow.frag",
    "surface.frag",
    "border.frag",
    "glitch.frag",
)

FEATURES = (
    "USE_ALT_TONEMAP",
    "USE_BLUR",
    "USE_BLUR_ALPHA_MASK",
    "USE_BLUR_MATTE",
    "USE_CM",
    "USE_DISCARD",
    "USE_ICC",
    "USE_MIRROR",
    "USE_MOTION_BLUR",
    "USE_RGBA",
    "USE_ROUNDING",
    "USE_SDR_MOD",
    "USE_TINT",
    "USE_TONEMAP",
)


def preprocess(tool: str, shader_dir: pathlib.Path, source: pathlib.Path) -> str:
    result = subprocess.run(
        [
            tool,
            "-E",
            f"-I{shader_dir}",
            str(source),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"preprocessing {source.name} failed:\n{result.stderr}")

    # Hyprland's CShaderLoader removes glslang's source-map directives before
    # handing the source to GLES. Match that production behaviour exactly.
    return "".join(
        line for line in result.stdout.splitlines(keepends=True)
        if not line.startswith("#line ")
    )


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: run-hyprland-corpus.py <shim> <libEGL> <gallium-dri> "
            "<state-matrix-binary> <Hyprland glsl dir>",
            file=sys.stderr,
        )
        return 2

    shim, vendor, gallium, binary, source_dir = map(pathlib.Path, sys.argv[1:])
    tool = shutil.which("glslangValidator")
    if not tool:
        print("glslangValidator is required", file=sys.stderr)
        return 77

    missing = [name for name in ("tex300.vert", *FRAGMENTS)
               if not (source_dir / name).is_file()]
    if missing:
        print(f"incomplete Hyprland shader tree: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="hyprland-g17p-corpus-") as tmp:
            root = pathlib.Path(tmp)
            working = root / "source"
            outputs = root / "preprocessed"
            shutil.copytree(source_dir, working)
            outputs.mkdir()

            # CShaderLoader supplies every feature define for every compile.
            # Exercise the exact baseline variant rather than the repository's
            # static all-features development header, whose combinations are
            # not valid for every standalone built-in shader.
            (working / "defines.h").write_text(
                "".join(f"#define {feature} 0\n" for feature in FEATURES)
            )

            inputs = []
            for name in ("tex300.vert", *FRAGMENTS):
                output = outputs / name
                output.write_text(preprocess(tool, working, working / name))
                inputs.append(output)

            wrapper = pathlib.Path(__file__).with_name("run-gl-shim.py")
            command = [
                sys.executable,
                str(wrapper),
                str(shim),
                str(vendor),
                str(gallium),
                str(binary),
                *(str(path) for path in inputs),
            ]
            return subprocess.call(command)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
