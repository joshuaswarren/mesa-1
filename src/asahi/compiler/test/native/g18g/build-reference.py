#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile identical probe inputs for G17G and G18G with Apple's toolchain.

This creates native reference binaries, not Mesa runtime qualification.
Generated binaries and compiler logs belong outside the source checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='append', help='Existing g17g reference basename')
    args = parser.parse_args()
    refs = Path(__file__).resolve().parent.parent/'g17g'
    names = args.probe or [p.stem for p in sorted(refs.glob('*.mtlp-json'))]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope='Apple compiler references only; no GPU execution',
                  hardware_execution=False, mesa_execution=False, probes=[])

    def run(argv, cwd, log):
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
        log.write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'{argv[1]} failed with {result.returncode}: {log}')
        return result.stdout

    report['metal_version'] = run(['xcrun','metal','--version'],out,out/'metal-version.txt').strip()
    arches = run(['xcrun','metal-nt','-archs'],out,out/'architectures.txt').split()
    assert 'applegpu_g18g' in arches and 'applegpu_g17g' in arches
    report['host_os'] = run(['sw_vers'],out,out/'host-os.txt').strip()
    try:
        for name in names:
            source, config = refs/(name+'.metal'), refs/(name+'.mtlp-json')
            assert source.is_file() and config.is_file() and Path(name).name == name
            folder = out/name; folder.mkdir()
            air, lib = folder/'probe.air', folder/'probe.metallib'
            run(['xcrun','metal','-std=metal3.0','-mmacosx-version-min=27.0',
                 '-c',str(source),'-o',str(air)],folder,folder/'metal.log')
            run(['xcrun','metallib',str(air),'-o',str(lib)],folder,folder/'metallib.log')
            probe = dict(name=name, source_sha256=digest(source), config_sha256=digest(config),
                         air_sha256=digest(air), metallib_sha256=digest(lib), targets={})
            for arch in ('applegpu_g17g','applegpu_g18g'):
                target = folder/arch; target.mkdir()
                binary = target/'probe.gpubin'
                command = ['xcrun','metal-nt','-arch',arch,'-platform_version','macos','27.0','27.0',
                           '-N',str(config),str(lib),'-save-temps','-o',str(binary)]
                run(command,target,target/'metal-nt.log')
                assert binary.stat().st_size > 0
                row = dict(sha256=digest(binary), size=binary.stat().st_size,
                           command=command, binary=str(binary.relative_to(out)))
                probe['targets'][arch] = row
            report['probes'].append(probe)
            print('COMPILED',name,'G17G and G18G',flush=True)
    finally:
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REFERENCE_MANIFEST',out/'manifest.json',flush=True)


if __name__ == '__main__':
    main()
