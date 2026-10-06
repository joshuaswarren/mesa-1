#!/usr/bin/env python3
"""Replay P00 native configuration in an isolated P09 build and stage a kit."""
import configparser
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(argv, env):
    print('+', shlex.join(map(str, argv)), flush=True)
    subprocess.run(argv, env=env, check=True)


def main():
    archive, expected_hash, candidate, work_arg, p00_arg = sys.argv[1:]
    if len(candidate) != 40 or any(c not in '0123456789abcdef' for c in candidate):
        raise ValueError('candidate commit')
    if os.uname().sysname != 'Linux' or os.uname().machine != 'aarch64' or os.sysconf('SC_PAGE_SIZE') != 16384:
        raise RuntimeError('Native Linux/aarch64/16K required')
    if digest(archive) != expected_hash:
        raise ValueError('archive hash')
    p00 = Path(p00_arg).resolve()
    if (p00 / 'SOURCE-COMMIT').read_text().strip() != 'a212d162f5e4728d6723a667096348c49e701c8e':
        raise ValueError('reviewed P00 identity')
    subprocess.run(['sha256sum', '--strict', '-c', 'ARTIFACTS.sha256'], cwd=p00,
                   stdout=subprocess.DEVNULL, check=True)
    work = Path(work_arg).resolve()
    work.mkdir(parents=True, exist_ok=False)
    source, build, stage = (work / n for n in ['source', 'build', 'stage'])
    source.mkdir()
    with tarfile.open(archive) as tf:
        tf.extractall(source, filter='data')
    source_files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*'))
                    if p.is_file() and not p.is_symlink()}
    archive_source_files = source_files
    # Overlay exact frozen blobs, including the original full G5 branch.
    for path in (p00 / 'reference').rglob('*'):
        if path.is_file():
            shutil.copy2(path, source / path.relative_to(p00 / 'reference'))
    subprocess.run(['sha256sum', '--strict', '-c', str(p00 / 'reference-inputs.sha256')],
                   cwd=source, stdout=subprocess.DEVNULL, check=True)
    source_files = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*'))
                    if p.is_file() and not p.is_symlink()}
    overlay = [name for name, sha in source_files.items() if archive_source_files.get(name) != sha]
    env = os.environ.copy()
    for key in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS']:
        env.pop(key, None)
    env['MESA_GIT_SHA1_OVERRIDE'] = candidate
    pins = p00 / 'receipts'
    compilers = json.loads((pins / 'PINNED-COMPILERS.json').read_text())
    env['CC'] = shlex.join(compilers['host']['c']['exelist'])
    env['CXX'] = shlex.join(compilers['host']['cpp']['exelist'])
    options = configparser.ConfigParser(interpolation=None)
    options.optionxform = str
    options.read(pins / 'PINNED-MESON-CMD-LINE.txt')
    if dict(options['properties']):
        raise RuntimeError('Unreviewed cross/native properties')
    prefix = '/opt/aurora-d3d12-phase2/' + candidate
    flags = [f'-D{k}={prefix if k == "prefix" else v}' for k, v in options['options'].items()]
    if options['options']['wrap_mode'] != 'nodownload':
        raise ValueError('wrap policy')
    setup = ['meson', 'setup', str(build), str(source)] + flags
    run(setup, env)
    actual = json.loads((build / 'meson-info/intro-buildoptions.json').read_text())
    actual_map = {x['name']: x['value'] for x in actual}
    for pin in json.loads((pins / 'PINNED-BUILD-OPTIONS.json').read_text()):
        expected = prefix if pin['name'] == 'prefix' else pin['value']
        if actual_map.get(pin['name']) != expected:
            raise ValueError('build option: ' + pin['name'])
    actual_compilers = json.loads((build / 'meson-info/intro-compilers.json').read_text())
    for lang in ['c', 'cpp']:
        for key in ['id', 'version', 'exelist', 'linker_id', 'linker_exelist']:
            if actual_compilers['host'][lang][key] != compilers['host'][lang][key]:
                raise ValueError('compiler: ' + lang + '/' + key)
    deps = json.loads((pins / 'PINNED-DEPENDENCIES.json').read_text())
    for name, version in deps.items():
        if subprocess.check_output(['pkg-config', '--modversion', name], env=env, text=True).strip() != version:
            raise ValueError('dependency version: ' + name)
    names = ['g1_smoke', 'g2_smoke', 'g3_smoke', 'g4_smoke', 'g4_triangle', 'g5_depth',
             'g5_descriptors', 'g5_atomics', 'g5_control_flow', 'g5_wave', 'g5_texture',
             'g6_geometry', 'g7_tessellation', 'g8_mesh', 'g9_amplification',
             'g10_vertex_buffers', 'g11_indexed_instanced']
    d12 = 'src/asahi/d3d12/'
    targets = [d12 + 'libd3d12agx.so'] + [d12 + 'd3d12agx_' + n for n in names]
    diagnostics = [d12 + 'd3d12agx_native_stage_certification',
                   'src/asahi/drm-shim/libasahi_noop_drm_shim.so']
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands'] + targets + diagnostics,
                                       env=env, text=True)
    if 'D12_D32_FIRST_PAIR_PROBE' in commands:
        raise RuntimeError('partial-suite graph')
    command = ['ninja', '-C', str(build), '-j3'] + targets + diagnostics
    run(command, env)
    stage.mkdir()
    (stage / 'bin').mkdir()
    (stage / 'lib').mkdir()
    receipts = stage / 'receipts'
    receipts.mkdir()
    diag = stage / 'diagnostics/P09'
    diag.mkdir(parents=True)
    for t in targets:
        shutil.copy2(build / t, stage / 'bin' / Path(t).name)
    for t in diagnostics:
        shutil.copy2(build / t, diag / Path(t).name)
    shutil.copy2(source / 'src/asahi/d3d12/tests/run-native-stage-certification.sh', diag)
    shutil.copy2(source / 'docs/d3d12-native-stage/build-native-stage-kit.py', diag)
    dependencies = {}
    for t in targets + diagnostics:
        resolved = subprocess.check_output(['ldd', str(build / t)], env=env, text=True)
        (receipts / (Path(t).name + '.ldd.txt')).write_text(resolved)
        for line in resolved.splitlines():
            words = line.split()
            if 'not found' in line:
                raise RuntimeError(line)
            if '=>' not in words or len(words) < 3 or not words[2].startswith('/'):
                continue
            name, path = Path(words[0]).name, Path(words[2]).resolve()
            if name == 'libd3d12agx.so':
                if path != (build / targets[0]).resolve():
                    raise ValueError('library resolution')
                continue
            if str(path).startswith('/opt/neo-mesa-'):
                raise ValueError('unrelated Mesa dependency')
            sha = digest(path)
            if name in dependencies and dependencies[name]['sha256'] != sha:
                raise ValueError('dependency collision')
            dependencies[name] = dict(source=str(path), sha256=sha)
            shutil.copy2(path, stage / 'lib' / name)
    shutil.copytree(p00 / 'reference', stage / 'reference')
    for name in ['reference-inputs.sha256', 'REFERENCE-COMMIT', 'KERNEL-IDENTITY.json']:
        shutil.copy2(p00 / name, stage / name)
    kernel = json.loads((stage / 'KERNEL-IDENTITY.json').read_text())
    if os.uname().release != kernel['release'] or digest(kernel['image_path']) != kernel['image_sha256']:
        raise ValueError('running kernel differs from the matched P00 Image; new reviewed kernel receipt required')
    kernel['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    probe = p00 / 'diagnostics/P00/kmd-params'
    kernel['kmd_feature_evidence'] = json.loads(subprocess.check_output(
        [str(probe), kernel['kmd_feature_evidence']['render_node']], env=env, text=True))
    if not kernel['kmd_feature_evidence']['exact_prior_barriers'] or kernel['kmd_feature_evidence']['gpu_generation'] != 17:
        raise ValueError('matched G17P KMD features required')
    kernel['boot_repin'] = 'P09 approved native build request; same release and trusted Image hash; read-only GET_PARAMS'
    (stage / 'KERNEL-IDENTITY.json').write_text(json.dumps(kernel, indent=2) + '\n')
    (stage / 'SOURCE-COMMIT').write_text(candidate + '\n')
    (receipts / 'SELECTED-COMMANDS.txt').write_text(commands)
    (receipts / 'SOURCE-FILES.json').write_text(json.dumps(source_files, indent=2) + '\n')
    for name in ['compile_commands.json', 'build.ninja']:
        shutil.copy2(build / name, receipts / name)
    for name in ['intro-buildoptions.json', 'intro-compilers.json', 'intro-dependencies.json']:
        shutil.copy2(build / 'meson-info' / name, receipts / name)
    binaries = {str(p.relative_to(stage)): digest(p) for p in sorted((stage / 'bin').iterdir())}
    receipt = dict(candidate=candidate, reference=(stage / 'REFERENCE-COMMIT').read_text().strip(),
        source_archive_sha256=expected_hash, source_files='receipts/SOURCE-FILES.json',
        frozen_overlay=overlay,
        candidate_library_sha256=binaries['bin/libd3d12agx.so'], binaries=binaries,
        runtime_dependencies=dependencies, setup_command=setup, build_command=command,
        build_options=actual, compiler_versions=actual_compilers, dependency_versions=deps,
        exact_compile_link_commands='receipts/SELECTED-COMMANDS.txt',
        source_directory=str(source), build_directory=str(build),
        D12_D32_FIRST_PAIR_PROBE=dict(present=False), GPU_submission=False,
        reference_inputs=json.loads((p00 / 'REFERENCE-BUILD.json').read_text())['reference_inputs'])
    (stage / 'REFERENCE-BUILD.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (stage / 'ARTIFACTS.sha256').write_text(''.join(
        f'{digest(p)}  {p.relative_to(stage)}\n' for p in sorted(stage.rglob('*')) if p.is_file()))
    print('P09_BUILD_STAGED', stage, flush=True)


if __name__ == '__main__':
    main()
