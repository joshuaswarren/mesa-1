# Sample-rate alpha coverage and depth investigation

Inputs `g17p-sample-alpha-depth.metal` and `.mtlp-json` compile 2x/4x
sample-dependent alpha and depth, including a pixel-coordinate dependency.
Toolchain v27.1.5194.15, `metal-nt -arch applegpu_g17p
-platform_version macos 27.0 27.0`. Use the same metal/metallib/native command
sequence as the adjacent explicit-sample-count witness.

Native object SHA256:
`8735f1c1edebd49e311e9bb4a6d041b46f336ab86aa461d0d3d05dd118697375`.
The nested 2x fragment text is 482 bytes, SHA256
`f7a4f2428101a566043b3c7a275e2cfc5dc533ce298bc0a6eff3c89a13fb20ef`;
4x is 482 bytes, SHA256
`7694639e4099a539a181a09cdd350d156de1cc8d0e04ae2fcffc3afb8525b47c`.

Unlike the earlier simple per-sample witnesses, this shader has an actual
loop: backward branch at text offset 0x1c8, one depth store at 0x16c and
one colour store at 0x1a4 in the loop. Do not generalize the earlier unrolling
observation to every native shader. Its combined pixel wait at 0x6e is
`8712560d0800`, exact compose round-trip `form 230 pixwait ep0 size6:
14:13 18:1 23:1 34:1`. The depth store is `d71454000003`, round-trip
`form 38 depthf ep0 size6: 9:5 12:1 13:1 17:1 25:1`; colour store
`e706540a00000cce08000008` round-trips as `form 211 lstcol ep0 size12:
2:10 9:2 13:1 26:15 35:9 37:7 38:1 44:12 51:2 52:1 55:1 65:1 66:1`.

This is evidence for native sample-dependent alpha/depth support, not proof
that Mesa's different split epilog has the same transaction structure. The
candidate extends Acelogic's J700 `5aa3f3324a0` pure-kill coalescing to the
invariant FragCoord/pixel-coordinate sources used by alpha-coverage dithering.
The union of sample kills can precede depth writes; coverage reads themselves
remain immovable. Host checks validate one combined kill and four staged depth
writes. The extended mask matrix covers every nonempty 2x/4x API mask against
all 256 exported masks; malformed colour protocols remain rejected.

## Hardware qualification

Neo, native Vulkan API, CTS 1.4.6.2, qualified DRM SHA256
`c3080a3c4674bcf085927037d0869b613dd88962c3f0ca0c88a7b9a4d563f4cf`:

- `j700-depth-coalesce-family28-baseline`: 12 Fail / 16 Pass before this change.
- `j700-depth-coalesce-family28-r1` and `-r2`: 28/28 Pass each, covering four
  static/dynamic depth-export shapes under seven pipeline modes.
- `j700-depth-coalesce-reject6-r1`: all six original compiler-rejection probes
  pass, including the two previously fixed alpha-to-one controls.
- `j700-depth-mask-controls16-r1`: two early-fragment SampleMask/depth controls
  pass; fourteen pipeline controls report NotSupported because they require
  `VK_EXT_post_depth_coverage`. Those fourteen are not qualification passes.
- Every run has zero GPU fault markers and retains the desktop.
- Full `agx_tests`, `asahi-g17p-early-demote-epilog`, and
  `asahi-g17p-msaa-epilog` host suites pass.

Candidate `k4-j700-depth-coalesce-r1` library SHA256:
`81674f316847409eaf18e25e16c23b61d080921953db28155858e6b4e3ac0ae5`.
No runtime sources changed after this candidate was packaged. This closes the
twelve observed depth-export failures; non-depth sample-rate A2C and the
pixel-centre interpolation failures remain separate unresolved work.
