# G17P fragment minimum-LOD sampling

The common Honeykrisp feature path advertises `shaderResourceMinLod` on
M1/M2. G17P keeps it disabled by default. The diagnostic token
`HK_G17P_UNVERIFIED_FEATURES=shader_resource_min_lod` exposes it for a bounded
hardware gate, not as a claim that the entire Vulkan feature is supported.

The G17P native compiler emits SMP `f140=2` for a register minimum. With a
bias, SMP retains `f139=3`; with explicit gradients it retains `f139=5`.
PACK_SMPIR publishes the minimum with type `f74=1`. A bias/minimum pair uses
one publication, with the minimum in `f16` and the bias in `f30`, whose type
and destination position are `f76=2`, `f77=16`.

Shared AGX NIR packs bias in the low half and minimum in the high half.
The G17P packer therefore supplies source addresses in the opposite order.
A gradient tuple appends the minimum as the low half of a padded final
32-bit component. No gradient rescaling or scalar explicit-LOD substitution
is involved. Implicit samples retain the pixel-entry and sample-group
protocol; explicit gradients remain outside sample groups.

The bounded implementation covers non-array fragment 2D and 3D samples,
including indexed descriptors, 2D fixed offsets, and 2D depth comparisons.
Shared lowering maps 1D cases to the 2D path. Arrays, cubes, sparse results,
3D offsets, dynamic offsets, and non-fragment clamp forms are not qualified
by this change. The packed bias/minimum source must fit below half 128,
as must the gradient tuple, matching the current staging operand bounds.

Native instruction fixtures cover the minimum operand and its composition
with bias, gradients, offsets, comparisons, dimensions, and descriptor
indexing. Stream tests verify the NIR half order, gradient append position,
resource footprint, and pixel-entry distinction. The opt-in CTS gate still
requires correct pixels on G17P before any claim of hardware qualification.
Passing that bounded gate alone does not justify enabling the global feature.

## Scoreboard operand coverage

A 3D gradient/minimum sample reads 28 half-register tokens: six coordinates,
fourteen gradient/minimum halves, and four halves for each descriptor handle.
The old 16-token list silently dropped the late descriptor dependency. In
the failing CTS shader, the descriptor load published half 12 on slot 2,
but the first minimum PACK did not wait for it. SMP waited only for its own
staging slot. The independent CTS-shaped probe also failed in 32-pixel groups.

The scoreboard list now holds 32 tokens and refuses overflow or out-of-range
registers instead of truncating them. Tests cover late texture and sampler
dependencies, both ordinary 3D gradients and gradients with a minimum, and
retention of unrelated asynchronous work. This changes dependency accounting,
not register allocation, sampler encoding, batching, or feature defaults.
