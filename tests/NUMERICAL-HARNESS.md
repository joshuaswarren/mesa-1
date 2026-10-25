# Numerical qualification harness

Driver source under test: Joshua Warren’s correctness-only adaptation
`5e45716701b64a4b627e7e60df5b781c65c87f42`. CI source is the previously checked
`001e33ae855bd6b9ddc28bfc60a6bd64d71bcf91` trial tree. This branch adds test
harness fixes only; the production PR and package source pins are unchanged.

The ULP checker uses a positive binary32 ULP at the magnitude of the correctly
rounded reference, calculated in binary64. It explicitly rejects a non-finite
result for a finite reference. Special reference values retain bitwise checks,
with NaN payloads ignored. Seven CPU controls cover both signs, a power-of-two
boundary, out-of-budget results, NaN/infinity, FLT_MAX and signed zero.

The original checker measures binary64 spacing, gives negative distances for
negative references and can accept NaN for a finite reference. These are oracle
defects, not observations of GPU arithmetic failure.

The following files were copied byte for byte from Joshua Warren’s public
source at `f144c7949b7d960f26677b671341d5a123057ddf` to supply the fixtures
missing from the correctness extraction. No CDM, scheduling or performance
source is imported.

- [precise_math.comp](https://github.com/joshuaswarren/mesa-1/blob/f144c7949b7d960f26677b671341d5a123057ddf/tests/precise_math.comp), Git blob `827941d2468890243996dc8d9ab7f74005dab062`.
- [compute_runner.cpp](https://github.com/joshuaswarren/mesa-1/blob/f144c7949b7d960f26677b671341d5a123057ddf/tests/compute_runner.cpp), Git blob `e16d45d9d61252d72b2389da8c5073c781e696ca`.

Validation: seven CPU oracle controls pass. No numerical GPU result is claimed by this commit.
