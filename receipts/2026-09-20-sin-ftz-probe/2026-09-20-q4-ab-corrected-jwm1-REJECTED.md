# Q4 A/B corrected — jwm1 G13G, candidate REJECTED (2026-09-20)

REJECTED: ctx1024 prefill +6.0% regression. Digests held on both arms.
Not an FTZ fix — clamped-SSBO-load restructure only.

## Raw data

| run | decode tok/s | prefill s | digest |
|---|---:|---:|---|
| base short r1 | 116.32 | 0.079 | `7fd25a869ff21678` |
| base short r2 | 115.42 | 0.078 | `7fd25a869ff21678` |
| base short r3 | 116.32 | 0.080 | `7fd25a869ff21678` |
| base ctx1024 r1 | 100.42 | 0.954 | `7da83f06ec9f001d` |
| base ctx1024 r2 | 100.08 | 0.956 | `7da83f06ec9f001d` |
| base ctx1024 r3 | 99.87 | 0.951 | `7da83f06ec9f001d` |
| cand short r1 | 113.94 | 0.077 | `7fd25a869ff21678` |
| cand short r2 | 115.91 | 0.079 | `7fd25a869ff21678` |
| cand short r3 | 115.81 | 0.078 | `7fd25a869ff21678` |
| cand ctx1024 r1 | 100.43 | 1.010 | `7da83f06ec9f001d` |
| cand ctx1024 r2 | 99.92 | 1.014 | `7da83f06ec9f001d` |
| cand ctx1024 r3 | 100.18 | 1.011 | `7da83f06ec9f001d` |

## Medians

| leg | base | cand | delta |
|---|---:|---:|---:|
| short decode | 116.32 | 115.81 | −0.44% |
| ctx1024 decode | 100.08 | 100.15 | +0.07% |
| ctx1024 prefill | 0.954 | 1.011 | **+6.0%** |

## Wheel SHAs

base libmlx: c28a485f98b55aadf12c68ae6d373f308517177c0cf33ad3c77ea82a796e931e
cand libmlx: 0300da4c2815440e537ee7c2036f448e13f7c2695c98dd8e8b8117bba226908d
fckey driver: aade269367b42001adb20156990570bf6ee1afd1576b43f778ee5c6a507f50b3
e167 driver: 7087accecede1f556ed604d28fe88a8f6e5e90df202bc8cdf50c4cea51c76bfd
base wheel: fe51534b1ed658f64f2e16f5ce4709c27cbb1c683388b50a2d04eacf6b362a01
cand wheel: a5792abfafc53a6a649fea0a6217d72f0e55a0e6f27cb22cccbb965e05c1dd8c

## 16-shape

NOT RUN. Qualification incomplete.
