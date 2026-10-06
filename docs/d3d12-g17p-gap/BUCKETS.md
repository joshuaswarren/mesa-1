# Accepted Neo B-only gap

| First observed mechanism | Cases |
|---|---:|
| native graphics admission | 291 |
| capability/API or adapter skip | 76 |
| assertion failure (unresolved) | 32 |
| texture compiler subset | 27 |
| depth admission | 26 |
| native meta compute program | 12 |
| typed buffer addressing | 9 |
| crash/timeout without mechanism | 7 |
| image atomics addressing | 7 |
| indirect compute admission | 3 |
| compute ABI | 2 |
| descriptor/binding | 2 |
| shader linkage | 1 |
| other diagnostic (unresolved) | 1 |

Total: 496. Full case names, A/B commands, log hashes, first assertions and plan slices are in BUCKETS.json and CASES.tsv. Counts describe the accepted 82dee6e1 measurement, not the newer fallback candidate.

The first 291 graphics-admission refusals mask later vertex/index/instancing, format/layout, descriptor and stage failures. Reclassify only after an explicit physical-admission run. Zero direct observations in a category do not prove that producer works.
