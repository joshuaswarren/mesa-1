# AGX3 generic instruction encoder

`agx3_compose.c` is one table-driven encoder for every form of both AGX3
encodings. `agx3b_tables.c` / `agx3c_tables.c` are generated from Apple's own
llvm::AGX3 encoder tables; treat them as data and do not edit them by hand.

The consumer supplies Apple's anonymous field values (`fields[160]`); the
encoder gates the form on its predicates, selects each choice group, and
deposits every fixed bit and field under a shared written-bit shadow, failing
closed on any contradiction, unconsumed nonzero field, or over-wide value.
