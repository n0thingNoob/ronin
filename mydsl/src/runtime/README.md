# runtime (phase 3, empty)

**Do not put code here until phases 1 and 2 are complete.**

This directory will hold a small C++ runtime library that codegen output
links against. The point is to keep generated code small and readable —
common helpers (cross-core pipe setup, semaphore-based credit flow control,
NOC coordinate resolution) live here as ordinary functions, not as expanded
inline boilerplate in every generated kernel.

Anticipated contents (subject to phase 2 lessons):

- A `CrossCorePipe` abstraction wrapping the producer-side `noc_async_write`
  + semaphore-inc pattern and the consumer-side wait + buffer-publish pattern
- Credit-based flow control helper
- Layout helpers for L1 buffer placement when a core hosts multiple producers
  or consumers

Tanto already provides intra-core primitives (`pipe<T>`, `math<T>`,
`semaphore`). This runtime layer adds **only** the cross-core composition
that Tanto leaves to the user.
