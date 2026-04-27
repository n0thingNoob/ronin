# frontend (phase 3, empty)

**Do not put code here until phases 1 and 2 are complete.**

This directory will hold the DSL's surface language: parser, AST, type
checker. Designing it before walking through the cross-core protocol by hand
(phase 2) is the single most likely way to ship a DSL that misses
backpressure, deadlock, or L1-budget concerns.

When phase 2 is done and `doc/notes.md` has a written-out cross-core
protocol, design discussions can begin. Anticipated open questions at that
point:

- Single-file kernel surface vs control + math split
- How the user describes the op-graph topology (textual vs builder API)
- How placement is expressed (manual annotations vs automatic)
- Type system: do tile shapes flow in stream types? activation/weight
  distinction?
- Fork/join semantics
- Whether the language is C++-embedded (like CUDA/Triton) or standalone

None of these have correct answers from first principles — they all depend
on what hurt during phase 2.
