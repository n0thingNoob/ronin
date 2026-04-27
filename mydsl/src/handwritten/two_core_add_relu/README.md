# two_core_add_relu — superseded

This directory's original plan (one demo with separate kernel binaries per
core, 6 device files) was replaced during phase 2 by **two simpler demos
that use MPMD via `core_id` runtime arg**:

- `../two_core_add/` — phase 2 step 1, 2-core dataflow with NOC L1→L1 mcast
  for inputs.
- `../three_core_add_relu/` — phase 2 step 2, full 3-core DAG (read /
  compute / forward / write), intermediate `T` never touches DRAM.

The reasons for the change are recorded in `../../../ROADMAP.md` (decision
log entry 2026-04-26 about Tanto's one-Kernel-per-Grid binding).

This directory is kept as a placeholder so the original phase-2 plan in git
history remains discoverable. No code lives here.
