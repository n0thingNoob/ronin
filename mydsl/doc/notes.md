# Design spec & gotchas

This is the load-bearing document for the project. It captures every
decision, primitive lowering, and ordering rule that's required to write or
edit handwritten kernels correctly. Phase 3 codegen will be driven by what is
written here.

If you (Claude or human) are about to change a kernel, scan this file first —
each section is a wish list of bugs you can avoid by knowing what's here.

---

## The 7-axis alignment contract (Tanto, single-core)

Discovered while reading `algo/src/basic/` for `eltwise_binary`. Every
hand-written operator must align these across reader/math/writer/host:

1. **Pipe identity** — each pipe appears in exactly two device files, with
   opposite roles (producer reserves/pushes; consumer waits/pops).
2. **Pipe metadata** — element type T, frame_size, depth, direction. Host
   declares once; both kernels must agree.
3. **Runtime args order** — host `set_args` list order must equal device
   kernel signature order. Position-based, no name check.
4. **Compile-time `param<>`** — `param<uint32>` in Tanto becomes a `-Pn=v`
   flag at translation; host must load the matching generated metal file.
5. **Global buffer** — element type, total size, page size between host
   `core::Global` and device `global<T>`.
6. **Tile-count triangle** — `m_N == num_blocks * block_tiles * 1024` (1024 =
   one tile in elements, 32x32). Iterate `num_blocks` outer, `block_tiles`
   inner. block_tiles == frame_size.
7. **`-DT=` define** — front.sh and host `m_defines` must both pass T.

Bugs ranked by how often they bite, descending:
1. runtime args order off by one
2. pipe direction reversed (deadlock)
3. frame_size != inner loop count (overrun)
4. tile-count triangle violated
5. param order off (wrong op)
6. host loads wrong metal file

---

## Cross-core pipe protocol (phase 2 — point-to-point)

Tanto's `pipe<T>` is declared on a `Grid` and gets the same L1 layout (offset,
size, frame_size) on every core in the grid. Each core has its own per-core
state (front/back pointers). Cross-core data movement happens by NOC writes
from the producer's L1 region into the consumer's identically-laid-out L1
region; per-core pipe state is then advanced on each side independently.

### Primitives (Tanto level)

| Tanto                                     | Lowers to                                          |
|------------------------------------------|----------------------------------------------------|
| `pipe.write_mcast(0, pipe, 0, n,         | `noc_async_write_multicast(local_l1, mcast_addr,  |
|  x0,y0,x1,y1, num_dests)`                |  n*itemsize, num_dests)`                          |
| `sem.inc(x, y, 1)`                       | `noc_semaphore_inc(get_noc_addr(x,y,sem.addr),1)` |
| `sem.set_mcast(sem, x0,y0,x1,y1, N)`     | `noc_semaphore_set_multicast(sem.addr, ..., N)`   |
| `sem.wait(N)`                            | `noc_semaphore_wait(sem.addr, N)`                 |
| `sem.set(N)`                             | `noc_semaphore_set(sem.addr, N)`                  |

For point-to-point use `write_mcast` with `x1=x0`, `y1=y0`, `num_dests=1` —
that's a 1-destination "mcast", which is just a single NOC write.

### Two-semaphore handshake (per block, one pipe)

Two semaphores, both declared on the cross-core grid:
- `sem_credit` — consumer increments to tell producer "I have buffer space"
- `sem_data`   — producer sets to tell consumer "data is in your L1"

```
PRODUCER WRITER (core 0)            CONSUMER READER (core 1)
pa.reserve_back();                  pa.reserve_back();
pa.read(0, ga, ...);                sem_credit.inc(P_x, P_y, 1);
read_barrier();                     sem_data.wait(1);
sem_credit.wait(1);                 sem_data.set(0);
sem_credit.set(0);                  pa.push_back();           // NOC fill arrived
pa.write_mcast(0, pa, 0, n,         (math then wait_fronts pa, computes, pops)
        C_x, C_y, C_x, C_y, 1);
write_barrier();
sem_data.inc(C_x, C_y, 1);
pa.push_back();
PRODUCER MATH (core 0): drain
pa.wait_front(); pa.pop_front();
```

Key lowering quirk: `pipe.write_mcast(0, pipe, 0, n, ...)` translates to
`noc_async_write_multicast(get_write_ptr(pipe.cb_id) + 0, ..., n*itemsize)`.
The source is `get_write_ptr` — the *next* slot to be written, i.e. the slot
just reserved but not yet pushed. So the producer must do
**reserve_back → fill (DRAM read) → mcast → push_back** within ONE kernel
(the writer, BRISC, since BRISC owns NOC0 for outbound writes). Splitting the
fill into the reader kernel (which then push_backs) is wrong: by the time the
writer's `wait_front` succeeds, `get_write_ptr` has already advanced to the
*next* slot, so the mcast picks up uninitialized data from a future slot.

This in turn means producer's pa, pb get push_backed every block but never
pop_fronted by anyone on the producer side — so the producer's math kernel
must drain them with a no-op `wait_front; pop_front` loop, otherwise the
writer's `reserve_back` blocks once depth fills (deadlock).

### Why two semaphores

A single semaphore deadlocks at block boundaries: producer's "data" signal
for block N+1 can race with consumer's "credit" signal for block N's reuse.
Two semaphores keep the credit and data channels orthogonal.

### Why credit-then-data (not data-then-credit)

Producer must NOT NOC-write until consumer has reserved the buffer slot. If
producer races ahead, it overwrites a frame the consumer is still draining.
So: consumer commits buffer first → producer fills it → consumer marks ready.

### Topology and deadlock

Producer→consumer is a DAG; no cycles, so cross-core pipes can't deadlock by
themselves. Per-core pipe deadlocks (math waits for reader that waits for
writer that waits for math …) are handled the same as single-core Tanto.

### MPMD: two styles

**Style A — SPMD-via-core_id** (used by `two_core_add`, `three_core_add_relu`):
one Kernel binary per role (reader/writer/math), bound to a single Grid
covering all cores. Per-core differences are selected by a `core_id`
runtime arg branched inside the kernel. Cheaper to compile, but every
core's binary contains every branch.

**Style B — True MPMD via multiple Grids + multiple Kernels** (used by
`mpmd_two_core_add`, verified PASS 2026-04-26): one Program owns multiple
Grid objects; each Kernel is bound to a sub-grid (one per role).
Cross-core pipes/semaphores are declared on the *union* grid so L1
layout matches across all cores; sub-grid kernels reference these pipes
by cb_id.

There is **no need for "manual NOC address arithmetic for cross-grid
pipes"** — the union-grid trick gives identical CB allocation on all
cores, so existing `pipe.write_mcast` / `pipe.write` primitives work
unchanged. The host wraps:

```cpp
m_grid_full = Grid(prog, 0, 0, 0, 1);   // pipes, sems
m_grid_head = Grid(prog, 0, 0, 0, 0);   // head's 3 kernels
m_grid_tail = Grid(prog, 0, 1, 0, 1);   // tail's 3 kernels
m_pa = Pipe(prog, m_grid_full, INPUT, T, depth, frame);
m_head_writer = Kernel(prog, m_grid_head, WRITER, ..., "head_writer.cpp", ...);
m_tail_reader = Kernel(prog, m_grid_tail, READER, ..., "tail_reader.cpp", ...);
// etc.
```

Each kernel's signature carries only the args its role needs (no
`core_id`, no unused globals). This is the shape phase 3's codegen will
emit naturally — one DSL spec → one kernel binary.

**When to use which**: SPMD-via-core_id for N-way data parallel (lots of
cores doing the same thing); true MPMD for pipeline stages that do
qualitatively different jobs (DRAM-fetch vs forward vs compute). Phase 3
defaults to true MPMD.

### MPMD: middle core that both consumes and forwards (phase 2 step 2)

`three_core_add_relu` chains head -> middle -> tail. Middle has the awkward
role of both *receiving* pa,pb (consumer-side handshake) AND *producing* pt
(producer-side mcast/forward).

The producer-side forward cannot reuse the 2-core `pipe.write_mcast` trick on
its own, because the data source is the math kernel (TRISC) and the NOC
forward must run on the writer kernel (BRISC), and **TRISC has no semaphore
primitives** (`tanto/src/device/metal/compute.h` exports no `noc_semaphore_*`
or `sem.set/wait/inc`). So math cannot directly tell writer "data is ready in
this pt slot, please mcast now".

Solution: introduce a **per-core helper pipe** `pt_local` (PipeKind::INTERMED)
that lives only logically on middle. Math packs into pt_local; writer
wait_fronts pt_local; the CB itself is the math->writer sync. Then writer
forwards pt_local's content to tail's pt with point-to-point
`pipe.write(0, pt, 0, n, x, y)`, which lowers to:

```
noc_async_write(
    get_read_ptr(pt_local.cb_id) + 0,                       // src on middle
    get_noc_addr(x, y, get_write_ptr(pt.cb_id) + 0),         // dst on tail
    n * itemsize)
```

Source uses `get_read_ptr` (the just-pushed pt_local slot — correct, no
cursor drift). Destination uses middle's `get_write_ptr(pt)` interpreted as a
remote address — this requires middle's pt write_ptr to track tail's pt
write_ptr. Achieved by having both cores `reserve_back` + `push_back` pt in
lockstep per iteration. Middle's math also has to drain pt locally
(`wait_front` + `pop_front`) so the cb doesn't fill up after `depth` pushes.

Per-core kernel logic:

```
MIDDLE READER (NCRISC):     classic consumer for pa, pb
  reserve_back pa,pb; sem_credit_ab.inc(head); sem_data_ab.wait/set; push_back

MIDDLE MATH (TRISC):
  reserve_back pt_local
  wait_front pa,pb
  acc.add(pa, pb) -> pack into pt_local
  pop_front pa,pb
  push_back pt_local
  wait_front pt; pop_front pt          // local drain for cb cursor

MIDDLE WRITER (BRISC):
  wait_front pt_local
  reserve_back pt                       // advances cursor lockstep with tail
  sem_credit_t.wait/set
  pt_local.write(0, pt, 0, n, tail_x, tail_y)   // NOC L1->L1
  write_barrier
  sem_data_t.inc(tail_x, tail_y, 1)
  pop_front pt_local
  push_back pt
```

Verified PASS in `mydsl/src/handwritten/three_core_add_relu/`, max_diff=0.

The overhead is one extra L1 cb (pt_local). The DSL goal: make this whole
"middle node of pipeline" a single primitive that hides pt_local entirely.

### Open questions

- Frame size > 1 across cores: does the L1 layout still match if both cores
  declare the same pipe with the same frame_size? Tanto says yes, since the
  pipe is grid-scoped — verified for frame_size=1 in step 1.
- Can `core_id=0` (no-op math) skip the math kernel entirely, or must it
  emit a stub? Step 1 confirmed: math kernel still gets dispatched on every
  grid core; the no-op branch is required (or it would deadlock waiting for
  TRISC startup).

---

## Build & run environment

The Jitte test binary needs **two** env vars set, both with non-obvious names:

```bash
TT_METAL_HOME=/home/yijia/ronin/jitte/home \
TT_ARCH=wormhole_b0 \
    ./prj/bin/handwritten/<name>/test_tanto
```

Common pitfalls:
- `JITTE_HOME` and `ARCH_NAME` look right but are silently ignored.
- `TT_METAL_HOME` must be absolute (`./...` resolves wrong if you cd'd
  somewhere else).
- The build scripts in `jitte/prj/handwritten/<name>/deploy_jitte.sh` use
  `JITTE_HOME` internally — but that's only for the `cp -R` step, not for
  the runtime.

The Jitte source directory layout it expects under `TT_METAL_HOME`:

```
$TT_METAL_HOME/
├── tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml   (must exist)
├── built/<core_count>/kernels/<kernel_name>/<hash>/    (auto-created)
└── mydsl/handwritten/<name>/device/{tanto,metal}/      (deploy_jitte.sh writes here)
```

## Decisions deferred

- phase 1: const `3.0` — runtime arg vs `param<uint32>`?
- phase 2: backpressure — credit-based or pure semaphore-pair?
- phase 3: kernel surface granularity (single file vs control+math)
