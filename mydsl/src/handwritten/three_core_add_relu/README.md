# three_core_add_relu (phase 2 step 2)

`C = relu(A + B)` over 4 tiles of bfloat16, split across 3 cores. Full DAG
with one read-only core, one forwarding-compute core, one compute-and-write
core. Intermediate `T = A + B` flows core→core via NOC L1→L1 and **never
touches DRAM**.

```
DRAM ga, gb ──► (0,0) head ──pa,pb──► (0,1) middle ──pt──► (0,2) tail ──► DRAM gc
                read+mcast            T = A+B,             C = relu(T),
                                      forward              write
```

All three cores share the same kernel binaries; roles selected by `core_id`.

## Files

```
device/tanto/three_core_add_relu_reader.cpp   role-branched on core_id
device/tanto/three_core_add_relu_math.cpp     role-branched on core_id
device/tanto/three_core_add_relu_writer.cpp   role-branched on core_id
device/metal/                                  generated
host/three_core_add_relu.{hpp,cpp}             wires grid, pipes, semaphores
test/main.cpp                                  bf16-clean A/B; relu reference
```

## Kernel signatures (runtime args)

**reader:**
```
kernel(pipe<T> pa, pipe<T> pb, pipe<T> pt,
       semaphore sem_credit_ab, semaphore sem_data_ab,
       semaphore sem_credit_t,  semaphore sem_data_t,
       uint32 head_x, uint32 head_y,
       uint32 mid_x,  uint32 mid_y,
       uint32 num_blocks, uint32 block_tiles,
       uint32 core_id)
```

**writer:**
```
kernel(global<T> ga, global<T> gb, global<T> gc,
       pipe<T> pa, pipe<T> pb, pipe<T> pt, pipe<T> pt_local, pipe<T> pc,
       semaphore sem_credit_ab, semaphore sem_data_ab,
       semaphore sem_credit_t,  semaphore sem_data_t,
       uint32 ga_pos, uint32 gb_pos, uint32 gc_pos,
       uint32 mid_x,  uint32 mid_y,
       uint32 tail_x, uint32 tail_y,
       uint32 num_blocks, uint32 block_tiles,
       uint32 core_id)
```

**math:**
```
kernel(pipe<T> pa, pipe<T> pb, pipe<T> pt, pipe<T> pt_local, pipe<T> pc,
       uint32 num_blocks, uint32 block_tiles,
       uint32 core_id)
```

## Per-core role branches

| core_id | logical | reader                          | math                                                  | writer                                                   |
|---------|---------|---------------------------------|-------------------------------------------------------|----------------------------------------------------------|
| 0       | (0,0) head   | no-op                       | drain pa, pb                                          | DRAM read → mcast pa,pb to middle → push                  |
| 1       | (0,1) middle | wait+credit+receive pa, pb  | reserve pt_local, add pa+pb → pack pt_local, push; drain pt | wait pt_local → reserve pt → forward pt_local→pt to tail → push pt |
| 2       | (0,2) tail   | wait+credit+receive pt      | reserve pc, relu(pt) → pack pc, push pc               | drain pc → DRAM                                           |

## Pipes & semaphores

```
m_pa, m_pb : INPUT,    depth=2, frame_size=1   cross-core 0→1
m_pt       : INPUT,    depth=2, frame_size=1   cross-core 1→2
m_pt_local : INTERMED, depth=2, frame_size=1   per-core helper on middle (math→writer handoff)
m_pc       : OUTPUT,   depth=2, frame_size=1   used only on tail

m_sem_credit_ab, m_sem_data_ab : channel 0→1
m_sem_credit_t,  m_sem_data_t  : channel 1→2
```

All declared on the 1×3 grid `(0,0)..(0,2)`.

## Why `pt_local` exists

Middle's math (TRISC) computes the data; middle's writer (BRISC) is the only
kernel that can issue NOC writes to tail. They cannot sync via semaphores
because **TRISC has no semaphore primitives** (`tanto/src/device/metal/compute.h`
exports none). The handoff has to go through a CB.

`pt_local` is a `PipeKind::INTERMED` pipe that lives logically only on
middle (head and tail get it on their L1 too, since pipes are grid-scoped,
but they don't touch it). Math packs into pt_local; writer wait_fronts
pt_local; the CB itself is the math→writer sync.

The forward from middle to tail uses `pt_local.write(0, pt, 0, n, tail_x,
tail_y)`, which lowers to a point-to-point `noc_async_write` with source
`get_read_ptr(pt_local)` (the just-pushed slot — correct, no cursor drift)
and destination at remote `get_write_ptr(pt)`. The destination address
relies on **middle's pt write_ptr being aligned with tail's**, which is
maintained by both cores doing `reserve_back` + `push_back` on pt in
lockstep, once per iteration.

## Why middle's math drains pt locally

Middle's writer pushes pt every iteration but no one consumes pt locally on
middle. After `depth` pushes, the cb fills and the writer's `reserve_back pt`
deadlocks. Middle's math therefore runs `pt.wait_front; pt.pop_front` per
iteration as bookkeeping.

## Per-iteration timeline

```
HEAD                          MIDDLE                           TAIL
                              (reader)                         (reader)
                              pa.reserve; pb.reserve           pt.reserve
(writer)                      sem_credit_ab.inc(head)          sem_credit_t.inc(middle)
pa.reserve; pb.reserve
pa.read(ga); pb.read(gb)
read_barrier
sem_credit_ab.wait/set        sem_data_ab.wait/set
pa.write_mcast → middle
pb.write_mcast → middle
write_barrier
sem_data_ab.inc(middle)
pa.push; pb.push              pa.push; pb.push
                              (math) reserve pt_local
                                     wait_front pa, pb
                                     acc.add → pack pt_local
                                     pop pa, pb
                                     push pt_local
                              (writer) wait_front pt_local
                                       pt.reserve
                                       sem_credit_t.wait/set
                                       pt_local.write → tail's pt
                                       write_barrier
                                       sem_data_t.inc(tail)
                                       pop pt_local
                                       pt.push                 sem_data_t.wait/set
                                                               pt.push
                              (math) wait_front pt; pop pt     (math) reserve pc
                                                                      wait_front pt
                                                                      relu(pt) → pack pc
                                                                      pop pt
                                                                      push pc
                                                               (writer) wait_front pc
                                                                        pc.write → gc
                                                                        write_barrier
                                                                        pop pc
```

## How to verify

Recipe is in the top-level `README.md`. Expected output:

```
three_core_add_relu: N=4096
  ...
  max_diff=0  num_bad=0/4096
RESULT: PASS
```

The test (`test/main.cpp`) uses bf16-clean integer inputs in `[-8, 7]` so
that some sums are negative (relu zeroes them) and some positive (relu
passes through), exercising both branches.
