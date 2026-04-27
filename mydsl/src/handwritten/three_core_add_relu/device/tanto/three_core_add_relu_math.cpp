// three_core_add_relu_math.cpp — phase 2 step 2
//
// One Tanto compute, three roles selected by core_id:
//   core_id == 0 (head): drain pa, pb. Head's writer keeps pushing them; if
//     no one pops, the pipe fills after `depth` iterations and writer's
//     reserve_back blocks. (Same as two_core_add producer drain.)
//   core_id == 1 (middle): T = pa + pb -> pt_local. Middle's writer consumes
//     pt_local and NOC-forwards to tail's pt. Middle also drains its own pt
//     locally (wait_front + pop_front) to keep cb cursors moving in lockstep
//     with tail's pt.
//   core_id == 2 (tail): C = relu(pa + pb) is split — pa+pb already done by
//     middle, tail just does relu(pt) -> pc.
//
// Why pt_local on middle: math (TRISC) cannot use semaphores, so handing data
// from math to writer (BRISC) needs a CB. pt_local is that CB.

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pt,
        pipe<T> pt_local,
        pipe<T> pc,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    if (core_id == 0) {
        // Head: drain pa, pb (head's writer pushes them; nothing else pops).
        for (uint32 i = 0; i < num_blocks; i++) {
            pa.wait_front();
            pb.wait_front();
            pa.pop_front();
            pb.pop_front();
        }
    } else if (core_id == 1) {
        // Middle: pa + pb -> pt_local; drain pt to keep cursor synced.
        for (uint32 block = 0; block < num_blocks; block++) {
            pt_local.reserve_back();
            pa.wait_front();
            pb.wait_front();
            math<T> acc;
            for (uint32 i = 0; i < block_tiles; i++) {
                acc.add(pa, pb, i, i, i);
            }
            for (uint32 i = 0; i < block_tiles; i++) {
                acc.pack(i, pt_local);
            }
            pa.pop_front();
            pb.pop_front();
            pt_local.push_back();
            // Drain pt locally (writer push_backs pt to advance write_ptr in
            // lockstep with tail; reading-side must also pop or the cb fills).
            pt.wait_front();
            pt.pop_front();
        }
    } else {
        // Tail: relu(pt) -> pc.
        for (uint32 block = 0; block < num_blocks; block++) {
            pc.reserve_back();
            pt.wait_front();
            for (uint32 i = 0; i < block_tiles; i++) {
                math<T> acc;
                acc.copy(pt, i, 0);
                acc.relu(0);
                acc.pack(0, pc);
            }
            pt.pop_front();
            pc.push_back();
        }
    }
}
