// two_core_add_math.cpp — phase 2 step 1
//
// core_id == 0 (producer): drain producer-local pa, pb. The producer's writer
// fills + mcasts + push_backs pa, pb each iteration; without a matching
// pop_front the producer's pipes fill up after `depth` iterations and the
// writer's reserve_back blocks. Math (TRISC) doing wait_front/pop_front is
// the cheapest drain.
// core_id == 1 (consumer): T = pa + pb -> pc.

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    if (core_id == 0) {
        // Producer: drain pa, pb (writer keeps pushing them).
        for (uint32 i = 0; i < num_blocks; i++) {
            pa.wait_front();
            pb.wait_front();
            pa.pop_front();
            pb.pop_front();
        }
    } else {
        for (uint32 block = 0; block < num_blocks; block++) {
            pc.reserve_back();
            pa.wait_front();
            pb.wait_front();
            math<T> acc;
            for (uint32 i = 0; i < block_tiles; i++) {
                acc.add(pa, pb, i, i, i);
            }
            for (uint32 i = 0; i < block_tiles; i++) {
                acc.pack(i, pc);
            }
            pa.pop_front();
            pb.pop_front();
            pc.push_back();
        }
    }
}
