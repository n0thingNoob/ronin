// tail_math.cpp — phase 2 step 3 (true MPMD)
//
// Tail core's math (TRISC): pc = pa + pb (per-tile reduce inside a block).
// Same shape as the consumer branch of two_core_add_math.cpp's core_id==1
// path, factored into its own kernel binary.

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 num_blocks,
        uint32 block_tiles) {
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
