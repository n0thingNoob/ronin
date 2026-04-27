// scalar_add_math.cpp — phase 1 v0
//
// Derived from algo/src/basic/device/tanto/eltwise_binary_math.cpp.
//
// v0 strategy: hardcode add. The eltwise_binary version had a param<>
// switching between add/sub/mul; we drop that — scalar_add is always add.
// This means front.sh can call --mode=compute without any -P0 flag.

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
