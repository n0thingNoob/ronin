// scalar_add_reader.cpp — phase 1 v0
//
// Derived from algo/src/basic/device/tanto/eltwise_binary_reader.cpp.
//
// v0 strategy: scalar_add is implemented as eltwise_add against a B tensor
// that the host fills with the constant 3.0. So this reader is *unchanged*
// from eltwise_binary's reader: it streams ga and gb from DRAM into pa and pb.

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        pa.reserve_back();
        pb.reserve_back();
        pa.read(0, ga, ga_pos, block_items);
        pb.read(0, gb, gb_pos, block_items);
        read_barrier();
        pa.push_back();
        pb.push_back();
        ga_pos += block_items;
        gb_pos += block_items;
    }
}
