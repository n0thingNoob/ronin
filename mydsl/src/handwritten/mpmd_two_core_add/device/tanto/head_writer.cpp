// head_writer.cpp — phase 2 step 3 (true MPMD)
//
// Head core's writer (BRISC): the producer. Reserves a slot in pa, pb,
// reads A,B from DRAM into the slot, waits for tail's credit, NOC mcasts
// the slot to tail (1 destination), signals data, push_back. Same shape
// as the producer branch of two_core_add_writer.cpp's core_id==0 path,
// but factored into its own kernel binary — this is what makes the demo
// "true MPMD".

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        semaphore sem_credit,
        semaphore sem_data,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 cons_x,
        uint32 cons_y,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        pa.reserve_back();
        pb.reserve_back();
        pa.read(0, ga, ga_pos, block_items);
        pb.read(0, gb, gb_pos, block_items);
        read_barrier();
        sem_credit.wait(1);
        sem_credit.set(0);
        pa.write_mcast(
            0, pa, 0, block_items,
            cons_x, cons_y, cons_x, cons_y, 1);
        pb.write_mcast(
            0, pb, 0, block_items,
            cons_x, cons_y, cons_x, cons_y, 1);
        write_barrier();
        sem_data.inc(cons_x, cons_y, 1);
        pa.push_back();
        pb.push_back();
        ga_pos += block_items;
        gb_pos += block_items;
    }
}
