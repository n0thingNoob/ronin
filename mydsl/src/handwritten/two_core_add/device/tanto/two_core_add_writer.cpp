// two_core_add_writer.cpp — phase 2 step 1
//
// core_id == 0 (producer): reserve pa, pb slot; DRAM-read A,B into the slot;
// wait for consumer's credit; NOC mcast the slot to consumer's pa, pb (1 dest);
// signal data; push_back. Doing reserve+fill+mcast+push in one kernel is
// required because `pipe.write_mcast` uses get_write_ptr for source (matches
// the just-filled, not-yet-pushed slot).
// core_id == 1 (consumer): drain pc to DRAM gc.
//
// Cross-core protocol: see ../../../doc/notes.md.

void kernel(
        global<T> ga,
        global<T> gb,
        global<T> gc,
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        semaphore sem_credit,
        semaphore sem_data,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 gc_pos,
        uint32 cons_x,
        uint32 cons_y,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    uint32 block_items = block_tiles * 1024;
    if (core_id == 0) {
        // Producer: DRAM read into pa, pb; mcast to consumer; push_back.
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
    } else {
        // Consumer: drain pc to DRAM.
        for (uint32 i = 0; i < num_blocks; i++) {
            pc.wait_front();
            pc.write(0, gc, gc_pos, block_items);
            write_barrier();
            pc.pop_front();
            gc_pos += block_items;
        }
    }
}
