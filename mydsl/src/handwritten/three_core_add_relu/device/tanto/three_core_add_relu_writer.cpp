// three_core_add_relu_writer.cpp — phase 2 step 2
//
// One Tanto writer, three roles selected by core_id:
//   core_id == 0 (head, core (0,0)):
//     Reserve pa, pb; DRAM-read A, B into the slot; wait middle's credit;
//     NOC mcast to middle (1 dest); signal data; push_back.
//   core_id == 1 (middle, core (0,1)):
//     Wait math's pt_local; reserve pt (cursor lockstep with tail);
//     wait tail's credit; NOC point-to-point write from pt_local read_ptr to
//     tail's pt write_ptr; signal data; pop pt_local; push pt.
//   core_id == 2 (tail, core (0,2)):
//     Drain pc to DRAM gc.
//
// Cross-core protocol: see ../../../doc/notes.md.

void kernel(
        global<T> ga,
        global<T> gb,
        global<T> gc,
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pt,
        pipe<T> pt_local,
        pipe<T> pc,
        semaphore sem_credit_ab,
        semaphore sem_data_ab,
        semaphore sem_credit_t,
        semaphore sem_data_t,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 gc_pos,
        uint32 mid_x,
        uint32 mid_y,
        uint32 tail_x,
        uint32 tail_y,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    uint32 block_items = block_tiles * 1024;
    if (core_id == 0) {
        // Head: DRAM read into pa, pb; mcast to middle; push_back.
        for (uint32 i = 0; i < num_blocks; i++) {
            pa.reserve_back();
            pb.reserve_back();
            pa.read(0, ga, ga_pos, block_items);
            pb.read(0, gb, gb_pos, block_items);
            read_barrier();
            sem_credit_ab.wait(1);
            sem_credit_ab.set(0);
            pa.write_mcast(
                0, pa, 0, block_items,
                mid_x, mid_y, mid_x, mid_y, 1);
            pb.write_mcast(
                0, pb, 0, block_items,
                mid_x, mid_y, mid_x, mid_y, 1);
            write_barrier();
            sem_data_ab.inc(mid_x, mid_y, 1);
            pa.push_back();
            pb.push_back();
            ga_pos += block_items;
            gb_pos += block_items;
        }
    } else if (core_id == 1) {
        // Middle: pt_local -> NOC -> tail's pt. Keep middle.pt cursor lockstep.
        for (uint32 i = 0; i < num_blocks; i++) {
            pt_local.wait_front();
            pt.reserve_back();
            sem_credit_t.wait(1);
            sem_credit_t.set(0);
            // pipe.write(src_offset, dst_pipe, dst_offset, count, x, y) lowers to
            // noc_async_write(get_read_ptr(self) + src_off,
            //                 get_noc_addr(x, y, get_write_ptr(dst) + dst_off),
            //                 count * itemsize)
            pt_local.write(0, pt, 0, block_items, tail_x, tail_y);
            write_barrier();
            sem_data_t.inc(tail_x, tail_y, 1);
            pt_local.pop_front();
            pt.push_back();
        }
    } else {
        // Tail: drain pc to DRAM.
        for (uint32 i = 0; i < num_blocks; i++) {
            pc.wait_front();
            pc.write(0, gc, gc_pos, block_items);
            write_barrier();
            pc.pop_front();
            gc_pos += block_items;
        }
    }
}
