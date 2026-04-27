// two_core_add_reader.cpp — phase 2 step 1
//
// One Tanto reader, two roles selected by core_id:
//   core_id == 0 (producer, core (0,0)):
//     No-op. The producer's writer does both the DRAM read of A/B and the
//     NOC mcast to the consumer (this matches the dw_spatial pattern, since
//     `pipe.write_mcast(0, pipe, 0, n, ...)` lowers to get_write_ptr-based
//     source; we must fill the slot, mcast it, then push_back — all on the
//     same RISC).
//   core_id == 1 (consumer, core (0,1)):
//     Reserve a slot, signal credit to producer, wait for NOC fill, push_back.
//
// Cross-core protocol: see ../../../doc/notes.md "Cross-core pipe protocol".

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        semaphore sem_credit,
        semaphore sem_data,
        uint32 prod_x,
        uint32 prod_y,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    if (core_id == 1) {
        // Consumer: reserve a slot, signal credit, wait for data.
        for (uint32 i = 0; i < num_blocks; i++) {
            pa.reserve_back();
            pb.reserve_back();
            sem_credit.inc(prod_x, prod_y, 1);
            sem_data.wait(1);
            sem_data.set(0);
            pa.push_back();
            pb.push_back();
        }
    }
    // core_id == 0: no-op.
}
