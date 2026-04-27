// three_core_add_relu_reader.cpp — phase 2 step 2
//
// One Tanto reader, three roles selected by core_id:
//   core_id == 0 (head, core (0,0)):
//     No-op. Head's writer does both DRAM read and NOC mcast (same reason as
//     two_core_add: pipe.write_mcast lowers to get_write_ptr-based source).
//   core_id == 1 (middle, core (0,1)):
//     Consumer of pa, pb from head. Reserve, signal credit, wait data, push_back.
//   core_id == 2 (tail, core (0,2)):
//     Consumer of pt from middle. Reserve, signal credit, wait data, push_back.
//
// Cross-core protocol: see ../../../doc/notes.md "Cross-core pipe protocol".

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pt,
        semaphore sem_credit_ab,
        semaphore sem_data_ab,
        semaphore sem_credit_t,
        semaphore sem_data_t,
        uint32 head_x,
        uint32 head_y,
        uint32 mid_x,
        uint32 mid_y,
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 core_id) {
    if (core_id == 1) {
        // Middle: consumer of pa, pb from head.
        for (uint32 i = 0; i < num_blocks; i++) {
            pa.reserve_back();
            pb.reserve_back();
            sem_credit_ab.inc(head_x, head_y, 1);
            sem_data_ab.wait(1);
            sem_data_ab.set(0);
            pa.push_back();
            pb.push_back();
        }
    } else if (core_id == 2) {
        // Tail: consumer of pt from middle.
        for (uint32 i = 0; i < num_blocks; i++) {
            pt.reserve_back();
            sem_credit_t.inc(mid_x, mid_y, 1);
            sem_data_t.wait(1);
            sem_data_t.set(0);
            pt.push_back();
        }
    }
    // core_id == 0: no-op.
}
