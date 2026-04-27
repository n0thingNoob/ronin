// tail_reader.cpp — phase 2 step 3 (true MPMD)
//
// Tail core's reader (NCRISC): the consumer's handshake. Reserves a slot
// in pa, pb, increments head's sem_credit, waits for head's sem_data,
// then push_back (NOC fill arrived in the slot during the wait).

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        semaphore sem_credit,
        semaphore sem_data,
        uint32 prod_x,
        uint32 prod_y,
        uint32 num_blocks,
        uint32 block_tiles) {
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
