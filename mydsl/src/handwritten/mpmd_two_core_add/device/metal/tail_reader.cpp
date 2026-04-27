
#include "tanto/dataflow.h"

#define T bfloat16

// tail_reader.cpp — phase 2 step 3 (true MPMD)
//
// Tail core's reader (NCRISC): the consumer's handshake. Reserves a slot
// in pa, pb, increments head's sem_credit, waits for head's sem_data,
// then push_back (NOC fill arrived in the slot during the wait).

void kernel(Pipe pa, Pipe pb, Semaphore sem_credit, Semaphore sem_data,
            uint32 prod_x, uint32 prod_y, uint32 num_blocks,
            uint32 block_tiles) {
  for (uint32 i = 0; i < num_blocks; i++) {
    cb_reserve_back(pa.cb_id, pa.frame_size);
    cb_reserve_back(pb.cb_id, pb.frame_size);
    noc_semaphore_inc(get_noc_addr(prod_x, prod_y, sem_credit.addr), 1);
    noc_semaphore_wait(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data.addr), 1);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data.addr), 0);
    cb_push_back(pa.cb_id, pa.frame_size);
    cb_push_back(pb.cb_id, pb.frame_size);
  }
}
void kernel_main() {
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(0);
  pa.frame_size = get_arg_val<uint32>(1);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(2);
  pb.frame_size = get_arg_val<uint32>(3);
  Semaphore sem_credit;
  sem_credit.addr = tanto_get_semaphore(get_arg_val<uint32>(4));
  Semaphore sem_data;
  sem_data.addr = tanto_get_semaphore(get_arg_val<uint32>(5));
  uint32 prod_x = get_arg_val<uint32>(6);
  uint32 prod_y = get_arg_val<uint32>(7);
  uint32 num_blocks = get_arg_val<uint32>(8);
  uint32 block_tiles = get_arg_val<uint32>(9);
  kernel(pa, pb, sem_credit, sem_data, prod_x, prod_y, num_blocks, block_tiles);
}

