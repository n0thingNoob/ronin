
#include "tanto/dataflow.h"

#define T bfloat16

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

void kernel(Pipe pa, Pipe pb, Semaphore sem_credit, Semaphore sem_data,
            uint32 prod_x, uint32 prod_y, uint32 num_blocks, uint32 block_tiles,
            uint32 core_id) {
  if (core_id == 1) {
    // Consumer: reserve a slot, signal credit, wait for data.
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
  // core_id == 0: no-op.
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
  uint32 core_id = get_arg_val<uint32>(10);
  kernel(pa, pb, sem_credit, sem_data, prod_x, prod_y, num_blocks, block_tiles,
         core_id);
}

