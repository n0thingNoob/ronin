
#include "tanto/dataflow.h"

#define T bfloat16

// three_core_add_relu_reader.cpp — phase 2 step 2
//
// One Tanto reader, three roles selected by core_id:
//   core_id == 0 (head, core (0,0)):
//     No-op. Head's writer does both DRAM read and NOC mcast (same reason as
//     two_core_add: pipe.write_mcast lowers to get_write_ptr-based source).
//   core_id == 1 (middle, core (0,1)):
//     Consumer of pa, pb from head. Reserve, signal credit, wait data,
//     push_back.
//   core_id == 2 (tail, core (0,2)):
//     Consumer of pt from middle. Reserve, signal credit, wait data, push_back.
//
// Cross-core protocol: see ../../../doc/notes.md "Cross-core pipe protocol".

void kernel(Pipe pa, Pipe pb, Pipe pt, Semaphore sem_credit_ab,
            Semaphore sem_data_ab, Semaphore sem_credit_t, Semaphore sem_data_t,
            uint32 head_x, uint32 head_y, uint32 mid_x, uint32 mid_y,
            uint32 num_blocks, uint32 block_tiles, uint32 core_id) {
  if (core_id == 1) {
    // Middle: consumer of pa, pb from head.
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_reserve_back(pa.cb_id, pa.frame_size);
      cb_reserve_back(pb.cb_id, pb.frame_size);
      noc_semaphore_inc(get_noc_addr(head_x, head_y, sem_credit_ab.addr), 1);
      noc_semaphore_wait(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data_ab.addr), 1);
      noc_semaphore_set(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data_ab.addr), 0);
      cb_push_back(pa.cb_id, pa.frame_size);
      cb_push_back(pb.cb_id, pb.frame_size);
    }
  } else if (core_id == 2) {
    // Tail: consumer of pt from middle.
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_reserve_back(pt.cb_id, pt.frame_size);
      noc_semaphore_inc(get_noc_addr(mid_x, mid_y, sem_credit_t.addr), 1);
      noc_semaphore_wait(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data_t.addr), 1);
      noc_semaphore_set(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_data_t.addr), 0);
      cb_push_back(pt.cb_id, pt.frame_size);
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
  Pipe pt;
  pt.cb_id = get_arg_val<uint32>(4);
  pt.frame_size = get_arg_val<uint32>(5);
  Semaphore sem_credit_ab;
  sem_credit_ab.addr = tanto_get_semaphore(get_arg_val<uint32>(6));
  Semaphore sem_data_ab;
  sem_data_ab.addr = tanto_get_semaphore(get_arg_val<uint32>(7));
  Semaphore sem_credit_t;
  sem_credit_t.addr = tanto_get_semaphore(get_arg_val<uint32>(8));
  Semaphore sem_data_t;
  sem_data_t.addr = tanto_get_semaphore(get_arg_val<uint32>(9));
  uint32 head_x = get_arg_val<uint32>(10);
  uint32 head_y = get_arg_val<uint32>(11);
  uint32 mid_x = get_arg_val<uint32>(12);
  uint32 mid_y = get_arg_val<uint32>(13);
  uint32 num_blocks = get_arg_val<uint32>(14);
  uint32 block_tiles = get_arg_val<uint32>(15);
  uint32 core_id = get_arg_val<uint32>(16);
  kernel(pa, pb, pt, sem_credit_ab, sem_data_ab, sem_credit_t, sem_data_t,
         head_x, head_y, mid_x, mid_y, num_blocks, block_tiles, core_id);
}

