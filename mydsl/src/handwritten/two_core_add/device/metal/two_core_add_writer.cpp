
#include "tanto/dataflow.h"

#define T bfloat16

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

void kernel(Global ga, Global gb, Global gc, Pipe pa, Pipe pb, Pipe pc,
            Semaphore sem_credit, Semaphore sem_data, uint32 ga_pos,
            uint32 gb_pos, uint32 gc_pos, uint32 cons_x, uint32 cons_y,
            uint32 num_blocks, uint32 block_tiles, uint32 core_id) {
  uint32 block_items = block_tiles * 1024;
  if (core_id == 0) {
    // Producer: DRAM read into pa, pb; mcast to consumer; push_back.
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_reserve_back(pa.cb_id, pa.frame_size);
      cb_reserve_back(pb.cb_id, pb.frame_size);
      noc_async_read_global_dram(get_write_ptr(pa.cb_id) + (0 << 1), ga.addr,
                                 ga.log2_page_size, ga_pos << 1,
                                 block_items << 1);
      noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                                 gb.log2_page_size, gb_pos << 1,
                                 block_items << 1);
      noc_async_read_barrier();
      noc_semaphore_wait(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit.addr), 1);
      noc_semaphore_set(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit.addr), 0);
      noc_async_write_multicast(
          get_write_ptr(pa.cb_id) + (0 << 1),
          get_noc_multicast_addr(cons_x, cons_y, cons_x, cons_y,
                                 get_write_ptr(pa.cb_id) + (0 << 1)),
          block_items << 1, 1);
      noc_async_write_multicast(
          get_write_ptr(pb.cb_id) + (0 << 1),
          get_noc_multicast_addr(cons_x, cons_y, cons_x, cons_y,
                                 get_write_ptr(pb.cb_id) + (0 << 1)),
          block_items << 1, 1);
      noc_async_write_barrier();
      noc_semaphore_inc(get_noc_addr(cons_x, cons_y, sem_data.addr), 1);
      cb_push_back(pa.cb_id, pa.frame_size);
      cb_push_back(pb.cb_id, pb.frame_size);
      ga_pos += block_items;
      gb_pos += block_items;
    }
  } else {
    // Consumer: drain pc to DRAM.
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_wait_front(pc.cb_id, pc.frame_size);
      noc_async_write_global_dram(get_read_ptr(pc.cb_id) + (0 << 1), gc.addr,
                                  gc.log2_page_size, gc_pos << 1,
                                  block_items << 1);
      noc_async_write_barrier();
      cb_pop_front(pc.cb_id, pc.frame_size);
      gc_pos += block_items;
    }
  }
}
void kernel_main() {
  Global ga;
  ga.addr = get_arg_val<uint32>(0);
  ga.log2_page_size = get_arg_val<uint32>(1);
  Global gb;
  gb.addr = get_arg_val<uint32>(2);
  gb.log2_page_size = get_arg_val<uint32>(3);
  Global gc;
  gc.addr = get_arg_val<uint32>(4);
  gc.log2_page_size = get_arg_val<uint32>(5);
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(6);
  pa.frame_size = get_arg_val<uint32>(7);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(8);
  pb.frame_size = get_arg_val<uint32>(9);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(10);
  pc.frame_size = get_arg_val<uint32>(11);
  Semaphore sem_credit;
  sem_credit.addr = tanto_get_semaphore(get_arg_val<uint32>(12));
  Semaphore sem_data;
  sem_data.addr = tanto_get_semaphore(get_arg_val<uint32>(13));
  uint32 ga_pos = get_arg_val<uint32>(14);
  uint32 gb_pos = get_arg_val<uint32>(15);
  uint32 gc_pos = get_arg_val<uint32>(16);
  uint32 cons_x = get_arg_val<uint32>(17);
  uint32 cons_y = get_arg_val<uint32>(18);
  uint32 num_blocks = get_arg_val<uint32>(19);
  uint32 block_tiles = get_arg_val<uint32>(20);
  uint32 core_id = get_arg_val<uint32>(21);
  kernel(ga, gb, gc, pa, pb, pc, sem_credit, sem_data, ga_pos, gb_pos, gc_pos,
         cons_x, cons_y, num_blocks, block_tiles, core_id);
}

