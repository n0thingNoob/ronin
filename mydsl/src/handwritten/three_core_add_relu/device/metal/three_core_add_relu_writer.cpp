
#include "tanto/dataflow.h"

#define T bfloat16

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

void kernel(Global ga, Global gb, Global gc, Pipe pa, Pipe pb, Pipe pt,
            Pipe pt_local, Pipe pc, Semaphore sem_credit_ab,
            Semaphore sem_data_ab, Semaphore sem_credit_t, Semaphore sem_data_t,
            uint32 ga_pos, uint32 gb_pos, uint32 gc_pos, uint32 mid_x,
            uint32 mid_y, uint32 tail_x, uint32 tail_y, uint32 num_blocks,
            uint32 block_tiles, uint32 core_id) {
  uint32 block_items = block_tiles * 1024;
  if (core_id == 0) {
    // Head: DRAM read into pa, pb; mcast to middle; push_back.
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
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit_ab.addr),
          1);
      noc_semaphore_set(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit_ab.addr),
          0);
      noc_async_write_multicast(
          get_write_ptr(pa.cb_id) + (0 << 1),
          get_noc_multicast_addr(mid_x, mid_y, mid_x, mid_y,
                                 get_write_ptr(pa.cb_id) + (0 << 1)),
          block_items << 1, 1);
      noc_async_write_multicast(
          get_write_ptr(pb.cb_id) + (0 << 1),
          get_noc_multicast_addr(mid_x, mid_y, mid_x, mid_y,
                                 get_write_ptr(pb.cb_id) + (0 << 1)),
          block_items << 1, 1);
      noc_async_write_barrier();
      noc_semaphore_inc(get_noc_addr(mid_x, mid_y, sem_data_ab.addr), 1);
      cb_push_back(pa.cb_id, pa.frame_size);
      cb_push_back(pb.cb_id, pb.frame_size);
      ga_pos += block_items;
      gb_pos += block_items;
    }
  } else if (core_id == 1) {
    // Middle: pt_local -> NOC -> tail's pt. Keep middle.pt cursor lockstep.
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_wait_front(pt_local.cb_id, pt_local.frame_size);
      cb_reserve_back(pt.cb_id, pt.frame_size);
      noc_semaphore_wait(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit_t.addr),
          1);
      noc_semaphore_set(
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_credit_t.addr),
          0);
      // pipe.write(src_offset, dst_pipe, dst_offset, count, x, y) lowers to
      // noc_async_write(get_read_ptr(self) + src_off,
      //                 get_noc_addr(x, y, get_write_ptr(dst) + dst_off),
      //                 count * itemsize)
      noc_async_write(
          get_read_ptr(pt_local.cb_id) + (0 << 1),
          get_noc_addr(tail_x, tail_y, get_write_ptr(pt.cb_id) + (0 << 1)),
          block_items << 1);
      noc_async_write_barrier();
      noc_semaphore_inc(get_noc_addr(tail_x, tail_y, sem_data_t.addr), 1);
      cb_pop_front(pt_local.cb_id, pt_local.frame_size);
      cb_push_back(pt.cb_id, pt.frame_size);
    }
  } else {
    // Tail: drain pc to DRAM.
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
  Pipe pt;
  pt.cb_id = get_arg_val<uint32>(10);
  pt.frame_size = get_arg_val<uint32>(11);
  Pipe pt_local;
  pt_local.cb_id = get_arg_val<uint32>(12);
  pt_local.frame_size = get_arg_val<uint32>(13);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(14);
  pc.frame_size = get_arg_val<uint32>(15);
  Semaphore sem_credit_ab;
  sem_credit_ab.addr = tanto_get_semaphore(get_arg_val<uint32>(16));
  Semaphore sem_data_ab;
  sem_data_ab.addr = tanto_get_semaphore(get_arg_val<uint32>(17));
  Semaphore sem_credit_t;
  sem_credit_t.addr = tanto_get_semaphore(get_arg_val<uint32>(18));
  Semaphore sem_data_t;
  sem_data_t.addr = tanto_get_semaphore(get_arg_val<uint32>(19));
  uint32 ga_pos = get_arg_val<uint32>(20);
  uint32 gb_pos = get_arg_val<uint32>(21);
  uint32 gc_pos = get_arg_val<uint32>(22);
  uint32 mid_x = get_arg_val<uint32>(23);
  uint32 mid_y = get_arg_val<uint32>(24);
  uint32 tail_x = get_arg_val<uint32>(25);
  uint32 tail_y = get_arg_val<uint32>(26);
  uint32 num_blocks = get_arg_val<uint32>(27);
  uint32 block_tiles = get_arg_val<uint32>(28);
  uint32 core_id = get_arg_val<uint32>(29);
  kernel(ga, gb, gc, pa, pb, pt, pt_local, pc, sem_credit_ab, sem_data_ab,
         sem_credit_t, sem_data_t, ga_pos, gb_pos, gc_pos, mid_x, mid_y, tail_x,
         tail_y, num_blocks, block_tiles, core_id);
}

