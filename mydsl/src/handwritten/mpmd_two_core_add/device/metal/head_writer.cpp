
#include "tanto/dataflow.h"

#define T bfloat16

// head_writer.cpp — phase 2 step 3 (true MPMD)
//
// Head core's writer (BRISC): the producer. Reserves a slot in pa, pb,
// reads A,B from DRAM into the slot, waits for tail's credit, NOC mcasts
// the slot to tail (1 destination), signals data, push_back. Same shape
// as the producer branch of two_core_add_writer.cpp's core_id==0 path,
// but factored into its own kernel binary — this is what makes the demo
// "true MPMD".

void kernel(Global ga, Global gb, Pipe pa, Pipe pb, Semaphore sem_credit,
            Semaphore sem_data, uint32 ga_pos, uint32 gb_pos, uint32 cons_x,
            uint32 cons_y, uint32 num_blocks, uint32 block_tiles) {
  uint32 block_items = block_tiles * 1024;
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
}
void kernel_main() {
  Global ga;
  ga.addr = get_arg_val<uint32>(0);
  ga.log2_page_size = get_arg_val<uint32>(1);
  Global gb;
  gb.addr = get_arg_val<uint32>(2);
  gb.log2_page_size = get_arg_val<uint32>(3);
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(4);
  pa.frame_size = get_arg_val<uint32>(5);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(6);
  pb.frame_size = get_arg_val<uint32>(7);
  Semaphore sem_credit;
  sem_credit.addr = tanto_get_semaphore(get_arg_val<uint32>(8));
  Semaphore sem_data;
  sem_data.addr = tanto_get_semaphore(get_arg_val<uint32>(9));
  uint32 ga_pos = get_arg_val<uint32>(10);
  uint32 gb_pos = get_arg_val<uint32>(11);
  uint32 cons_x = get_arg_val<uint32>(12);
  uint32 cons_y = get_arg_val<uint32>(13);
  uint32 num_blocks = get_arg_val<uint32>(14);
  uint32 block_tiles = get_arg_val<uint32>(15);
  kernel(ga, gb, pa, pb, sem_credit, sem_data, ga_pos, gb_pos, cons_x, cons_y,
         num_blocks, block_tiles);
}

