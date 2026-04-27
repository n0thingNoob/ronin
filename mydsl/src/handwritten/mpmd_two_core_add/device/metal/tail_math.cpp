
#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// tail_math.cpp — phase 2 step 3 (true MPMD)
//
// Tail core's math (TRISC): pc = pa + pb (per-tile reduce inside a block).
// Same shape as the consumer branch of two_core_add_math.cpp's core_id==1
// path, factored into its own kernel binary.

void kernel(Pipe pa, Pipe pb, Pipe pc, uint32 num_blocks, uint32 block_tiles) {
  tanto_unpack_binary_init(pa.cb_id, pb.cb_id);
  tanto_add_init();
  tanto_pack_init(pc.cb_id);
  for (uint32 block = 0; block < num_blocks; block++) {
    cb_reserve_back(pc.cb_id, pc.frame_size);
    cb_wait_front(pa.cb_id, pa.frame_size);
    cb_wait_front(pb.cb_id, pb.frame_size);
    tile_regs_acquire();
    tile_regs_wait();
    for (uint32 i = 0; i < block_tiles; i++) {
      add_tiles(pa.cb_id, pb.cb_id, i, i, i);
    }
    for (uint32 i = 0; i < block_tiles; i++) {
      pack_tile(i, pc.cb_id);
    }
    cb_pop_front(pa.cb_id, pa.frame_size);
    cb_pop_front(pb.cb_id, pb.frame_size);
    cb_push_back(pc.cb_id, pc.frame_size);
    tile_regs_commit();
    tile_regs_release();
  }
}
void MAIN {
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(0);
  pa.frame_size = get_arg_val<uint32>(1);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(2);
  pb.frame_size = get_arg_val<uint32>(3);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(4);
  pc.frame_size = get_arg_val<uint32>(5);
  uint32 num_blocks = get_arg_val<uint32>(6);
  uint32 block_tiles = get_arg_val<uint32>(7);
  tanto_compute_init();
  kernel(pa, pb, pc, num_blocks, block_tiles);
}
} // namespace NAMESPACE

