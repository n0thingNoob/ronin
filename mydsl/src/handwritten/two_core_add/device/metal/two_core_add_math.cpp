
#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// two_core_add_math.cpp — phase 2 step 1
//
// core_id == 0 (producer): drain producer-local pa, pb. The producer's writer
// fills + mcasts + push_backs pa, pb each iteration; without a matching
// pop_front the producer's pipes fill up after `depth` iterations and the
// writer's reserve_back blocks. Math (TRISC) doing wait_front/pop_front is
// the cheapest drain.
// core_id == 1 (consumer): T = pa + pb -> pc.

void kernel(Pipe pa, Pipe pb, Pipe pc, uint32 num_blocks, uint32 block_tiles,
            uint32 core_id) {
  tanto_unpack_binary_init(pa.cb_id, pb.cb_id);
  tanto_add_init();
  tanto_pack_init(pc.cb_id);
  if (core_id == 0) {
    // Producer: drain pa, pb (writer keeps pushing them).
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_wait_front(pa.cb_id, pa.frame_size);
      cb_wait_front(pb.cb_id, pb.frame_size);
      cb_pop_front(pa.cb_id, pa.frame_size);
      cb_pop_front(pb.cb_id, pb.frame_size);
    }
  } else {
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
  uint32 core_id = get_arg_val<uint32>(8);
  tanto_compute_init();
  kernel(pa, pb, pc, num_blocks, block_tiles, core_id);
}
} // namespace NAMESPACE

