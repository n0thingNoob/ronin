
#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// head_math.cpp — phase 2 step 3 (true MPMD)
//
// Head core's math (TRISC): drain pa, pb. Head's writer push_backs pa, pb
// every iteration but no other kernel pop_fronts on head; without the
// drain, after `depth` iterations the writer's reserve_back blocks
// (deadlock). Math is the cheapest place to do the bookkeeping.

void kernel(Pipe pa, Pipe pb, uint32 num_blocks, uint32 block_tiles) {
  for (uint32 i = 0; i < num_blocks; i++) {
    cb_wait_front(pa.cb_id, pa.frame_size);
    cb_wait_front(pb.cb_id, pb.frame_size);
    cb_pop_front(pa.cb_id, pa.frame_size);
    cb_pop_front(pb.cb_id, pb.frame_size);
  }
}
void MAIN {
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(0);
  pa.frame_size = get_arg_val<uint32>(1);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(2);
  pb.frame_size = get_arg_val<uint32>(3);
  uint32 num_blocks = get_arg_val<uint32>(4);
  uint32 block_tiles = get_arg_val<uint32>(5);
  tanto_compute_init();
  kernel(pa, pb, num_blocks, block_tiles);
}
} // namespace NAMESPACE

