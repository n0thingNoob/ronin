
#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// three_core_add_relu_math.cpp — phase 2 step 2
//
// One Tanto compute, three roles selected by core_id:
//   core_id == 0 (head): drain pa, pb. Head's writer keeps pushing them; if
//     no one pops, the pipe fills after `depth` iterations and writer's
//     reserve_back blocks. (Same as two_core_add producer drain.)
//   core_id == 1 (middle): T = pa + pb -> pt_local. Middle's writer consumes
//     pt_local and NOC-forwards to tail's pt. Middle also drains its own pt
//     locally (wait_front + pop_front) to keep cb cursors moving in lockstep
//     with tail's pt.
//   core_id == 2 (tail): C = relu(pa + pb) is split — pa+pb already done by
//     middle, tail just does relu(pt) -> pc.
//
// Why pt_local on middle: math (TRISC) cannot use semaphores, so handing data
// from math to writer (BRISC) needs a CB. pt_local is that CB.

void kernel(Pipe pa, Pipe pb, Pipe pt, Pipe pt_local, Pipe pc,
            uint32 num_blocks, uint32 block_tiles, uint32 core_id) {
  tanto_relu_init();
  if (core_id == 0) {
    // Head: drain pa, pb (head's writer pushes them; nothing else pops).
    for (uint32 i = 0; i < num_blocks; i++) {
      cb_wait_front(pa.cb_id, pa.frame_size);
      cb_wait_front(pb.cb_id, pb.frame_size);
      cb_pop_front(pa.cb_id, pa.frame_size);
      cb_pop_front(pb.cb_id, pb.frame_size);
    }
  } else if (core_id == 1) {
    tanto_unpack_binary_init(pa.cb_id, pb.cb_id);
    tanto_add_init();
    tanto_pack_init(pt_local.cb_id);
    // Middle: pa + pb -> pt_local; drain pt to keep cursor synced.
    for (uint32 block = 0; block < num_blocks; block++) {
      cb_reserve_back(pt_local.cb_id, pt_local.frame_size);
      cb_wait_front(pa.cb_id, pa.frame_size);
      cb_wait_front(pb.cb_id, pb.frame_size);
      tile_regs_acquire();
      tile_regs_wait();
      for (uint32 i = 0; i < block_tiles; i++) {
        add_tiles(pa.cb_id, pb.cb_id, i, i, i);
      }
      for (uint32 i = 0; i < block_tiles; i++) {
        pack_tile(i, pt_local.cb_id);
      }
      cb_pop_front(pa.cb_id, pa.frame_size);
      cb_pop_front(pb.cb_id, pb.frame_size);
      cb_push_back(pt_local.cb_id, pt_local.frame_size);
      // Drain pt locally (writer push_backs pt to advance write_ptr in
      // lockstep with tail; reading-side must also pop or the cb fills).
      cb_wait_front(pt.cb_id, pt.frame_size);
      cb_pop_front(pt.cb_id, pt.frame_size);
      tile_regs_commit();
      tile_regs_release();
    }
  } else {
    tanto_unpack_unary_init(pt.cb_id);
    tanto_copy_init();
    tanto_pack_init(pc.cb_id);
    // Tail: relu(pt) -> pc.
    for (uint32 block = 0; block < num_blocks; block++) {
      cb_reserve_back(pc.cb_id, pc.frame_size);
      cb_wait_front(pt.cb_id, pt.frame_size);
      for (uint32 i = 0; i < block_tiles; i++) {
        tile_regs_acquire();
        tile_regs_wait();
        copy_tile(pt.cb_id, i, 0);
        relu_tile(0);
        pack_tile(0, pc.cb_id);
        tile_regs_commit();
        tile_regs_release();
      }
      cb_pop_front(pt.cb_id, pt.frame_size);
      cb_push_back(pc.cb_id, pc.frame_size);
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
  Pipe pt;
  pt.cb_id = get_arg_val<uint32>(4);
  pt.frame_size = get_arg_val<uint32>(5);
  Pipe pt_local;
  pt_local.cb_id = get_arg_val<uint32>(6);
  pt_local.frame_size = get_arg_val<uint32>(7);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(8);
  pc.frame_size = get_arg_val<uint32>(9);
  uint32 num_blocks = get_arg_val<uint32>(10);
  uint32 block_tiles = get_arg_val<uint32>(11);
  uint32 core_id = get_arg_val<uint32>(12);
  tanto_compute_init();
  kernel(pa, pb, pt, pt_local, pc, num_blocks, block_tiles, core_id);
}
} // namespace NAMESPACE

