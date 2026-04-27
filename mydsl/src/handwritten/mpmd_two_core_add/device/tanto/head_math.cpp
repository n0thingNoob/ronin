// head_math.cpp — phase 2 step 3 (true MPMD)
//
// Head core's math (TRISC): drain pa, pb. Head's writer push_backs pa, pb
// every iteration but no other kernel pop_fronts on head; without the
// drain, after `depth` iterations the writer's reserve_back blocks
// (deadlock). Math is the cheapest place to do the bookkeeping.

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        uint32 num_blocks,
        uint32 block_tiles) {
    for (uint32 i = 0; i < num_blocks; i++) {
        pa.wait_front();
        pb.wait_front();
        pa.pop_front();
        pb.pop_front();
    }
}
