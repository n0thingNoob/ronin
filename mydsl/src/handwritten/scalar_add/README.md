# scalar_add (phase 1)

`C[i] = A[i] + 3.0`, bfloat16, 4 tiles, single core, DRAM in / DRAM out.

Goal: hand-write the 4 source files below until they run on Jitte and PASS
against a CPU reference. No skipping, no codegen, no shortcuts.

## Files to write (in this order)

1. `device/tanto/scalar_add_reader.cpp` — reads `ga` from DRAM into pipe `pa`
2. `device/tanto/scalar_add_math.cpp` — pops `pa`, adds constant, pushes `pc`
3. `device/tanto/scalar_add_writer.cpp` — pops `pc`, writes `gc` to DRAM
4. `host/scalar_add.cpp` (+ `.hpp`) — allocates buffers, configures pipes,
   creates kernels, sets runtime args, enqueues
5. `test/main.cpp` — fills A with known data, calls host, compares C against
   a CPU reference

## Reference

`algo/src/basic/device/tanto/eltwise_binary_{reader,math,writer}.cpp` and
`algo/src/basic/host/tanto/eltwise_binary.cpp` are the closest example. The
shape is: same minus pipe `pb` (no second input).

## Open design point

The constant `3.0` can be passed two ways. Pick one before writing the math
kernel:

- **runtime arg** (`uint32 scalar` in kernel signature, host packs `bfloat16`
  bits into uint32): flexible, no need to recompile to change value
- **`param<uint32>` compile-time constant**: cleaner, value baked in at
  translation, but requires `-P0=...` flag in `front.sh` and host loads a
  specifically-named metal file

Tradeoff to think about: when the DSL eventually emits this kernel, which
mechanism is easier to codegen?

## Build

After the 4 files are written:
```
cd ../../../prj/handwritten/scalar_add/
./front.sh                # tanto translates device/tanto → device/metal
cd ../../../jitte/prj/handwritten/scalar_add/
./deploy_jitte.sh         # copies device/ into JITTE_HOME
./build_test_tanto.sh     # compiles test binary against jitte libs
```

Run the produced binary; it should print PASS.

## Done when

- binary exists and runs on Jitte
- output matches CPU reference within bfloat16 tolerance
- you can answer (without checking notes) what each of the 7 alignment axes
  in `doc/notes.md` is set to in your code
