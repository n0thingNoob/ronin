# scalar_add per-core spec (mydsl phase 2.5)
#
# Mirrors the handwritten kernel under
# src/handwritten/scalar_add/device/tanto/. Generator emits the same 3
# .cpp files into src/codegen/out/scalar_add/device/tanto/.
#
# The host wrapper (src/handwritten/scalar_add/host/scalar_add.{cpp,hpp})
# is reused unchanged — host generation is out of scope for v0.

spec = {
    "name": "scalar_add",
    "T": "bfloat16",
    "inputs": [
        {"name": "a", "pipe": "pa"},
        {"name": "b", "pipe": "pb"},
    ],
    "outputs": [
        {"name": "c", "pipe": "pc"},
    ],
    "compute": "c = a + b",
}
