# codegen (phase 2.5: per-core kernel generator)

Collapses the 3-file Tanto kernel boilerplate (reader / math / writer) into
**one Python spec per core**. Generated kernels match the layout under
`../handwritten/<name>/device/tanto/`. Tanto's frontend then translates them
to TT-Metalium just like handwritten kernels.

## Scope (v0)

- Single core only. Cross-core wiring (NOC L1→L1) is phase 2 / 3 work — see
  `../handwritten/two_core_add_relu/`.
- Exactly one output pipe.
- Compute is a literal string `OUT = A OP B` with `OP` in `{+, -, *}`.
- Host wrapper is **not** generated. v0 reuses the handwritten host wrapper
  unchanged; the generated kernels deploy into the same JITTE_HOME slot.

Anything outside that envelope: error out. Phase 3 will replace the literal
string with a real parser and start generating the host program.

## Spec format

```python
# src/codegen/specs/<name>.py
spec = {
    "name": "scalar_add",
    "T": "bfloat16",
    "inputs":  [{"name": "a", "pipe": "pa"},
                {"name": "b", "pipe": "pb"}],
    "outputs": [{"name": "c", "pipe": "pc"}],
    "compute": "c = a + b",
}
```

## Layout

```
codegen/
├── kernel_gen.py        entrypoint: python3 kernel_gen.py specs/<name>.py
├── templates/
│   ├── reader.cpp.tmpl
│   ├── math.cpp.tmpl
│   └── writer.cpp.tmpl
├── specs/
│   └── scalar_add.py
└── out/                 generated, gitignore-friendly
    └── <name>/device/tanto/{<name>_reader,<name>_math,<name>_writer}.cpp
```

## Running

```
# generate only
prj/codegen/scalar_add/gen.sh

# end-to-end: generate -> diff vs handwritten -> translate -> deploy -> build -> run
prj/codegen/scalar_add/verify.sh
```

`verify.sh` deploys generated kernels on top of the handwritten JITTE_HOME
slot and reuses the handwritten host wrapper. It is intentional that the
generated bodies match the handwritten ones byte-for-byte (only header
comments differ); the diff in step 2 of `verify.sh` is the visual proof.
