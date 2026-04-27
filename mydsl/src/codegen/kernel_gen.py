#!/usr/bin/env python3
"""mydsl per-core kernel generator (phase 2.5).

Loads a Python spec dict and emits the 3 Tanto kernel .cpp files
(reader/writer/math) for one core, matching the layout under
src/handwritten/<name>/device/tanto/.

v0 scope: single-core, exactly one output pipe, binary compute expression
of the form `OUT = A OP B` with OP in {+, -, *}. Anything else: error.
"""

import argparse
import importlib.util
import string
import sys
from pathlib import Path

CODEGEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODEGEN_DIR.parents[1]
TEMPLATE_DIR = CODEGEN_DIR / "templates"
DEFAULT_OUT_ROOT = CODEGEN_DIR / "out"

INDENT8 = " " * 8
INDENT12 = " " * 12

BINARY_OPS = {"+": "add", "-": "sub", "*": "mul"}


def load_spec(spec_path: Path) -> dict:
    spec_path = spec_path.resolve()
    loader_spec = importlib.util.spec_from_file_location("user_spec", spec_path)
    mod = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(mod)
    return mod.spec


def parse_compute(compute: str, inputs: list, outputs: list) -> dict:
    s = compute.replace(" ", "")
    if "=" not in s:
        raise ValueError(f"compute must be 'OUT = A OP B': {compute!r}")
    lhs, rhs = s.split("=", 1)
    out_names = [o["name"] for o in outputs]
    in_names = [i["name"] for i in inputs]
    if lhs not in out_names:
        raise ValueError(f"compute LHS {lhs!r} not in outputs {out_names}")
    for sym, op_name in BINARY_OPS.items():
        if sym in rhs:
            a, b = rhs.split(sym, 1)
            if a not in in_names or b not in in_names:
                raise ValueError(
                    f"compute operands {a!r},{b!r} not in inputs {in_names}")
            return {"op": op_name, "lhs": lhs, "operands": [a, b]}
    raise ValueError(
        f"v0 only supports binary compute with op in {list(BINARY_OPS)}: {compute!r}")


def join_indented(items: list, indent: str) -> str:
    return "\n".join(indent + s for s in items)


def build_reader_subs(spec: dict) -> dict:
    inputs = spec["inputs"]
    return {
        "name": spec["name"],
        "global_args": ",\n".join(f"{INDENT8}global<T> g{i['name']}" for i in inputs),
        "pipe_args":   ",\n".join(f"{INDENT8}pipe<T> {i['pipe']}"     for i in inputs),
        "pos_args":    ",\n".join(f"{INDENT8}uint32 g{i['name']}_pos" for i in inputs),
        "reserve_block": join_indented(
            [f"{i['pipe']}.reserve_back();" for i in inputs], INDENT8),
        "read_block": join_indented(
            [f"{i['pipe']}.read(0, g{i['name']}, g{i['name']}_pos, block_items);"
             for i in inputs], INDENT8),
        "push_block": join_indented(
            [f"{i['pipe']}.push_back();" for i in inputs], INDENT8),
        "pos_inc_block": join_indented(
            [f"g{i['name']}_pos += block_items;" for i in inputs], INDENT8),
    }


def build_writer_subs(spec: dict) -> dict:
    outputs = spec["outputs"]
    return {
        "name": spec["name"],
        "global_args": ",\n".join(f"{INDENT8}global<T> g{o['name']}" for o in outputs),
        "pipe_args":   ",\n".join(f"{INDENT8}pipe<T> {o['pipe']}"     for o in outputs),
        "pos_args":    ",\n".join(f"{INDENT8}uint32 g{o['name']}_pos" for o in outputs),
        "wait_block": join_indented(
            [f"{o['pipe']}.wait_front();" for o in outputs], INDENT8),
        "write_block": join_indented(
            [f"{o['pipe']}.write(0, g{o['name']}, g{o['name']}_pos, block_items);"
             for o in outputs], INDENT8),
        "pop_block": join_indented(
            [f"{o['pipe']}.pop_front();" for o in outputs], INDENT8),
        "pos_inc_block": join_indented(
            [f"g{o['name']}_pos += block_items;" for o in outputs], INDENT8),
    }


def build_math_subs(spec: dict, parsed: dict) -> dict:
    inputs = spec["inputs"]
    outputs = spec["outputs"]
    if len(outputs) != 1:
        raise ValueError("v0 math: exactly one output pipe required")
    pipe_list = inputs + outputs
    in_pipe = {i["name"]: i["pipe"] for i in inputs}
    operand_pipes = [in_pipe[n] for n in parsed["operands"]]
    return {
        "name": spec["name"],
        "compute_repr": spec["compute"],
        "pipe_args": ",\n".join(f"{INDENT8}pipe<T> {p['pipe']}" for p in pipe_list),
        "reserve_out_block": join_indented(
            [f"{o['pipe']}.reserve_back();" for o in outputs], INDENT8),
        "wait_in_block": join_indented(
            [f"{i['pipe']}.wait_front();" for i in inputs], INDENT8),
        "compute_call": f"acc.{parsed['op']}({', '.join(operand_pipes)}, i, i, i);",
        "pack_block": join_indented(
            [f"acc.pack(i, {o['pipe']});" for o in outputs], INDENT12),
        "pop_in_block": join_indented(
            [f"{i['pipe']}.pop_front();" for i in inputs], INDENT8),
        "push_out_block": join_indented(
            [f"{o['pipe']}.push_back();" for o in outputs], INDENT8),
    }


def render(template_name: str, subs: dict) -> str:
    text = (TEMPLATE_DIR / template_name).read_text()
    return string.Template(text).substitute(subs)


def generate(spec_path: Path, out_root: Path) -> Path:
    spec = load_spec(spec_path)
    parsed = parse_compute(spec["compute"], spec["inputs"], spec["outputs"])
    name = spec["name"]
    out_dir = out_root / name / "device" / "tanto"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        (f"{name}_reader.cpp", "reader.cpp.tmpl", build_reader_subs(spec)),
        (f"{name}_math.cpp",   "math.cpp.tmpl",   build_math_subs(spec, parsed)),
        (f"{name}_writer.cpp", "writer.cpp.tmpl", build_writer_subs(spec)),
    ]
    for fname, tmpl, subs in artifacts:
        target = out_dir / fname
        target.write_text(render(tmpl, subs))
        try:
            rel = target.relative_to(REPO_ROOT)
        except ValueError:
            rel = target
        print(f"[gen] {rel}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="mydsl per-core kernel generator")
    ap.add_argument("spec", help="Path to spec .py (e.g. src/codegen/specs/scalar_add.py)")
    ap.add_argument("--out", default=None,
                    help=f"Output root (default: {DEFAULT_OUT_ROOT})")
    args = ap.parse_args()
    out_root = Path(args.out).resolve() if args.out else DEFAULT_OUT_ROOT
    generate(Path(args.spec), out_root)


if __name__ == "__main__":
    main()
