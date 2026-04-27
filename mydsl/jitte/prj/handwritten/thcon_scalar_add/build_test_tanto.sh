#!/bin/bash

# Compile the thcon_scalar_add Stage C and Stage D test binaries against
# Jitte libs. Pattern follows scalar_add/build_test_tanto.sh.

set -e

CXX=/usr/lib/llvm-17/bin/clang++

JITTE=../../../../../jitte
TANTO=../../../../../tanto

SRC=../../../../src/handwritten/thcon_scalar_add
BIN=../../bin/handwritten/thcon_scalar_add

mkdir -p $BIN

JITTE_LIBS=(
    $TANTO/jitte/lib/host/core.a
    $JITTE/lib/tt_metal/tt_metal.a
    $JITTE/lib/tt_metal/tt_metal_impl.a
    $JITTE/lib/tt_metal/tt_metal_detail.a
    $JITTE/lib/tt_metal/jit_build.a
    $JITTE/lib/tt_metal/common.a
    $JITTE/lib/tt_metal/llrt.a
    $JITTE/lib/tt_metal/emulator.a
    $JITTE/lib/tt_metal/device.a
    $JITTE/lib/tt_metal/yaml_cpp.a
    $JITTE/lib/device/api.a
    $JITTE/lib/device/dispatch.a
    $JITTE/lib/device/ref.a
    $JITTE/lib/device/riscv.a
    $JITTE/lib/device/core.a
    $JITTE/lib/device/arch.a
    $JITTE/lib/device/schedule.a
    $JITTE/lib/whisper/riscv.a
    $JITTE/lib/whisper/linker.a
    $JITTE/lib/whisper/interp.a
)

build_one() {
    local main_file=$1
    local out_name=$2
    $CXX -std=c++20 -stdlib=libstdc++ -O3 -o $BIN/$out_name \
        -I $SRC \
        -I $TANTO/src \
        $SRC/host/*.cpp \
        $SRC/test/$main_file \
        "${JITTE_LIBS[@]}"
}

# Stage C binary (default kernel = thcon_scalar_add_kernel.cpp).
build_one main.cpp test_tanto

# Stage D binary (kernel = thcon_atomic_inc_kernel.cpp).
build_one main_atomic_inc.cpp test_atomic_inc

# Stage E binary (kernel = thcon_runtime_arg_kernel.cpp; host→ThCon-GPR bridge).
build_one main_runtime_arg.cpp test_runtime_arg
