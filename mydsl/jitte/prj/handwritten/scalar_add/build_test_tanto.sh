#!/bin/bash

# Compile the scalar_add test binary against Jitte libs.
# Pattern follows algo/jitte/prj/basic/build_test_tanto.sh.
#
# Assumes Jitte and Tanto have already been built under ../../../../jitte and
# ../../../../tanto respectively.

set -e

CXX=/usr/lib/llvm-17/bin/clang++

JITTE=../../../../../jitte
TANTO=../../../../../tanto

SRC=../../../../src/handwritten/scalar_add
BIN=../../bin/handwritten/scalar_add

mkdir -p $BIN

$CXX -std=c++20 -stdlib=libstdc++ -O3 -o $BIN/test_tanto \
    -I $SRC \
    -I $TANTO/src \
    $SRC/host/*.cpp \
    $SRC/test/*.cpp \
    $TANTO/jitte/lib/host/core.a \
    $JITTE/lib/tt_metal/tt_metal.a \
    $JITTE/lib/tt_metal/tt_metal_impl.a \
    $JITTE/lib/tt_metal/tt_metal_detail.a \
    $JITTE/lib/tt_metal/jit_build.a \
    $JITTE/lib/tt_metal/common.a \
    $JITTE/lib/tt_metal/llrt.a \
    $JITTE/lib/tt_metal/emulator.a \
    $JITTE/lib/tt_metal/device.a \
    $JITTE/lib/tt_metal/yaml_cpp.a \
    $JITTE/lib/device/api.a \
    $JITTE/lib/device/dispatch.a \
    $JITTE/lib/device/ref.a \
    $JITTE/lib/device/riscv.a \
    $JITTE/lib/device/core.a \
    $JITTE/lib/device/arch.a \
    $JITTE/lib/device/schedule.a \
    $JITTE/lib/whisper/riscv.a \
    $JITTE/lib/whisper/linker.a \
    $JITTE/lib/whisper/interp.a
