#!/bin/bash

# Copy translated metal kernels into JITTE_HOME for jit_build to find them.

set -e

export JITTE_HOME=../../../../../jitte/home

mkdir -p $JITTE_HOME/mydsl/handwritten/three_core_add_relu

cp -R -v ../../../../src/handwritten/three_core_add_relu/device \
    $JITTE_HOME/mydsl/handwritten/three_core_add_relu
