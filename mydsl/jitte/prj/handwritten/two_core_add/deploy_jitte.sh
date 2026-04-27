#!/bin/bash

# Copy translated metal kernels into JITTE_HOME for jit_build to find them.

set -e

export JITTE_HOME=../../../../../jitte/home

mkdir -p $JITTE_HOME/mydsl/handwritten/two_core_add

cp -R -v ../../../../src/handwritten/two_core_add/device \
    $JITTE_HOME/mydsl/handwritten/two_core_add
