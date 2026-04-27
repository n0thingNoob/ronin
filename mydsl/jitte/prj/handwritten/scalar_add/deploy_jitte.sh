#!/bin/bash

# Copy translated metal kernels into JITTE_HOME for jit_build to find them.
# Pattern follows algo/jitte/prj/deploy_jitte.sh.

set -e

export JITTE_HOME=../../../../../jitte/home

mkdir -p $JITTE_HOME/mydsl/handwritten/scalar_add

cp -R -v ../../../../src/handwritten/scalar_add/device \
    $JITTE_HOME/mydsl/handwritten/scalar_add
