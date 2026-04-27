#!/bin/bash

# Copy the thcon_scalar_add kernel into JITTE_HOME so jit_build can find it.
# Pattern follows scalar_add/deploy_jitte.sh.

set -e

export JITTE_HOME=../../../../../jitte/home

mkdir -p $JITTE_HOME/mydsl/handwritten/thcon_scalar_add

cp -R -v ../../../../src/handwritten/thcon_scalar_add/device \
    $JITTE_HOME/mydsl/handwritten/thcon_scalar_add
