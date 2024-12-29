#!/usr/bin/env bash

cd build && rm -rf ./* && cmake .. -DPYTHON_EXECUTABLE=$(which python3) && make VERBOSE=1
