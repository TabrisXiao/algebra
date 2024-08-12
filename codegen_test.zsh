#!/bin/bash
clear
cmake -B ../build
cmake --build ../build --config Debug --parallel
./../build/bin/codegen