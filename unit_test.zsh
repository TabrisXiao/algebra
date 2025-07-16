#!/bin/bash
clear
cmake -Bbuild
cmake --build build --config Debug --parallel
./../build/bin/unit_tests