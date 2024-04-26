#!/bin/bash
clear
cmake -B ../build
cmake --build ../build --config Debug --parallel
chomd +x ../build/bin/unit_tests
./../build/bin/unit_tests