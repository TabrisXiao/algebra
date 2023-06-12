cls
cmake -Bbuild
cmake --build build --parallel
.\build\bin\sketchCodegen.exe resources/sketch/math/aab/ops.lgft lgl/include/math/aab/

.\build\bin\sketchCodegen.exe resources/sketch/math/analysis/ops.lgft lgl/include/math/analysis/