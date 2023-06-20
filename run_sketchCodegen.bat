cls
cmake -Bbuild
cmake --build build --parallel
.\build\bin\sketchCodegen.exe -src resources/sketch/math/ -dst lgl/include/math/ -i resources/sketch/math/
