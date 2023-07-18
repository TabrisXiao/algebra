cls
cmake -Bbuild
cmake --build build --parallel
.\build\bin\compiler.exe -src test/compiler/test.lgf