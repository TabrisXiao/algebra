cls
cmake -Bbuild
cmake --build build --config Debug --parallel 
.\build\bin\codegen.exe .\test\codegen\test.input