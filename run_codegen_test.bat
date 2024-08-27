cls
cmake -Bbuild
cmake --build build --config Debug --parallel 
.\build\bin\codegen.exe -r .\lgf\resources\codegen\math .\lgf\include\libs\generated\math