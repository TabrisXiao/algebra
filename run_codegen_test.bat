cls
cmake -Bbuild
cmake --build build --config Debug --parallel 
.\build\bin\codegen.exe -r .\lgf\resources\codegen .\lgf\include\libs\generated