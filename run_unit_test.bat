cls
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -Bbuild
cmake --build build --parallel
.\build\bin\unit_tests.exe