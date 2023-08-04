## Install MLIR

configure the cmake using the 15.x release version:
```cmake
cmake ..\llvm -G "Visual Studio 17 2022" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build . --target tools/mlir/test/check-mlir --config Debug
```

Configure, Build and test:
```
.\run.bat
```