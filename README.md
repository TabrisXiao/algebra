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

## The bell is ringing!

* Create pass for analysis the association and the distribution
* Create op interface to manipulate the commutations for binaryOp.

### Issues

* The print from `Inverse`/`negative` can't show input type.

## Architecture
1. Concrete math object definition -> Schedule lowering/optimizing passes
2. Construct the operations from abstract math operation.
    * Bind the abstract types to concrete object types
4. Lowering/optimizing through scheduled passes to a concrete operation list
3. Serializing to a executable

### Abstract Algebra Dialect (AADialect)

The Ops:
* Multiply 
* Add
* Inverse
* Negative
* AElemDecl