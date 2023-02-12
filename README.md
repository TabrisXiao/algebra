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

## Documents

* The function of the `context` is to represent the envelop of operations and recording the informations of the envelop. Right now, the `region` object is used to describe the boundary of operations, so the `context` records to the global information for this boundary. The principle of the `context` is that:
    * `context` should have the information to find the larger `context` that contained it. 
    * The information included in the `context` should independent from any other information outside this `context`.

* `region` is basically a graph that consists by operations and elements. The desgin of the region is:
    * It can be contained by other `region` to form a nested structure.
    * The operations inside of a region should be isolated from other operations from outside. Elements inside of the region can only be passed to the outside by some specific operations, like `return` (Haven't implemented yet)
    * The elements can be passed into region as inputs.(Haven't implemented yet)

* The `element` object is a wrapper for any values or information would be used to exchange in between operations. Elements have to be owned by some operations and should never be created alone.

* The `operation` represents the actual operation we would like to execute. It is implemented as a vertex of a graph:
    * If some elements are owned by a operation, then this operation should be the only operation that can modify these elements. Other operations should never modify the elements.
    * Operations can only communiated with each others by passing the `element` objects. 
    * If a operation takes some elements owning by other operations as inputs, then this operation is consided as connected from those operations. 

* The `opBuilder` is the interface object used to create operations and manage the `context`. The `context` should be created by `opBuilder` to maintain the `context`