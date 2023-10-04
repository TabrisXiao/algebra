## LGF Compiler

Logic graph frame (LGF) compiler is a compiler to converting logic structrue into graphs then optimizing and transforming the inputs to process the logic inference or calculation.  

To use the language to use this compiler is following the following syntax rules:

## Syntax

A variable is defined implicitly, the type of the variable is inferred from the context. But the type of one variable can not be changed from one to other non-compatable one. 

Function has to be defined initiated with `def`:
```
// return type int
def sum (int a, int b) -> int{
    return a+b;
}
// no return value
def check(int a) {
    return;
}
```
The dot operator `.` can be used to access the member function of a module. It can also used to avoid the name confliction. However, it can't be used to access the member functions that accessing member variables (since these functions can only be called after a module initialized).

## Design

LGFCompiler consists by three parts:
* Syntax parser: parse the codes into AST
* LGFTranslator: translate the AST to Logic graph
* LGFInference: produce the final APIs from the Logic graph

### Module
The conception of module is similar to the conception of `class`. It is a set containing functions and/or variables as member functions/variables. Functions can be accessed from outside of the module but the variables can only be accessed after the module is initialized. To access the module, the dot `.` symbol is used to access modules. The usage can be illustrated as: 
```
import math;
lib = math();
a = math.sum(2,3); // can access the function in math without self as argument
lib.a = a;
lib.check(); // here the check is assume to be defined as check(self) inside the math module
```


#### LGF modules
A LGF modules includes the definitions for types, operations and the transform passes. For each module, it should have a corresponding header file encoded using the lgf syntax inorder to register the identifiers to parser. 


### Linear Algebra
The `LinearAlg` lib contains the following ops:
* `UnitMatrix(int n)`: define a unit matrix with dimension `nxn`.
* `Transpose(matrix)`: Transpose a matrix.
* `Determinant(matrix)`: Calculate the determinant of the matrix.
* `Scaling(matrix, scale)`: Multiply each element of `matrix` by `scale`.
* `Slice(matrix)`: Slice the matrix into row vectors.
* `DirectProduct(matrix, matrix)`: 

### Planning
* Support the binary op: `<=`, `>=`, `==`, etc.
* Add verification for ops. 
* design the APIs arch.
* Change the IR of `accessOp` to show the module name that the accessing object belongs to.
* add `verify()` function to `operation` to prevent cycle dependency and other issues.
