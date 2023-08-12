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


## Design

LGFCompiler consists by three parts:
* Syntax parser: parse the codes into AST
* LGFTranslator: translate the AST to Logic graph
* LGFInference: produce the final APIs from the Logic graph

### Module
The conception of module is similar to the conception of `class`. It is a set containing functions and/or variables as member functions/variables. Functions can be accessed from outside of the module but the variables can only be accessed after the module is initialized. Member functions can only access the module variable by using the `self` keyword like `python`. To access the module, the dot `.` symbol is used to access modules. The usage can be illustrated as: 
```
import math;
lib = math();
a = math.sum(2,3); // can access the function in math without self as argument
lib.a = a;
lib.check(); // here the check is assume to be defined as check(self) inside the math module
```


#### LGF modules
A LGF modules includes the definitions for types, operations and the transform passes. For each module, it should have a corresponding header file encoded using the lgf syntax inorder to register the identifiers to parser. 

### Feature need to support
* Support the binary op: `<=`, `>=`, `==`, etc.
* Add the scope name for each identifiers
* Design the module struct
* Add `import` feature to load modules
* APIs design
* Support the function name overloading.
* Parse function definition block.
* Add verification pass for ops. 

### In progress
* create `refOp` to get the member variable to avoid to create links inbetween the member function and member variable, this op should hold the reference value from the module or function but it should not be a user of that value. The problem is that we need to check if the operator owning the value referred is still valid or not. (need to verify)
* Fix the issue of getting the function call such like `math::UnitMatrix(2)`.