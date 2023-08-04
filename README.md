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
