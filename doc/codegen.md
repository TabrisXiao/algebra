
# Codegen 

This is a library used to generate the data from logic graph. The suppose of this codegen library is used for easing the operation updates raising from the updates of LGF. Here I document the lex of the codegen. It is a mixture of the cpp and my customized keywords. 

## Keyword
The operation declaration should be
```cpp
@LGF::operation opname <opInterface> {
    // comments are still initiated with //
    // LGF keywords are initiated with @LGF::
    // the inheritation can be followed inside < >. separated by comma

    // the customized function needs to contained in
    // the def block and it will be directly added as member 
    // function directly to the defining operation.
    def{
        void customizedFunc(){
            return;
        }
    }
    
}
```