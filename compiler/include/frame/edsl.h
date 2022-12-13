
#ifndef FRAME_EDSL_H_
#define FRAME_EDSL_H_

#include "config.h"
#include "variable.h"
#include "symbolTable.h"
#include "init.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "AA/AADialect.h"
#include "AA/AATypes.h"
#include "AA/AAOps.h"

namespace MC{
class edslBase {
    public : 
    edslBase () = default ;
    ~edslBase () = default; 
};

class ops {
    public : 
    ops (llvm::StringRef name, mlir::MLIRContext *ctx):builder(ctx){
        _ops_name = name;
    }
    ~ops(){}
    variable declVar(std::string id);
    variable add(variable &x, variable &y);
    variable multiply(variable &x, variable &y);
    variable inverse(variable &x);
    variable negative(variable &x);
    std::string generateID(std::string id);
    mlir::Location getNamedLoc(std::string loc);
    void init();
    void dump(){module.dump();}

    private:
    mlir::OpBuilder builder;
    std::string _ops_name;
    int index_=0;
    mlir::ModuleOp module;
    //mlir::MLIRContext ctx;
};
}


#endif