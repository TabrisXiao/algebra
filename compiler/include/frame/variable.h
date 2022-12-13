
#ifndef EDSL_VARIABLE_H_
#define EDSL_VARIABLE_H_

#include <string.h>
#include "symbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace MC{
class variable : public mlir::Value {
    // the variable is the object used in edsl for ops and dialects.
    // The type is instanced according to its id when generating the
    // IRs.
    public:
    variable() = default;
    //variable(llvm::StringRef _id) {id=_id;};
    variable(mlir::Value val, std::string id_): mlir::Value(val.getImpl()), id(id_) {
    }
    virtual ~variable() = default;
    std::string getID(){return id;}

    private:
    std::string id;
};
}


#endif