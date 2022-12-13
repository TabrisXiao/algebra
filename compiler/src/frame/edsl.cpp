
#include "edsl.h"
#include "mlir/IR/BuiltinOps.h"

namespace MC{
std::string ops::generateID(std::string id){
    return std::string(_ops_name)+"_"+id+"_"+std::string(std::to_string(index_));
}

mlir::Location ops::getNamedLoc(std::string loc){
    return mlir::NameLoc::get(mlir::StringAttr::get(builder.getContext(), generateID(loc)));
}

void ops::init(){
    module = builder.create<::mlir::ModuleOp>(
        getNamedLoc(generateID("module"))
    );
    builder.setInsertionPointToStart(module.getBody());
}

variable ops::declVar(std::string id){
    auto op = builder.create<MC::AA::AElemDecl>(
        getNamedLoc(generateID("declVar")),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), id)
    );
     variable x(op.output(), id);
    return x;
}

variable ops::add(variable &x, variable &y){
    auto op = builder.create<MC::AA::Add>(
        getNamedLoc(generateID("Add")),
        x,
        y
    );
    return variable(op.output(), x.getID());
}

variable ops::multiply(variable &x, variable &y){
    auto op = builder.create<MC::AA::Multiply>(
        getNamedLoc(generateID("Multiply")),
        x,
        y
    );
    return variable(op.output(), x.getID());
}

variable ops::inverse(variable &x){
    auto op = builder.create<MC::AA::Inverse>(
        getNamedLoc(generateID("Inverse")),
        x
    );
    return variable(op.output(), x.getID());
}

variable ops::negative(variable &x){
    auto op = builder.create<MC::AA::Negative>(
        getNamedLoc(generateID("Negative")),
        x
    );
    return variable(op.output(), x.getID());
}

}
