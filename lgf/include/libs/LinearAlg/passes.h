#ifndef LIB_LINEARALG_PASSES_H
#define LIB_LINEARALG_PASSES_H
#include "lgf/painter.h"
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "lgf/pass.h"

namespace lgf{

namespace LinearAlg{

class UnitMatrixInterfaceInit : public rewriter<funcDefineOp>{
    public:
    UnitMatrixInterfaceInit() = default;
    bool rewrite(painter & p, funcDefineOp* op){
        auto users = op->getCallee()->getUsers();
        for(auto & user : users){

        }
        return 0;
    }
};


class InterfaceInitPass : public passBase{
    public: 
    InterfaceInitPass(moduleOp* op) : passBase("LinearAlgInterfaceInitPass") {
        module = op;
    }
    virtual bool run() final { return 0; }
    moduleOp *module = nullptr;
};

std::unique_ptr<passBase> createInterfaceInitPass(moduleOp *m){
    return std::make_unique<InterfaceInitPass>(m);
}
}
}

#endif