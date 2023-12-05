#ifndef LIB_LINEARALG_PASSES_H
#define LIB_LINEARALG_PASSES_H
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "lgf/pass.h"

namespace lgf::LinearAlg{

class InterfaceInitRewriter : public rewriter<funcDefineOp>{
    public:
    InterfaceInitRewriter() = default;
    resultCode rewrite(painter p, funcDefineOp* op){
        auto users = op->getCallee()->getUsers();
        // resultCode ret;
        // for(auto & user : users){
        //     auto tempg = p.getGraph();
        //     auto fc = dynamic_cast<funcCallOp*>(user);
        //     p.gotoGraph(user->getParentGraph());
        //     if(op->id=="UnitMatrix"){
        //         p.replaceOp<declOp>( user, fc->arg(0));
        //     }
        //     p.gotoGraph(tempg);
        // }
        return resultCode::pass();
    }
};


class InterfaceInitPass : public passBase{
    public: 
    InterfaceInitPass(moduleOp* op) : passBase("LinearAlgInterfaceInitPass") {
        module = op;
    }
    virtual resultCode run() final { 
        module->erase();
        return resultCode::pass(); 
    }
    moduleOp *module = nullptr;
};

std::unique_ptr<passBase> createInterfaceInitPass(moduleOp *m){
    return std::make_unique<InterfaceInitPass>(m);
}

}// namespace lgf::LinearAlg

#endif