#ifndef LIB_AAB_PASSES_H
#define LIB_AAB_PASSES_H
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "lgf/pass.h"

namespace lgf::AAB{
class InterfaceInitRewriter : public rewriter<funcDefineOp>{
    public:
    InterfaceInitRewriter() = default;
    virtual bool rewrite(painter p, funcDefineOp *op){
        auto users = op->getCallee()->getUsers();
        for(auto & user : users){
            auto tempg = p.getGraph();
            auto fc = dynamic_cast<funcCallOp*>(user);
            p.gotoGraph(user->getParentGraph());
            if(op->id=="Power"){
                p.replaceOp<powerOp>( user, fc->arg(0), fc->arg(1));
            }else if(op->id=="Sin"){
                p.replaceOp<funcSineOp>( user, fc->arg(0));
            }else if(op->id=="Cos"){
                p.replaceOp<funcCosOp>( user, fc->arg(0));
            }else if(op->id=="Derivative"){
                p.replaceOp<derivativeOp>( user, fc->arg(0), fc->arg(1));
            }else if(op->id=="Factor"){
                p.replaceOp<factorOp>( user, fc->arg(0), fc->arg(1));
            }else if(op->id=="Distribute"){
                p.replaceOp<distributeOp>( user, fc->arg(0));
            }
            p.gotoGraph(tempg);
        }
        return 0;
    } 
};

class InterfaceInitPass : public passBase{
    public:
    InterfaceInitPass(moduleOp *m)
    : passBase("AABInterfaceInitPass")
    , module(m) {}
    virtual bool run() final{
        painter p(getContext());
        addRewriter<InterfaceInitRewriter>(); 
        applyRewriterOnce(p, module);
        module->erase();
        return 0;
    }
    moduleOp* module = nullptr;
};

std::unique_ptr<passBase> createInterfaceInitPass(moduleOp *m){
    return std::make_unique<InterfaceInitPass>(m);
}
}

#endif