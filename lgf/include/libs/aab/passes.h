#ifndef LIB_AAB_PASSES_H
#define LIB_AAB_PASSES_H
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "lgf/pass.h"

namespace lgf::AAB{
class InterfaceInitRewriter : public rewriter<funcDefineOp>{
    public:
    InterfaceInitRewriter() = default;
    virtual bool rewrite(painter &p, funcDefineOp *op){
        auto users = op->getCallee()->getUsers();
        for(auto & user : users){
            auto tempg = p.current_graph;
            if(op->id=="Power"){
                p.gotoGraph(user->getParentGraph());
                p.replaceOp<powerOp>( user, op->argument(0), op->argument(1));
                p.gotoGraph(tempg);
            }else if(op->id=="Sin"){
                p.gotoGraph(user->getParentGraph());
                p.replaceOp<funcSineOp>( user, op->argument(0));
                p.gotoGraph(tempg);
            }else if(op->id=="Cos"){
                p.gotoGraph(user->getParentGraph());
                p.replaceOp<funcCosOp>( user, op->argument(0));
                p.gotoGraph(tempg);
            }
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
        return 0;
    }
    moduleOp* module = nullptr;
};

std::unique_ptr<passBase> createInterfaceInitPass(moduleOp *m){
    return std::make_unique<InterfaceInitPass>(m);
}
}

#endif