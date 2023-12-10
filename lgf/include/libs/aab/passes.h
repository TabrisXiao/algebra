#ifndef LIB_AAB_PASSES_H
#define LIB_AAB_PASSES_H
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "lgf/pass.h"

namespace lgf::AAB{
class InterfaceInitRewriter : public rewriter<funcDefineOp>{
    public:
    InterfaceInitRewriter() = default;
    virtual resultCode rewrite(painter p, funcDefineOp *op){
        auto users = op->getCallee()->getUsers();
        resultCode ret;
        for(auto & user : users){
            auto tempg = p.getGraph();
            auto fc = dynamic_cast<funcCallOp*>(user);
            p.gotoGraph(user->getParentGraph());
            if(op->id=="Power"){
                p.replaceOp<powerOp>( user, fc->arg(0), fc->arg(1));
                ret.add(resultCode::success());
            }else if(op->id=="Sin"){
                p.replaceOp<funcSineOp>( user, fc->arg(0));
                ret.add(resultCode::success());
            }else if(op->id=="Cos"){
                p.replaceOp<funcCosOp>( user, fc->arg(0));
                ret.add(resultCode::success());
            }else if(op->id=="Derivative"){
                p.replaceOp<derivativeOp>( user, fc->arg(0), fc->arg(1));
                ret.add(resultCode::success());
            }else if(op->id=="Factor"){
                // p.replaceOp<factorOp>( user, fc->arg(0), fc->arg(1));
                // ret.add(resultCode::success());
            }else if(op->id=="Distribute"){
                p.replaceOp<distributeOp>( user, fc->arg(0));
                ret.add(resultCode::success());
            }else if(op->id=="Differentiate"){
                p.replaceOp<differentiateOp>( user, fc->arg(0), fc->arg(1));
                ret.add(resultCode::success());
            }else if(op->id=="Mapping"){
                p.replaceOp<mappingOp>( user, fc->getReturnType(), fc->arg(0));
                ret.add(resultCode::success());
            }else if(op->id=="Inverse"){
                p.replaceOp<inverseOp>( user, fc->arg(0) );
                ret.add( resultCode::success() );
            }
            p.gotoGraph(tempg);
        }
        return ret;
    } 
};

class InterfaceInitPass : public passBase{
    public:
    InterfaceInitPass(moduleOp *m)
    : passBase("AABInterfaceInitPass")
    , module(m) {}
    virtual resultCode run() final{
        painter p(getContext());
        addRewriter<InterfaceInitRewriter>(); 
        auto result = applyRewriterOnce(p, module);
        module->erase();
        return result;
    }
    moduleOp* module = nullptr;
};

std::unique_ptr<passBase> createInterfaceInitPass(moduleOp *m){
    return std::make_unique<InterfaceInitPass>(m);
}
}

#endif