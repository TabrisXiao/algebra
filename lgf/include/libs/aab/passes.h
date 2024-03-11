#ifndef LIB_AAB_PASSES_H
#define LIB_AAB_PASSES_H
#include "libs/Builtin/ops.h"
#include "ops.h"
#include "libs/AAB/types.h"
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
            if(op->id=="Factor"){
                // p.replaceOp<factorOp>( user, fc->arg(0), fc->arg(1));
                // ret.add(resultCode::success());
            }else if(op->id=="Distribute"){
                p.replaceOp<distributeOp>( user, fc->arg(0));
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

class commutableProductRewriter: public rewriter<productOp>{
    public:
    commutableProductRewriter() = default;
    virtual resultCode rewrite(painter p, productOp *op){
        // this rewriter rewrites a commutableProductOp into a productOp
        // with the same inputs but with the inputs sorted by their
        // hash value
        auto result = resultCode::pass();
        bool isCommutative = true;
        auto inputs = op->getInputs();
        std::sort(inputs.begin(), inputs.end(), [](value* a, value* b){
            return a < b;
            //return a->getHashValue() < b->getHashValue();
        });
        for(auto it = inputs.begin(); it!= inputs.end(); it++){
            if( auto axiom = (*it)->getType().getDesc<algebraAxiom>() ){
                if(!axiom->is(algebraAxiom::multiply_commutable)){
                    isCommutative = false;
                    break;
                }
            }
        }
        if(!isCommutative) return result;
        auto newop = p.replaceOp<commutableProductOp>(op, inputs);
        result.add(resultCode::success());
        return result;
    }
};

class AAProcess: public passBase{
    public:
    AAProcess()
    : passBase("AAProcess") {}
    virtual resultCode run() final{
        painter p(getContext());
        addRewriter<commutableProductRewriter>(); 
        auto result = applyRewriterGreedy(p, getGraph());
        return result;
    }
};

std::unique_ptr<passBase> createAAProcess(){
    return std::make_unique<AAProcess>();
}
} // namespace lgf::AAB

#endif