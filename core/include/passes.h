
#ifndef PASSES_H_
#define PASSES_H_

#include "ops.h"
#include "aog.h"

namespace aog{

class convertAddToSumRewriter : public rewriter<addOp> {
    public : 
    convertAddToSumRewriter (){}
    virtual bool rewrite(opRewriter &rewriter, addOp *origOp) override{
        rewriter.replaceOp<sumOp>(origOp, origOp->lhs(), origOp->rhs());
        return 1;
    }
};

class fuseAddToSumRewriter : public rewriter<sumOp>{
    public: 
    fuseAddToSumRewriter(){}
    virtual bool rewrite(opRewriter &rewriter, sumOp *origOp) override{
        bool needScan = 1;
        while( needScan){
            // find the input coming from another sumOp
            sumOp* op = nullptr;
            auto & inputs = origOp->getInputs();
            for(auto iter = inputs.begin(); iter!=inputs.end(); iter++){
                if(op = dynamic_cast<sumOp*>((*iter)->getDefiningOp())){
                    auto & newinputs = op->getInputs();
                    auto niter = inputs.insert(iter, newinputs.begin(), newinputs.end());
                    inputs.erase(niter+newinputs.size());
                    for(auto newinput : newinputs){
                        auto iop = newinput->getDefiningOp();
                        origOp->linkFrom(*iop);
                    }
                    break;
                } else { needScan = 0; }
            }
            if(op) rewriter.removeOp(op);
        }
        return 1;
    }
};

class convertAddToSumPass : public passBase{
    public:
    convertAddToSumPass(): passBase("convert_add_to_sum_pass") {
        addRewriter<convertAddToSumRewriter>();
    }
};

class fuseAddToSumPass : public passBase{
    public:
    fuseAddToSumPass(): passBase("fuse_add_to_sum_pass") {
        addRewriter<fuseAddToSumRewriter>();
    }
};

void createConvertAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<convertAddToSumPass>());
}

void createFuseAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<fuseAddToSumPass>());
}

}



#endif