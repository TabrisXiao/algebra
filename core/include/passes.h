
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
        auto users = origOp->output()->getUsers();
        for(auto _op : users){
            if(auto op = dynamic_cast<sumOp*>(_op)){
                for(auto e : origOp->getInputs()){
                    op->acceptInput(e);
                }
            }
        }
        return 1;
    }
    void combinedIntoOne(sumOp *op1, sumOp *op2){
        // assume the output of op1 is used by op2 as input
        auto & inputs = op2->getInputs();
        element * ie = op1->output();
        // merge the inputs of op2 to op1
        for(auto e : inputs){
            if(ie == e) continue;
            op1->acceptInput(e);
        }
        //
    }
};

class convertAddToSumPass : public passBase{
    public:
    convertAddToSumPass(): passBase("convert_add_to_sum_pass") {
        addRewriter<convertAddToSumRewriter>();
    }
};

void createConvertAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<convertAddToSumPass>());
}
}



#endif