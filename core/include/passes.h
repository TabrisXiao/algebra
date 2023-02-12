
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