
#ifndef PASSES_H_
#define PASSES_H_

#include "ops.h"
#include "aog.h"

namespace aog{

class fuseAddToSumRewriter : public rewriter<addOp> {
    public : 
    fuseAddToSumRewriter (){}
    virtual bool rewrite(addOp *origOp) override{
        std::cout<<"addOp found"<<std::endl;
        return 1;
    }
};

class fuseAddToSumPass : public passBase{
    public:
    fuseAddToSumPass(): passBase("fuse_add_to_sum_pass") {
        addRewriter<fuseAddToSumRewriter>();
    }
};

void createFuseAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<fuseAddToSumPass>());
}
}



#endif