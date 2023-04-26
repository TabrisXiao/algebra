
#ifndef PASSES_H_
#define PASSES_H_

#include "ops.h"
#include "lgf/pass.h"
#include <unordered_map>

namespace lgf{

// -------------------- convertAddToSumPass ---------------------
class convertAddToSumRewriter : public rewriter<addOp> {
    public : 
    convertAddToSumRewriter (){}
    virtual bool rewrite(painter &rewriter, addOp *origOp) override{
        rewriter.replaceOp<sumOp>(origOp, origOp->lhs(), origOp->rhs());
        return 1;
    }
};

class convertAddToSumPass : public passBase{
    public:
    convertAddToSumPass(): passBase("convert_add_to_sum_pass") {}
    bool run() final{
        auto reg = getGraph();
        painter pntr(reg);
        pntr.addRewriter<convertAddToSumRewriter>();
        pntr.applyRewriterGreedy();
        return 0;
    }
};


void createConvertAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<convertAddToSumPass>());
}

// -------------------- fuseAddToSumPass ---------------------
class fuseAddToSumRewriter : public rewriter<sumOp>{
    public: 
    fuseAddToSumRewriter(){}
    virtual bool rewrite(painter &rewriter, sumOp *origOp) override{
        // find the input coming from another sumOp
        sumOp* op = nullptr;
        std::vector<operation*> users = origOp->output().getUsers();
        if(users.size() == 0)
            return 0;
        op = dynamic_cast<sumOp*>(users[0]);
        if(!op) return 0;

        auto inputs = op->getInputs();
        for(auto& iter = inputs.begin(); iter!=inputs.end(); iter++){
            if(origOp == (*iter).getDefiningOp<sumOp>()) continue;
            origOp->registInput(*iter);
        }
        op->replaceBy(origOp);
        op->dropAllInputs();
        op->setRemovable();
        return 1;
    }
};

class fuseSumOpPass : public passBase{
    public:
    fuseSumOpPass(): passBase("fuse_add_to_sum_pass") {}
    bool run() final{
        painter pntr(getGraph());
        pntr.addRewriter<fuseAddToSumRewriter>();
        pntr.applyRewriterGreedy();
        return 0;
    }
};

void createFuseSumOpPassPass(passManager &pm){
    pm.addPass(std::make_unique<fuseSumOpPass>());
}

// // -------------------- lhsAssociatePass ---------------------
// class lhsAssociateRewriter : public rewriter<sumOp> {
//     public:
//     lhsAssociateRewriter () = default;
//     virtual bool rewrite(painter &rewriter, sumOp *origOp) override{
//         auto & inputs = origOp->getInputs();
//         std::unordered_map<value *, std::vector<multiplyOp*>> map;
//         for(auto input : inputs){
//             if(auto op = input->getDefiningOp<multiplyOp>()){
//                 auto lhs = op->lhs();
//                 if(map.find(lhs)==map.end()){
//                     map.insert({lhs, {op}});
//                 } else {
//                     map[lhs].push_back(op);
//                 }
//             }
//         }
//         for(auto & [value, ops] : map){
//             if(ops.size()< 2) continue;
//             std::vector<value*> rhsValues;
//             rhsValues.push_back(ops[0]->rhs());
//             for(auto iter=ops.begin()+1; iter!=ops.end(); iter++){
//                 rhsValues.push_back((*iter)->rhs());
//                 rewriter.removeOp(*iter);
//             }
//             auto sumop = rewriter.create<sumOp>(rhsValues);
//             auto mulop = rewriter.replaceOp<multiplyOp>(ops[0], value, sumop->output());
//         }
//         return 1;
//     }
// };

// class lhsAssociatePass : public passBase{
//     public:
//     lhsAssociatePass(): passBase("lhs_associate_pass") {}
//     bool run() final{
//         painter pntr(getRegion());
//         pntr.addRewriter<lhsAssociateRewriter>();
//         pntr.applyRewriterGreedy();
//         return 0;
//     }
// };

// void createLhsAssociatePass(passManager &pm){
//     pm.addPass(std::make_unique<lhsAssociatePass>());
// }

// // -------------------- rhsAssociatePass ---------------------
// class rhsAssociateRewriter : public rewriter<sumOp> {
//     public:
//     rhsAssociateRewriter () = default;
//     virtual bool rewrite(painter &rewriter, sumOp *origOp) override{
//         auto & inputs = origOp->getInputs();
//         std::unordered_map<value *, std::vector<multiplyOp*>> map;
//         for(auto input : inputs){
//             if(auto op = input->getDefiningOp<multiplyOp>()){
//                 auto rhs = op->rhs();
//                 if(map.find(rhs)==map.end()){
//                     map.insert({rhs, {op}});
//                 } else {
//                     map[rhs].push_back(op);
//                 }
//             }
//         }
//         for(auto & [value, ops] : map){
//             if(ops.size()< 2) continue;
//             std::vector<value*> lhsValues;
//             lhsValues.push_back(ops[0]->lhs());
//             for(auto iter=ops.begin()+1; iter!=ops.end(); iter++){
//                 lhsValues.push_back((*iter)->lhs());
//                 rewriter.removeOp(*iter);
//             }
//             auto sumop = rewriter.create<sumOp>(lhsValues);
//             auto mulop = rewriter.replaceOp<multiplyOp>(ops[0], sumop->output(), value);
//         }
//         return 1;
//     }
// };

// class rhsAssociatePass : public passBase{
//     public:
//     rhsAssociatePass(): passBase("rhs_associate_pass") {}
//     bool run() final{
//         painter pntr(getRegion());
//         pntr.addRewriter<rhsAssociateRewriter>();
//         pntr.applyRewriterGreedy();
//         return 0;
//     }
// };

// void createRhsAssociatePass(passManager &pm){
//     pm.addPass(std::make_unique<rhsAssociatePass>());
// }

}


#endif