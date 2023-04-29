
#ifndef PASSES_H_
#define PASSES_H_

#include "ops.h"
#include "lgf/pass.h"
#include <unordered_map>

namespace lgf{

void createConvertAddToSumPass(passManager &pm);

void createFuseSumOpPassPass(passManager &pm);

void createLhsAssociatePass(passManager &pm);

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