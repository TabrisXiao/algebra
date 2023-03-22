
#ifndef PASSES_H_
#define PASSES_H_

#include "ops.h"
#include "pass.h"
#include <unordered_map>

namespace aog{

// -------------------- convertAddToSumPass ---------------------
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
    convertAddToSumPass(): passBase("convert_add_to_sum_pass") {}
    bool run() final{
        graphModifier gm;
        gm.addRewriter<convertAddToSumRewriter>();
        auto reg = getRegion();
        gm.walkApplyOnce(reg);
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

class fuseAddToSumPass : public passBase{
    public:
    fuseAddToSumPass(): passBase("fuse_add_to_sum_pass") {}
    bool run() final{
        graphModifier gm;
        gm.addRewriter<fuseAddToSumRewriter>();
        auto reg = getRegion();
        gm.walkApplyOnce(reg);
        return 0;
    }
};

void createFuseAddToSumPass(passManager &pm){
    pm.addPass(std::make_unique<fuseAddToSumPass>());
}

// -------------------- lhsAssociatePass ---------------------
class lhsAssociateRewriter : public rewriter<sumOp> {
    public:
    lhsAssociateRewriter () = default;
    virtual bool rewrite(opRewriter &rewriter, sumOp *origOp) override{
        auto & inputs = origOp->getInputs();
        std::unordered_map<element *, std::vector<multiplyOp*>> map;
        for(auto input : inputs){
            if(auto op = input->getDefiningOp<multiplyOp>()){
                auto lhs = op->lhs();
                if(map.find(lhs)==map.end()){
                    map.insert({lhs, {op}});
                } else {
                    map[lhs].push_back(op);
                }
            }
        }
        for(auto & [value, ops] : map){
            if(ops.size()< 2) continue;
            std::vector<element*> rhsValues;
            rhsValues.push_back(ops[0]->rhs());
            for(auto iter=ops.begin()+1; iter!=ops.end(); iter++){
                rhsValues.push_back((*iter)->rhs());
                rewriter.removeOp(*iter);
            }
            auto sumop = rewriter.create<sumOp>(rhsValues);
            auto mulop = rewriter.replaceOp<multiplyOp>(ops[0], value, sumop->output());
        }
        return 1;
    }
};

class lhsAssociatePass : public passBase{
    public:
    lhsAssociatePass(): passBase("lhs_associate_pass") {}
    bool run() final{
        graphModifier gm;
        gm.addRewriter<lhsAssociateRewriter>();
        auto reg = getRegion();
        gm.walkApplyOnce(reg);
        return 0;
    }
};

void createLhsAssociatePass(passManager &pm){
    pm.addPass(std::make_unique<lhsAssociatePass>());
}

// -------------------- rhsAssociatePass ---------------------
class rhsAssociateRewriter : public rewriter<sumOp> {
    public:
    rhsAssociateRewriter () = default;
    virtual bool rewrite(opRewriter &rewriter, sumOp *origOp) override{
        auto & inputs = origOp->getInputs();
        std::unordered_map<element *, std::vector<multiplyOp*>> map;
        for(auto input : inputs){
            if(auto op = input->getDefiningOp<multiplyOp>()){
                auto rhs = op->rhs();
                if(map.find(rhs)==map.end()){
                    map.insert({rhs, {op}});
                } else {
                    map[rhs].push_back(op);
                }
            }
        }
        for(auto & [value, ops] : map){
            if(ops.size()< 2) continue;
            std::vector<element*> lhsValues;
            lhsValues.push_back(ops[0]->rhs());
            for(auto iter=ops.begin()+1; iter!=ops.end(); iter++){
                lhsValues.push_back((*iter)->rhs());
                rewriter.removeOp(*iter);
            }
            auto sumop = rewriter.create<sumOp>(lhsValues);
            auto mulop = rewriter.replaceOp<multiplyOp>(ops[0], sumop->output(), value);
        }
        return 1;
    }
};

class rhsAssociatePass : public passBase{
    public:
    rhsAssociatePass(): passBase("rhs_associate_pass") {}
    bool run() final{
        graphModifier gm;
        gm.addRewriter<rhsAssociateRewriter>();
        auto reg = getRegion();
        gm.walkApplyOnce(reg);
        return 0;
    }
};

void createRhsAssociatePass(passManager &pm){
    pm.addPass(std::make_unique<rhsAssociatePass>());
}

}



#endif