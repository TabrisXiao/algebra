
#include "libs/Builtin/ops.h"
#include "libs/AAB/ops.h"

namespace lgf::AAB{

resultCode addOp::rewrite(painter p, operation *op){
    p.replaceOp<sumOp>(op, op->inputValue(0), op->inputValue(1));
    return resultCode::success();
}

resultCode multiplyOp::rewrite(painter p, operation *op){
    p.replaceOp<productOp>(op, op->inputValue(0), op->inputValue(1));
    return resultCode::success();
}

resultCode sumOp::rewrite(painter p, operation *op){
    // check all input values and merge all sumOps into one
    resultCode result;
    auto iter = op->getInputs().begin();
    while(iter!=op->getInputs().end()){
        auto input = *iter;
        if(auto sum = input->getDefiningOp<sumOp>()){
            iter = p.replaceInputByDefOpInputs(iter, op);
            result.add(resultCode::success());
        } else {
            iter++;
        }
    }
    return result;
}

bool productOp::checkInverse(value* lhs, value* rhs){
    // check if lhs are inverse of rhs
    if(auto inv = lhs->getDefiningOp<inverseOp>()){
        if(inv->inputValue(0) == rhs) return true;
    }
    return false;
}

resultCode productOp::rewrite(painter p, operation *op){
    // check all input values and merge all productOps into one
    resultCode result;
    auto iter = op->getInputs().begin();
    while(iter!=op->getInputs().end()){
        auto input = *iter;
        if(auto product = input->getDefiningOp<productOp>()){
            iter = p.replaceInputByDefOpInputs(iter, op);
            result.add(resultCode::success());
        } else {
            iter++;
        }
    }
    // check if adjacent inputs are the inverse of each other, if so, remove them.
    // if no inputs left, replace the input by declaring a constant 1;
    iter = op->getInputs().begin();
    while(iter!=op->getInputs().end()-1){
        if(checkInverse(*iter, *(iter+1)) || checkInverse(*(iter+1), *iter)){
            iter = op->getInputs().erase(iter, iter+2);
            result.add(resultCode::success());
        }else iter++;
    }
    return result;
}

// this function transform the a*(b+c) into a*b+a*c
void distributeOp::transform(painter p, productOp *mulop, std::vector<value*>::iterator& iter, sumOp* addop){
    auto newaddop = p.sketch<sumOp>((*iter)->getType());
    for(auto input : addop->getInputs()){
        auto newop = p.sketch<productOp>(input->getType());
        for(auto it = mulop->getInputs().begin(); it != iter; it++){
            newop->registerInput(*it);
        }
        newop->registerInput(input);
        if(iter != mulop->getInputs().end()){
            for(auto it = iter+1; it != mulop->getInputs().end(); it++){
                newop->registerInput(*it);
            }
        }
        p.setPaintPointBefore(mulop);
        p.addOpToCurrentGraph(newop);
        newaddop->registerInput(newop->output());
    }
    p.setPaintPointAfter(mulop);
    p.addOpToCurrentGraph(newaddop);
    mulop->replaceBy(newaddop);
    matchCheck(p, newaddop);
}

resultCode distributeOp::matchCheck(painter p, operation *op){
    auto mulop = dynamic_cast<productOp*>(op);
    for(auto input : op->getInputs()){
        if( mulop && input->getDefiningOp<sumOp>()){
            auto iter = std::find(mulop->getInputs().begin(), mulop->getInputs().end(), input);
            transform(p, mulop, iter, input->getDefiningOp<sumOp>());
            return (resultCode::success());
        } else {
            auto result = matchCheck(p, input->getDefiningOp());
            if(result.isSuccess()) return result;
        }
    }
    return resultCode::pass();
}

resultCode distributeOp::rewrite(painter p, operation *op){
    // check all input values and merge all productOps into one
    auto input = dynamic_cast<distributeOp*>(op)->input();
    auto result = matchCheck(p, input->getDefiningOp());
    if(!result.isSuccess()){
        op->replaceBy(input->getDefiningOp());
        return resultCode::success();
    }
    return result;
}
}// namespace lgf::AAB