
#include "libs/Builtin/ops.h"
#include "libs/AAB/ops.h"

namespace lgf::AAB{

resultCode addOp::rewrite(painter p, operation *op){
    p.replaceOp<sumOp>(op, op->inputValue(0), op->inputValue(1));
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
}// namespace lgf::AAB