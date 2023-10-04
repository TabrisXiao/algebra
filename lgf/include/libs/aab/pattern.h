#ifndef LIB_AAB_PATTERN_H
#define LIB_AAB_PATTERN_H
#include "lgf/utils.h"
namespace lgf{
using namespace utils;

template<typename opType1, typename opType2>
logicResult rDistributivePattern(operation *root){
    // the pattern is
    //    opType1
    //    /    \
    // value   opType2
    auto op1 = dynamic_cast<opType1*>(root);
    if(!op1) return logicResult::fail();
    opType2* op2 = root->inputValue(1)->getDefiningOp<opType2>();
    if(op2)
        return logicResult::success();
    return logicResult::fail();
}
template<typename opType1, typename opType2>
logicResult lDistributivePattern(operation *root){
    // the pattern is
    //    opType1
    //    /    \
    // opType2   value
    auto op1 = dynamic_cast<opType1*>(root);
    if(!op1) return logicResult::fail();
    opType2* op2 = root->inputValue(0)->getDefiningOp<opType2>();
    if(op2)
        return logicResult::success();
    return logicResult::fail();
}

template<typename opType1, typename opType2>
logicResult AssociativePattern(operation *root, int index = 0){
    // pattern:
    //      opType1
    //      /     \
    //  opType2   opType2
    //    /  \     /  \
    //  val1  xx  val1 yy
    //    |
    //   the index-th value
    auto op1 = dynamic_cast<opType1*>(root);
    if(!op1) return logicResult::fail();
    opType2* op2 = root->inputValue(0)->getDefiningOp<opType2>(); 
    opType2* op3 = root->inputValue(1)->getDefiningOp<opType2>();
    if(op2 && op3){
        if(index < op2->getInputSize() && index < op3->getInputSize()){
            if(op2->inputValue(index) == op3->inputValue(index)) 
                return logicResult::success();
        }
    }
    return logicResult::fail();
}


}

#endif