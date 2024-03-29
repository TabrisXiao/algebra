
#include "libs/SIO/exporter.h"
#include "libs/Builtin/ops.h"
#include "libs/SIO/ops.h"

namespace lgf::SIO{

std::string export2Txt::process(value* val ){
    std::string res;
    if(auto op = val->getDefiningOp<SIO::scalarProductOp>()){
        for(auto & input : op->getInputs()){
            res += process(input) + "*";
        }
        res.pop_back();
    }else if(auto op = val->getDefiningOp<sumOp>()){
        for(auto & input : op->getInputs()){
            res += process(input) + "+";
        }
        res.pop_back();
    }else if(auto op = val->getDefiningOp<funcOp>()){
        res += op->getFuncName() + "(";
        for(auto & input : op->getInputs()){
            res += process(input) + ",";
        }
        res.pop_back();
        res += ")";
    }else if(auto op = val->getDefiningOp<symbolOp>()){
        res += op->getSymbol();
    }else if(auto op = val->getDefiningOp<lgf::assignOp>()){
        res = process(op->lhs()) +" = "+ process(op->rhs());
    }else if(auto op = val->getDefiningOp<partialD>()){
        res = "d("+process(op->inputValue(0))+")/d("+process(op->inputValue(1))+")";
    }else {
        auto rep = "\n"+val->getDefiningOp()->represent();
        THROW("The following Op is not supported in export2Txt: "+rep+"\n");
    }
    return res;
}

void export2Txt::run(graph* entry){
    auto list = entry->getNodeList();
    bool mainFound = false;
    auto module = dynamic_cast<moduleOp*>(entry);
    if(module && module->name == "main"){
        mainFound = true;
    }
    if( mainFound ) os<<"--- export main to text: ---\n\n";
    for(auto & node : list ){
        if(auto op = dynamic_cast<returnOp*>(node)){
            if(op->getInputSize() == 1){
                auto val = op->inputValue(0);
                os<<process(val);
            }
        }else if(auto module = dynamic_cast<moduleOp*>(node)){
            for(auto & input : module->getInputs()){
                run(module);
                
            }
        }
    }
    if( mainFound ) os<<"\n\n--- end ---\n";
}

}// namespace lgf::SIO