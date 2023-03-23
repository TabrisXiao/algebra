#ifndef OPINTERFACE_H
#define OPINTERFACE_H

#include "operation.h"

namespace aog{
template<typename groupType>
class opGroup{
    public:
    opGroup() = default;
    bool normalization(){
        return groupType::normalize(dynamic_cast<operation*>(this));
    }
};

// several built-in groups are defined below:

// for operation that inputs are commutable 
class commutableOp : public opGroup<commutableOp>{
    public:
    commutableOp () = default;
    static bool normalize(operation* op){
        std::vector<element*> & vec = op->getInputs();
        std::sort(vec.begin(), vec.end(), [](element* a, element* b) { return a < b; });
        return 1;
    }
};

}

#endif