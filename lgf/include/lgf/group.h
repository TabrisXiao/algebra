#ifndef GROUP_H
#define GROUP_H
#include <memory.h>
#include "operation.h"
#include "painter.h"

namespace lgf{

// group is used to classify operations based on general properties,
// usages, or whatever general ideas. And also provide interface methods
// to this group. 
template<typename concreteType>
class group{
    public: 
    group () = default;
    virtual ~group(){}

    // method to check if an op belongs to certain group.
    static bool isGroup(operation *op){
        if(auto gp = dynamic_cast<group<concreteType>>(op))
            return true;
        return false;
    }
};

}

#endif