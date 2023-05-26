#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{


class moduleOp : public graph{
public:
    moduleOp() : graph("module"){}
    ~moduleOp(){}
    virtual std::string represent() {return getSID();}
};
//----------------------------------------
}
#endif