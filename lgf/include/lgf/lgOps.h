#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{
class moduleOp : public operation{
public:
    moduleOp(){ setTypeID("module"); }
    ~moduleOp(){std::cout<<"deleted"<<std::endl;}
    virtual graph* getSubgraph() override final {return &block;}
    std::string represent() {return getTypeID();}
    virtual void printOp() override final{
        global::stream::getInstance().printIndent();
        global::stream::getInstance()<<getTypeID()<<" : ";
        block.print();
        global::stream::getInstance()<<"\n";
    }
    void assignID(int n=0){ block.assignID(n); }
    graph block;
};
}
#endif