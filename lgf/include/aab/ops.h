
#ifndef AAOPS_H_
#define AAOPS_H_
#include "lgf/operation.h"
#include "lgf/lgf.h"
#include "groups.h"
//#include "lgf/group.h"
//#include "lgf/lgInterfaces.h"
//#include "pass.h"
//#include "interfaces.h"

namespace lgf
{
// a binary op is an operation taking exactly two values as inputs
// and produce eactly one outputValue value
class binaryOp : public operation{
    public:
    binaryOp() = default;
    static binaryOp* build(value &lhs, value &rhs){
        auto op = new binaryOp();
        op->registerInput(lhs, rhs);
        op->createValue();
        return op;
    }
    value& lhs() {return inputValue(0);}
    value& rhs() {return inputValue(1);}
};
class defOp : public operation{
public: 
    defOp() = default;
    static defOp* build(std::string id_="Unknown"){
        auto op = new defOp();
        auto& val = op->createValue();
        val.setSID(id_);
        op->setSID("Def");
        return op;
    }
    std::string represent() final{
        printer p;
        p<<outputValue().represent();
        p<< " = "<<getSID();
        return p.dump();
    }
};

class addOp : public binaryOp, public commutable{
    public :
    addOp() = default;
    static addOp* build(value& lhs, value& rhs){
        auto op = new addOp();
        op->registerInput(lhs, rhs);
        op->setSID("Add");
        return op;
    }
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : "<<lhs().represent() << " + "<<rhs().represent();
        return p.dump();
    }
};

class multiplyOp : public binaryOp {
    public :
    public :
    multiplyOp() = default;
        
    static multiplyOp* build(value& lhs, value& rhs){
        auto op = new multiplyOp();
        op->setSID("Multiply");
        op->registerInput(lhs, rhs);
        return op;
    }
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : "<<lhs().represent() << " * "<<rhs().represent();
        return p.dump();
    }
};

class sumOp : public operation, public commutable{
    public:
    sumOp() = default;
    template <typename... ARGS>
    static sumOp* build(ARGS &...args) {
        auto op = new sumOp();
        op->registerInput(args...);
        op->setSID("Sum");
        op->createValue();
        return op;
    }
    static sumOp* build(std::vector<value> vals){
        auto op = new sumOp();
        op->createValue();
        op->setSID("Sum");
        for(auto& v : vals)
            op->registerInput(v);
        return op;
    }
    void addInput(value& val) { registerInput(val); }

    std::string represent() override {
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : ";
        id_t n = getInputSize(), i=0;
        for(auto j =0 ;j<n; j++){
            p<<inputValue(j).represent();
            i++;
            if(i!= n) p<<" + ";
        }
        return p.dump();
    }
};
}

#endif