
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
    binaryOp(value &lhs, value &rhs){
        registerInput(lhs, rhs);
        createValue();
    }
    value& lhs() {return inputValue(0);}
    value& rhs() {return inputValue(1);}
};
class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        auto& val = createValue();
        val.setSID(id_);
        setSID("Def");
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
    addOp(value& lhs, value& rhs) : binaryOp(lhs, rhs) {
        setSID("Add");
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
    multiplyOp(value& lhs, value& rhs) : binaryOp(lhs, rhs) {
        setSID("Multiply");
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
    template <typename... ARGS>
    sumOp(ARGS &...args) {
        registerInput(args...);
        setSID("Sum");
        createValue();
    }
    sumOp(std::vector<value> vals){
        createValue();
        setSID("Sum");
        for(auto& v : vals)
            registerInput(v);
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