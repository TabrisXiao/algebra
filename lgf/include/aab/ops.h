
#ifndef AAOPS_H_
#define AAOPS_H_
#include "lgf/operation.h"
#include "lgf/lgOps.h"
#include "groups.h"
//#include "lgf/group.h"
//#include "lgf/lgInterfaces.h"
//#include "pass.h"
//#include "interfaces.h"

namespace lgf
{
class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        auto& val = createValue();
        val.setTypeID(id_);
        setTypeID("Def");
    }
    std::string represent() final{
        printer p;
        p<<output().represent();
        p<< " = "<<getTypeID();
        return p.dump();
    }
};

class addOp : public binaryOp, public commutable{
    public :
    addOp(value& lhs, value& rhs) : binaryOp(lhs, rhs) {
        setTypeID("Add");
    }
    virtual std::string represent() override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " + "<<rhs().represent();
        return p.dump();
    }
};

class multiplyOp : public binaryOp {
    public :
    public :
    multiplyOp(value& lhs, value& rhs) : binaryOp(lhs, rhs) {
        setTypeID("Multiply");
    }
    virtual std::string represent() override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " * "<<rhs().represent();
        return p.dump();
    }
};

class sumOp : public operation, public commutable{
    public:
    template <typename... ARGS>
    sumOp(ARGS &...args) {
        registInput(args...);
        setTypeID("Sum");
        createValue();
    }
    sumOp(std::vector<value> vals){
        createValue();
        setTypeID("Sum");
        for(auto& v : vals)
            registInput(v);
    }
    void addInput(value& val) { registInput(val); }

    std::string represent() override {
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : ";
        id_t n = getInputSize(), i=0;
        for(auto j =0 ;j<n; j++){
            p<<input(j).represent();
            i++;
            if(i!= n) p<<" + ";
        }
        return p.dump();
    }
};
}

#endif