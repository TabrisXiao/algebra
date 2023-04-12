
#ifndef AAOPS_H_
#define AAOPS_H_
#include "lgf/operation.h"
//#include "lgf/group.h"
//#include "lgf/lgInterfaces.h"
//#include "pass.h"
//#include "interfaces.h"

namespace lgf
{


class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        auto val = createValue();
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

class addOp : public operation{
    public :
    addOp(value& lhs, value& rhs){
        registInput(lhs, rhs);
        setTypeID("Add");
        createValue();
    }
    value& lhs() {return input(0);}
    value& rhs() {return input(1);}
    virtual std::string represent() override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " + "<<rhs().represent();
        return p.dump();
    }
};

class multiplyOp : public operation {
    public :
    public :
    multiplyOp(value& lhs, value& rhs){
        registInput(lhs, rhs);
        setTypeID("Multiply");
        createValue();
    }
    value& lhs() {return input(0);}
    value& rhs() {return input(1);}
    virtual std::string represent() override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " * "<<rhs().represent();
        return p.dump();
    }
};

class sumOp : public operation{
    public:
    template <typename... ARGS>
    sumOp(ARGS &...args) {
        registInput(args...);
        setTypeID("Sum");
        createValue();
    }
    sumOp(){
        createValue();
        setTypeID("Sum");
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