
#ifndef AAOPS_H_
#define AAOPS_H_
#include "operation.h"
//#include "pass.h"
//#include "interfaces.h"

namespace aog
{
class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        createValue();
        setTypeID("Def");
    }
    const value& output() const {return getOutput();}
    std::string represent() const final{
        printer p;
        p<<output().represent();
        p<< " = "<<getTypeID();
        return p.dump();
    }
};

class addOp : public operation{
    public :
    addOp(value lhs, value rhs){
        registInput(lhs, rhs);
        setTypeID("Add");
        createValue();
    }
    const value& lhs() const{return inputRef(0);}
    const value& rhs() const{return inputRef(1);}
    const value& output() const {return getOutput();}
    virtual std::string represent() const override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " + "<<rhs().represent();
        return p.dump();
    }
};

class multiplyOp : public operation {
    public :
    public :
    multiplyOp(value lhs, value rhs){
        registInput(lhs, rhs);
        setTypeID("Multiply");
        createValue();
    }
    const value& lhs() const{return inputRef(0);}
    const value& rhs() const{return inputRef(1);}
    const value& output() const {return getOutput();}
    virtual std::string represent() const override{
        printer p;
        p<<output().represent();
        p<<" = "<<getTypeID()<<" : "<<lhs().represent() << " * "<<rhs().represent();
        return p.dump();
    }
};

// class sumOp : public operation, public opGroup<commutable>{
//     public:
//     template <typename... ARGS>
//     sumOp(ARGS *...args) {
//         acceptInput(args...);
//         defineElement();
//         setTypeID("Sum");
//     }
//     sumOp(std::vector<value*> & values){
//         for(auto val : values) acceptInput(val);
//         defineElement();
//         setTypeID("Sum");
//     }
//     value* output(){return &(values[0]);}

//     std::string represent(context *ctx) final {
//         printer p;
//         p.addToken(output()->represent(ctx));
//         p.addToken("=");
//         p.addToken(getID());
//         p.addToken(":");
//         int n = inputElements.size(), i=0;
//         for(auto e : inputElements){
//             p.addToken(e->represent(ctx));
//             i++;
//             if(i!= n) p.addToken("+");
//         }
//         return p.getString();
//     }
// };
}

#endif