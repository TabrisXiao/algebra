
#ifndef AAOPS_H_
#define AAOPS_H_
#include "opBuilder.h"
#include "pass.h"
#include "interfaces.h"

namespace aog
{
class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        defineElement();
        output()->setTypeID(id_);
        setTypeID("Def");
    }
    element* output(){return &(elements[0]);}
    std::string represent(context *ctx) final{
        printer p;
        p.addToken(output()->represent(ctx));
        p.addToken("=");
        p.addToken(getID());
        return p.getString();
    }
};
class addOp : public operation{
    public :
    addOp(element *lhs, element *rhs){
        acceptInput(lhs, rhs);
        defineElement();
        setTypeID("Add");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element* output(){return &(elements[0]);}
    std::string represent(context *ctx) final {
        printer p;
        p.addToken(output()->represent(ctx));
        p.addToken("=");
        p.addToken(getID());
        p.addToken(":");
        p.addToken(lhs()->represent(ctx));
        p.addToken("+");
        p.addToken(rhs()->represent(ctx));
        return p.getString();
    }
};

class multiplyOp : public operation {
    public :
    multiplyOp(element *lhs, element *rhs) {
        acceptInput(lhs, rhs);
        defineElement();
        setTypeID("Multiply");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element* output(){return &(elements[0]);}
    std::string represent(context *ctx) final{
        printer p;
        p.addToken(output()->represent(ctx));
        p.addToken("=");
        p.addToken(getID());
        p.addToken(":");
        p.addToken(lhs()->represent(ctx));
        p.addToken("*");
        p.addToken(rhs()->represent(ctx));
        return p.getString();
    }
};

class sumOp : public operation, public opGroup<commutable>{
    public:
    template <typename... ARGS>
    sumOp(ARGS *...args) {
        acceptInput(args...);
        defineElement();
        setTypeID("Sum");
    }
    sumOp(std::vector<element*> & values){
        for(auto val : values) acceptInput(val);
        defineElement();
        setTypeID("Sum");
    }
    element* output(){return &(elements[0]);}

    std::string represent(context *ctx) final {
        printer p;
        p.addToken(output()->represent(ctx));
        p.addToken("=");
        p.addToken(getID());
        p.addToken(":");
        int n = inputElements.size(), i=0;
        for(auto e : inputElements){
            p.addToken(e->represent(ctx));
            i++;
            if(i!= n) p.addToken("+");
        }
        return p.getString();
    }
};
}

#endif