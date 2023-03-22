
#ifndef AAOPS_H_
#define AAOPS_H_
#include "aog.h"

namespace aog
{


class defOp : public operation{
public: 
    defOp(std::string id_="Unknown") {
        defineElement();
        output()->setID(id_);
        setID("Def");
    }
    element* output(){return &(elements[0]);}
    void represent(std::ostream &os, context *ctx){
        output()->represent(os,ctx);
    }
};
class addOp : public operation{
    public :
    addOp(element *lhs, element *rhs){
        acceptInput(lhs, rhs);
        defineElement();
        setID("Add");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element* output(){return &(elements[0]);}
    void represent(std::ostream &os, context *ctx){
        output()->represent(os, ctx);
        os<<" = ";
        lhs()->represent(os, ctx);
        os<<" + ";
        rhs()->represent(os, ctx);
    }
};

class multiplyOp : public operation {
    public :
    multiplyOp(element *lhs, element *rhs) {
        acceptInput(lhs, rhs);
        defineElement();
        setID("Multiply");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element* output(){return &(elements[0]);}
    void represent(std::ostream &os, context *ctx){
        output()->represent(os,ctx);
        os<<" = ";
        lhs()->represent(os, ctx);
        os<<" * ";
        rhs()->represent(os, ctx);
    }
};

class sumOp : public operation {
    public:

    template <typename... ARGS>
    sumOp(ARGS *...args) {
        acceptInput(args...);
        defineElement();
        setID("Sum");
    }
    sumOp(std::vector<element*> & values){
        for(auto val : values) acceptInput(val);
        defineElement();
        setID("Sum");
    }
    element* output(){return &(elements[0]);}

    void represent(std::ostream &os, context *ctx){
        output()->represent(os,ctx);
        os<<" = ";
        int n = inputElements.size(), i=0;
        for(auto e : inputElements){
            e->represent(os, ctx);
            i++;
            if(i!= n) os<<" + ";
        }
    }
};
}

#endif