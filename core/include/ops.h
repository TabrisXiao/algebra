
#ifndef AAOPS_H_
#define AAOPS_H_
#include "aog.h"

namespace aog
{


class defOp : public operation{
public: 
    defOp(context *ctx, std::string id_="Unknown") : operation(ctx) {
        reserveElement();
        output().setID(id_);
        setID("Def");
    }
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
    }
};
class addOp : public operation{
    public :
    addOp(context *ctx, element &lhs, element &rhs): operation(ctx){
        registerInput(lhs, rhs);
        reserveElement();
        setID("Add");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
        os<<" = ";
        lhs()->represent(os);
        os<<" + ";
        rhs()->represent(os);
    }
};

class multiplyOp : public operation{
    public :
    multiplyOp(context *ctx, element &lhs, element &rhs): operation(ctx){
        registerInput(lhs, rhs);
        reserveElement();
        setID("Multiply");
    }
    element* lhs(){return inputElements[0];}
    element* rhs(){return inputElements[1];}
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
        os<<" = ";
        lhs()->represent(os);
        os<<" * ";
        rhs()->represent(os);
    }
};

class sumOp : public operation{
    public:

     template <typename... ARGS>
    sumOp(context *ctx, ARGS &...args): operation(ctx){
        registerInput(args...);
        reserveElement();
        setID("Sum");
    }
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
        os<<" = ";
        int n = inputElements.size(), i=0;
        for(auto e : inputElements){
            e->represent(os);
            i++;
            if(i!= n) os<<" + ";
        }
    }
};
}

#endif