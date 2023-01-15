
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
        elhs=&lhs;
        erhs=&rhs;
        setID("Add");
    }
    element* lhs(){return elhs;}
    element* rhs(){return erhs;}
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
        os<<" = ";
        elhs->represent(os);
        os<<" + ";
        erhs->represent(os);
    }
    private:
    element *elhs, *erhs;
};

class multiplyOp : public operation{
    public :
    multiplyOp(context *ctx, element &lhs, element &rhs): operation(ctx){
        registerInput(lhs, rhs);
        reserveElement();
        elhs=&lhs;
        erhs=&rhs;
        setID("Multiply");
    }
    element* lhs(){return elhs;}
    element* rhs(){return erhs;}
    element& output(){return elements[0];}
    void represent(std::ostream &os){
        output().represent(os);
        os<<" = ";
        elhs->represent(os);
        os<<" * ";
        erhs->represent(os);
    }
    private:
    element *elhs, *erhs;
};
}

#endif