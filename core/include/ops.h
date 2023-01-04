
#ifndef AAOPS_H_
#define AAOPS_H_
#include "aog.h"

namespace aog
{


class defOp : public operation{
public: 
    defOp(context *ctx, std::string id_="Unknown") : operation(ctx) {
        _output = createElement();
        _output.setID(id_);
        setID("Def");
    }
    element& output(){return _output;}
    void represent(std::ostream &os){
        _output.represent(os);
    }
    element& _output;
};
class addOp : public operation{
    public :
    addOp(context *ctx, element &lhs, element &rhs): operation(ctx){
        addInputVertex(lhs, rhs);
        _output = createElement();s
        elhs=&lhs;
        erhs=&rhs;
        setID("Add");
    }
    element* lhs(){return elhs;}
    element* rhs(){return erhs;}
    element& output(){return _output;}
    void represent(std::ostream &os){
        _output.represent(os);
        os<<" = ";
        elhs->represent(os);
        os<<" + ";
        erhs->represent(os);
    }
    private:
    element *elhs, *erhs;
    element * _output;
};

class multiplyOp : public operation{
    public :
    multiplyOp(context *ctx, element &lhs, element &rhs): operation(ctx){
        addInputVertex(lhs, rhs);
        _output = createElement();
        elhs=&lhs;
        erhs=&rhs;
        setID("Multiply");
    }
    element* lhs(){return elhs;}
    element* rhs(){return erhs;}
    element& output(){return _output;}
    void represent(std::ostream &os){
        _output.represent(os);
        os<<" = ";
        elhs->represent(os);
        os<<" * ";
        erhs->represent(os);
    }
    private:
    element *elhs, *erhs;
    element * _output;
};
}

#endif