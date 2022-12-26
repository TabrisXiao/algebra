
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include "dgraph.h"
#include "global.h"
#include "utility.h"

// abstract operation graph

namespace aog{
class context;
class element : public dgl::edge{
public:
    void assignID(std::string &id){_id = id;}
    void assignTraceID(int value){traceID = value;}
    virtual void represent(std::ostream &os){
        os<<"%";
        if(traceID > -1) os<<traceID;
        else os<<"Unknown";
        os<<" <"<<_id<<">";
    }
    std::string _id = "UnknownType";
    int traceID = -1;
};

class operation : public dgl::vertex{
public : 
    operation() = default;
    virtual void represent(std::ostream &os) = 0;
    void print(){
        Xos<<"%"<<traceID<<"_"<<opName<<" : ";
        represent(Xos);
    }
    void addContext(context *_ctx);
    context* getContext(){return ctx;}
    void printTraceID(std::ostream os){ os<<traceID; }
    void assignTraceIDToOutput();
    int traceID = -1;
    int elementTraceIDStart = -1;
    std::string opName;
    context *ctx;
};

class context{
public : context () = default;
    virtual ~context(){}
    void registerOp(operation*op){
        _ops.push_back(op);
        op->addContext(this);
    }
    dgl::vertex* entry_point = nullptr;
    std::vector<operation*> _ops;

    void indent(){
        for (int i = 0; i < curIndent; i++)
            Xos << "  ";
    }
    void print(){
        Xos<<"Context {"<<std::endl;
        Indent idlv(curIndent);
        for(auto op : _ops){
            indent();
            //Xos<<"%"<<ops_counter++<<" : ";
            op->print();
        }
        Xos<<"}\n";
    }

    int ops_counter = 0;
    int elem_counter = 0;
    int curIndent=0;
};
}

#endif