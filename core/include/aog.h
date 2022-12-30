
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include "dgraph.h"
#include "global.h"

// abstract operation graph

namespace aog{
class context;
class operation;
class objInfo {
    public: 
    objInfo() = default;
    objInfo(context *ctx_): ctx(ctx_){
    }
    void setTraceID(int id_){traceID = id_;}
    void setTraceID();
    void setID(const char * _id){id = _id;}
    void setID(std::string& _id){id = _id;}
    void printIndent();
    context * getContext() {return ctx;}
    void printTraceID(std::ostream os){ os<<traceID; }
    context *ctx;
    int traceID = -1;
    std::string id="Unknown";
};

class element : public dgl::edge , public objInfo{
public:
    element() = default;
    element(context *ctx_): objInfo(ctx_){}
    virtual ~element(){}
    virtual void represent(std::ostream &os){
        os<<"%";
        if(traceID > -1) os<<traceID;
        else os<<"Unknown";
        os<<" <"<<id<<">";
    }
    operation* getDefiningOp();
    template<class opType>
    opType* getDefiningOp();
};

class operation : public dgl::vertex, public objInfo{
public : 
    operation(context *ctx_) : objInfo(ctx_){};
    virtual void represent(std::ostream &os) = 0;
    void print(){
        printIndent();
        Xos<<"%"<<traceID<<"_"<<id<<" : ";
        represent(Xos);
        Xos<<"\n";
    }
    void setContext(context *_ctx);
    context* getContext(){return ctx;}
    void setTraceIDToOutput();
    int elementTraceIDStart = -1;
};

class context{
public : context () = default;
    virtual ~context(){}
    void registerOp(operation*op){
        _ops.push_back(op);
        op->setContext(this);
    }
    std::vector<operation*>& getOps(){return _ops;}
    dgl::vertex* entry_point = nullptr;
    std::vector<operation*> _ops;
    void print();

    int ops_counter = 0;
    int elem_counter = 0;
    int curIndent=0;
};

class region : public dgl::graph{
public : 
    region() = default;
    region(context *ctx_){ctx = ctx_;}
    void printRegion();
    void printOps(){
        auto fn = [&](vertex* _op){
            auto op = dynamic_cast<operation*>(_op);
            op->setTraceID();
            op->setTraceIDToOutput();
            op->print();
        };
        BFWalk(fn);
    }
    context *ctx= nullptr;
};

class moduleOp : public operation, public region{
public:
    moduleOp(context *ctx, std::string _id="module") : 
    operation(ctx),
    block(ctx){
        setID(_id);
    }
    void represent(std::ostream &os){
        printRegion();
    }
    region block;
};

class builder {
public:
    builder(context *ctx_): ctx(ctx_){}
    template<typename obj, typename...ARGS>
    void setInsertPoint(operation* op){
        
    }
    void build(ARGS&...args){
    }
    context * ctx;
    region *insert_region=nullptr;
};

template<typename concreteOp>
class rewriter {
    public : rewriter(context *ctx_) {ctx=ctx;}
    virtual bool rewrite(concreteOp *op) = 0;
    int applyRewrite(operation* op){
        // rewrite return value: 
        // 1 matched and rewrited
        // 0 matched but failed rewritten
        // -1 didn't match
        if(auto cop = dynamic_cast<concreteOp*>(op))
            return int(rewrite(cop));
        else return -1;
    }
    context *ctx;
};

class passBase {
public :
    passBase () = default;
    virtual int run(operation* op) = 0;
};

class passManager{
    public : 
    passManager(context * _ctx): ctx(_ctx){}
    bool runPasses(){
        for(auto pass : passes){
            for(auto op : ctx->getOps()){
                if(pass->run(op)==0) return 1;
            }
        }
        return 0;
    }
    std::vector<passBase*> passes;
    context *ctx;
};
}

#endif