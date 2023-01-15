
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include "sdgraph.h"
#include "global.h"
#include <unordered_map>

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

class element : public objInfo{
public:
    element() = default;
    element(context *ctx_): objInfo(ctx_){}
    element(operation * op);
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
    operation *defOp = nullptr;
};

class operation : public sdgl::vertex, public objInfo{
public : 
    operation(context *ctx_) : objInfo(ctx_){};
    virtual void represent(std::ostream &os) = 0;
    virtual void print(){
        printIndent();
        Xos<<id<<" : ";
        represent(Xos);
        Xos<<"\n";
    }
    template <typename... ARGS>
    void registerInput(ARGS &...args)
    {
        auto elems = {&args...};
        std::unordered_map<operation*, bool> buffer;
        for (auto e : elems)
        {
            auto ptr = dynamic_cast<element*>(e)->getDefiningOp();
            if(buffer.find(ptr) == buffer.end()) {
                buffer[ptr] = 1;
                linkFrom(*ptr);
            }
        }
    }
    void reserveElement(int n = 1){
        for(auto i=0; i<n; i++){
            elements.push_back(element(this));
        }
    }
    void setContext(context *_ctx);
    context* getContext(){return ctx;}
    std::vector<element>& getOutputs(){return elements;}
    element & getOutput(int n){ return elements[n];}
    void setTraceIDToOutput();
    int elementTraceIDStart = -1;
    std::vector<element> elements;
};

class context{
public : context () = default;
    virtual ~context(){}

    int ops_counter = 0;
    int elem_counter = 0;
    int curIndent=0;
};

class region : public sdgl::sdgraph{
public : 
    region() = default;
    region(context *ctx_){ctx = ctx_;}
    void printRegion();
    inline void printOps(){
        getEntryVertex().BFWalk([&](sdgl::vertex* _op){
            if(auto op = dynamic_cast<operation*>(_op)){
                op->setTraceID();
                op->setTraceIDToOutput();
                op->print();
            }
        });
    }
    context *ctx= nullptr;
};

class moduleOp : public operation{
public:
    moduleOp(context *ctx, std::string _id="module") : 
    operation(ctx),
    block(ctx){
        setID(_id);
        block.getEntry().hasOutput();  
    }
    region& getRegion(){return block;}
    void represent(std::ostream &os){
        block.printRegion();
    }
    region block;
};

class opBuilder {
    class regionPtrHelper{
    public : 
        regionPtrHelper(opBuilder *builder) : ptr(builder){}
        ~regionPtrHelper(){ 
        }
        region *currentRegion=nullptr, *previousRegion=nullptr;
        opBuilder* ptr= nullptr;
    };
public:
    opBuilder(context *ctx_) : ctx(ctx_){
        entranceModule = new moduleOp(ctx, "module");
        setInsertPoint(&(entranceModule->getRegion()));
    }
    void setInsertPoint(region* reg){
        currentRegion = reg;
    }
    template<typename obj, typename...ARGS>
    obj* create(ARGS &...args){
        auto ptr = new obj(ctx, args...);
        if(!(ptr->hasInput())){
            currentRegion->addL1Vertex(dynamic_cast<sdgl::vertex*>(ptr));
        }
        return ptr;
    }
    context * ctx;
    moduleOp *entranceModule = nullptr;
    region *currentRegion=nullptr;
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
    void remove(operation* op){
        delete op;
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
        return 0;
    }
    bool runPassThroughRegion(region* reg, passBase* pass){
        reg->getEntry().BFWalk([&](sdgl::vertex* _op){
            auto op = dynamic_cast<operation*>(_op);
            pass->run(op);
        });
        return 0;
    }
    std::vector<passBase*> passes;
    context *ctx;
};
}

#endif