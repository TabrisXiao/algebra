
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include "sdgraph.h"
#include "global.h"
#include <unordered_map>
#include <memory.h>

// abstract operation graph

namespace aog{
class context;
class operation;
class region;

class objInfo {
    public: 
    objInfo() = default;
    void setTraceID(int id_){traceID = id_;}
    void setTraceID(context * ctx);
    void setID(const char * _id){id = _id;}
    void setID(std::string& _id){id = _id;}
    void printIndent(context *ctx);
    void printTraceID(std::ostream os){ os<<traceID; }
    int traceID = -1;
    std::string id="Unknown";
};

class element : public objInfo{
public:
    element() = default;
    element(operation * op);
    virtual ~element(){}
    virtual void represent(std::ostream &os, context *ctx){
        os<<"%";
        if(traceID > -1) os<<traceID;
        else os<<"Unknown";
        os<<" <"<<id<<">";
    }
    operation* getDefiningOp();
    void addUser(operation* op){ users.push_back(op); }
    std::vector<operation*> & getUsers(){return users;}
    template<class opType>
    opType* getDefiningOp();
    operation *defOp = nullptr;
    std::vector<operation*> users;
};

class operation : public sdgl::vertex, public objInfo{
public : 
    operation() : objInfo(){};
    virtual void represent(std::ostream &os, context *ctx) = 0;
    virtual void print(context *ctx){
        printIndent(ctx);
        Xos<<id<<" : ";
        represent(Xos, ctx);
        Xos<<"\n";
    }
    template <typename... ARGS>
    void acceptInput(ARGS ...args)
    {
        auto elems = {args...};
        std::unordered_map<operation*, bool> buffer;
        for (auto e : elems)
        {
            auto ptr = dynamic_cast<element*>(e)->getDefiningOp();
            if(buffer.find(ptr) == buffer.end()) {
                buffer[ptr] = 1;
                linkFrom(*ptr);
            }
            e->addUser(this);
            inputElements.push_back(e);
        }
    }
    std::vector<element*> & getInputs(){return inputElements;}
    void defineElement(int n = 1){
        for(auto i=0; i<n; i++){
            elements.push_back(element(this));
        }
    }
    int getOutputSize(){return int(elements.size());}
    std::vector<element>& getOutputs(){return elements;}
    element & getOutput(int n){ return elements[n];}
    void setTraceIDToOutput(context *ctx);
    int elementTraceIDStart = -1;
    std::vector<element> elements;
    std::vector<element*> inputElements;
};

class context{
public : 
    context () = default;
    virtual ~context(){}

    int ops_counter = 0;
    int elem_counter = 0;
    int curIndent=0;
    void resetCounts(){
        ops_counter = 0;
        elem_counter = 0;
        curIndent=0;
    }
    region* getRegion(){return _region;}
    region * _region = nullptr;
    context* parent_ctx = nullptr, *root_ctx = nullptr;
};

class region : public sdgl::sdgraph{
public : 
    region() = default;
    void printRegion(context *ctx);
    inline void printOps(context *ctx){
        getEntryVertex().BFWalk([&](sdgl::vertex* _op){
            if(auto op = dynamic_cast<operation*>(_op)){
                op->setTraceID(ctx);
                op->setTraceIDToOutput(ctx);
                op->print(ctx);
            }
        });
    }
};

class moduleOp : public operation{
public:
    moduleOp(std::string _id="module") {
        setID(_id);
        block.getEntry().hasOutput();  
    }
    region& getRegion(){return block;}
    void represent(std::ostream &os, context *ctx){
        block.printRegion(ctx);
    }
    region block;
};

class opModifier {
    public : 
    opModifier() = default;
    template<typename obj, typename...ARGS>
    obj* create(ARGS ...args){
        auto ptr = new obj(args...);
        if(!(ptr->hasInput())){
            reg->addL1Vertex(dynamic_cast<sdgl::vertex*>(ptr));
        }
        return ptr;
    }
    void replaceOperation(operation *origOp, operation *newOp){
        auto out_ops = origOp->getOutVertices();
        for(auto op_ : out_ops){
            newOp->linkTo(*op_);
        }
        origOp->detach();
    }
    // replace the element by the new element for all users
    void replaceElement(element *e, element* newe){
        auto & vec = e->getUsers();
        for(auto op : vec){
            auto & inputs = op->getInputs();
            for(auto i=0 ; i<inputs.size(); i++){
                if(inputs[i]==e){
                    inputs[i]=newe;
                    break;
                }
            }
        }
    }
    // create a new type T op and replace the origOp.
    // for both operations has to have only one elements;
    template<typename T, typename...ARGS>
    T* replaceOp(operation *origOp, ARGS ...args){
        auto op = create<T>(args...);
        // the input vertices is linked when creating the op, 
        // so we only need to link the output vertices
        replaceElement(&(origOp->getOutput(0)), &(op->getOutput(0)));
        replaceOperation(origOp, op);
        return op;
    }
    void setWorkRegion(region * reg_) { reg = reg_;}
    region *reg = nullptr;
};

class opBuilder : public opModifier{
    class regionPtrHelper{
    public : 
        regionPtrHelper(opBuilder *builder) : ptr(builder){}
        ~regionPtrHelper(){ 
        }
        region *currentRegion=nullptr, *previousRegion=nullptr;
        opBuilder* ptr= nullptr;
    };
public:
    opBuilder(context *ctx_) {
        ctx = ctx_;
        entranceModule = new moduleOp("module");
        setWorkRegion(&(entranceModule->getRegion()));
    }
   
    context * getContext(){return ctx;}
    moduleOp *entranceModule = nullptr;
    context *ctx;
};

class opRewriter : public opModifier{
    public : 
    opRewriter(moduleOp *op){
        entranceModule = op;
        setWorkRegion(&(entranceModule->getRegion()));
    }
    moduleOp *entranceModule = nullptr;
};

class rewriterBase {
    public:
    rewriterBase() = default;
    virtual ~rewriterBase() = default;
    virtual int execute(opRewriter &, operation * op) = 0;
};

template<typename concreteOp>
class rewriter : public rewriterBase{
    public : rewriter() = default;
    virtual bool rewrite(opRewriter &, concreteOp *op) = 0;
    virtual int execute(opRewriter & rewriter,operation* op) override final{
        // rewrite return value: 
        // 1 matched and rewrited
        // 0 matched but failed rewritten
        // -1 didn't match
        if(auto cop = dynamic_cast<concreteOp*>(op))
        {
            return int(rewrite(rewriter, cop));
        }
        else {
            return -1;
        }
    }
    opBuilder * builder;
};

class passBase {
public :
    passBase (const char * name) : _pass_name (name) {}
    void run(opRewriter & rewriter, operation* op) {
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(rewriter, op);
        }
    }
    template<typename T, typename ...ARGS>
    void addRewriter(ARGS...arg){ 
        auto ptr = std::make_unique<T>(arg...);
        rewriters.push_back(std::move(ptr));
    }
    std::vector<std::unique_ptr<rewriterBase>> rewriters;
    std::string _pass_name;
};

class passManager{
    public : 
    passManager(moduleOp * op): 
        entranceOp(op),
        _rw(op)
        {}
    bool runPasses(){
        return 0;
    }
    void run(){
        region * reg = &(entranceOp->getRegion());
        for(auto pass=passes.begin(); pass!=passes.end(); pass++){
            std::cout<<" running pass: "<<(*pass).get()->_pass_name<<std::endl;
            std::cout<<std::endl;
            runPassThroughRegion(reg, (*pass).get());
        }
    }
    bool runPassThroughRegion(region* reg, passBase* pass){
        reg->getEntryVertex().BFWalk([&](sdgl::vertex* _op){
            auto op = dynamic_cast<operation*>(_op);
            pass->run(_rw, op);
        });
        return 0;
    }

    void addPass(std::unique_ptr<passBase> ps){
        passes.push_back(std::move(ps));
    }
    std::vector<std::unique_ptr<passBase>> passes;
    moduleOp * entranceOp;
    opRewriter _rw;
};
}

#endif