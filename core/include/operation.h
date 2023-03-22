
#ifndef OPERATION_H_
#define OPERATION_H_
#include <unordered_set>
#include "sdgraph.h"
#include "global.h"

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
    std::string getID(){return id;}
    void printIndent(context *ctx);
    void printTraceID(std::ostream os){ os<<traceID; }
    int traceID = -1;
    std::string id="Unknown";
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
    std::vector<operation*> getUsers();
    template<class opType>
    opType* getDefiningOp(){
        return  dynamic_cast<opType*>(defOp);
    }
    operation *defOp = nullptr;
};

class operation : public sdgl::vertex, public objInfo{
public : 
    operation() : objInfo(){};
    virtual ~operation(){
        inputElements.clear();
    }
    virtual void represent(std::ostream &os, context *ctx) = 0;
    virtual void printOp(context *ctx){
        printIndent(ctx);
        Xos<<id<<" : ";
        represent(Xos, ctx);
        Xos<<"\n";
    }
    void print(context *ctx);
    template <typename... ARGS>
    void acceptInput(ARGS ...args)
    {
        auto elems = {args...};
        std::unordered_set<operation*> buffer;
        for (auto e : elems)
        {
            auto ptr = dynamic_cast<element*>(e)->getDefiningOp();
            // linkFrom function prevents to add a node already added.
            // so no worry to check again.
            linkFrom(*ptr);
            //push_back_unique_ptr<element*>(e, inputElements);
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
    std::size_t hashID;
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
                op->printOp(ctx);
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

}

#endif