
#ifndef OPERATION_H_
#define OPERATION_H_
#include <unordered_set>
#include <string>
#include "printer.h"
#include "sdgraph.h"
#include "global.h"
#include "exception.h"

namespace aog{
class context;
class operation;
class region;

class objInfo {
    public: 
    objInfo() = default;
    void setTraceID(int id_){traceID = id_;}
    void setTraceID(context * ctx);
    void setTypeID(const char * _id){tid = _id;}
    void setTypeID(std::string& _id){tid = _id;}
    std::string getID(){return tid;}
    std::string getTypeID(){return tid;}
    void printIndent(context *ctx);
    void printTraceID(std::ostream os){ os<<traceID; }
    int traceID = -1;
    std::string tid="Unknown";
};

class element : public objInfo{
public:
    element() = default;
    element(operation * op);
    virtual ~element(){}
    virtual std::string represent(context *ctx){
        printer p;
        p.addString("%");
        std::string id = traceID > -1 ? std::to_string(traceID) :"Unknown";
        p.addString(id);
        p.addToken("<"+tid+">");
        return p.getString();
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
    virtual std::string represent(context *ctx) = 0;
    virtual void printOp(context *ctx){
        printIndent(ctx);
        Xos<<represent(ctx);
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
    template<typename type>
    bool isInherit(){
        if(auto p = dynamic_cast<type*>(this)) return true;
        return false;
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
    inline void printOps(context *ctx);
};

class moduleOp : public operation{
public:
    moduleOp();
    ~moduleOp(){std::cout<<"deleted"<<std::endl;}
    region* getRegion(){return &block;}
    std::string represent(context *ctx){return getTypeID();}
    virtual void printOp(context *ctx) override final{
        printIndent(ctx);
        Xos<<getTypeID()<<" : ";
        block.printRegion(ctx);
        Xos<<"\n";
    }
    region block;
};

class context{
public : 
    context () {
        op = new moduleOp();
        _region = op->getRegion();
    }
    virtual ~context(){}
    // assign ID to each operations or elements contained in this context
    void assignID();
    int ops_counter = 0;
    int elem_counter = 0;
    int curIndent=0;
    void resetCounts(){
        ops_counter = 0;
        elem_counter = 0;
        curIndent=0;
    }
    region * getRegion(){
        CHECK_CONDITION(_region!=nullptr);
        return _region;
        }
    moduleOp * getModuleOp(){return op;}
    //region* getRegion(){return _region;}
    region * _region = nullptr;
    moduleOp * op;
};

}

#endif