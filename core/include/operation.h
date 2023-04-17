
#ifndef OPERATION_H_
#define OPERATION_H_
#include <unordered_set>
#include <string>
#include "printer.h"
#include "dgraph.h"
#include "global.h"
#include "exception.h"

namespace aog{
class context;
class operation;
class dependency;
class region;

class objInfo {
    public: 
    objInfo() = default;
    objInfo(const objInfo & info){
        tid = info.getTypeID();
        traceID = info.getTraceID();
    }
    void setTraceID(int id_){traceID = id_;}
    void setTypeID(const char * _id){tid = _id;}
    void setTypeID(std::string& _id){tid = _id;}
    int getTraceID() const {return traceID;}
    std::string getTypeID() const {return tid;}
    void printTraceID(std::ostream os){ os<<traceID; }

    private:
    int traceID = -1;
    std::string tid="Unknown";
};

class value : public objInfo{
public:
    value() = default;
    value(const value & val);
    value(operation * op, int id);
    virtual ~value() = default;
    virtual std::string represent() const; 
    void print() const;
    template<class opType>
    opType* getDefiningOp(){ return dynamic_cast<opType*>(val->getDefiningOp()); }
    void setDefiningOp(operation *op){defOp = op;}
    operation * getDefiningOp() const {return defOp;}
    std::vector<operation*> getUsers() const;
    dependency* atDependency() const;
    // this iid is used to build connection between the value and
    // dependency. The iid has to be the same in the value and 
    // the dependency containning this value.
    int getIID() const { return iid; }

    // drop the specified user
    void dropUser(operation* op);

    private:
    operation *defOp = nullptr;
    const int iid = -1;
};

class dependency : public dgl::dedge {
    public : 
    dependency() = default;
    dependency(operation* op, int id): val(op, id), iid(id) {}
    bool checkIID(int id) const {return iid == id;}
    int getIID() const {return iid;}
    void print() const { val.print(); }
    value * atValue(){ return &val; }
    operation * getDefiningOp() const {return val.getDefiningOp();}
    const value& getValue() const {return val;}
    // internal id
    const int iid = -1;
    value val;
};

class operation : public dgl::vertex, public objInfo{
public : 
    operation() : objInfo(){};
    virtual ~operation(){
    }
    virtual std::string represent()const = 0;
    virtual void printOp();
    void print();
    // get the ptr to the dependency contained in this operation, having the specified iid.
    dependency * atDependency(int iid);
    template <typename... ARGS>
    void registInput(ARGS ...args)
    {
        // inputs are suppose to be value type.
        auto values = {args...};
        for (auto& val : values)
        {
            if(auto d = val.atDependency()){
                d->connect(val.getDefiningOp(), this);
            }
        }
    }
    value * createValue();

    // any functions getting the inputs should be const as they are not
    // suppose to modify the inputs. Any changes for the inputs should 
    // happen inside of the operation defining them.
    const value& getInput(int n =0 ) const;
    //int getInputSize() const;

    // assign trace id to the value created in this operation. 
    // The start of the trace id is specified by the argument n.
    void assignValueID(int& n);

    // disconnect from the dependency of the specified value
    void dropInput(value &val);

    // detach the input edges from this operation and disconnect
    // the outputs from their users, this operation still connecting to
    // the output edges.
    // for example, detach v3 will leads to 
    //    v1                       v1
    //   / \                      /
    //  v2  v3    detach v3:     v2      v3
    //     /                              |
    //    v4                        v4
    void detach();
    
    // replace the old value by the new value
    void replaceInputValue(value & oldV, value & newV);
    
    // template<typename type>
    // bool isInherit(){
    //     if(auto p = dynamic_cast<type*>(this)) return true;
    //     return false;
    // }
    int getOutputSize() const {return int(outEdges.size());}
    //std::vector<value>& getOutputs() {return values;}
    value * atOutput(int n=0){ return outputs[n].atValue();}
    const value& getOutput(int n=0) const {return outputs[n].getValue();}
    //void setTraceIDToOutput(context *ctx);
    int valueTraceIDStart = -1;
    std::vector<dependency> outputs;
};

class region : public dgl::graph{
public : 
    region() = default;
    void printRegion();
    inline void printOps();
    template<typename callable>
    void walk(callable && fn ){
        auto callwrapper = [&](dgl::vertex* vert){
            auto op = dynamic_cast<operation*>(vert);
            fn(op);
        };
        BFWalk(callwrapper);
    }
    void assignID(int n=0);
};

class moduleOp : public operation{
public:
    moduleOp();
    ~moduleOp(){std::cout<<"deleted"<<std::endl;}
    region* getRegion(){return &block;}
    std::string represent() const {return getTypeID();}
    virtual void printOp() override final{
        global::stream::getInstance().printIndent();
        global::stream::getInstance()<<getTypeID()<<" : ";
        block.printRegion();
        global::stream::getInstance()<<"\n";
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
    // assign ID to each operations or values contained in this context
    void assignID(){}
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