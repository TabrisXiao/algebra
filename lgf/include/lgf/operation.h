
#ifndef OPERATION_H_
#define OPERATION_H_
#include <unordered_set>
#include <queue>
#include <map>
#include <set>
#include <string>
#include "printer.h"
//#include "dgraph.h"
#include "global.h"
#include "exception.h"

// logic graph frameworks
namespace lgf{
class context;
class operation;
class dependence;
class value;
class valueRef;
class graph;
typedef size_t id_t;

class objInfo {
    public: 
    objInfo() {};
    objInfo(const objInfo & info){
        tid = info.getTypeID();
        traceID = info.getTraceID();
    }
    void setTraceID(int id_){traceID = id_;}
    void setTypeID(const char * _id){tid = _id;}
    void setTypeID(std::string& _id){tid = _id;}
    int getTraceID() const {return traceID;}
    std::string getTypeID() const { 
        return tid; }
    void printTraceID(std::ostream os){ os<<traceID; }

    private:
    int traceID = -1;
    std::string tid="Unknown";
};

template<typename obj>
obj& get_vec_elem_with_check(size_t n, std::vector<obj>& vec){
    auto vector_size = vec.size();
    CHECK_CONDITION(n<vector_size, "The query index exceeded the vector size.");
    return vec[n];
}

class value : public objInfo{
public:
    value() = default;
    value(operation * op, id_t id);
    virtual ~value() = default;
    virtual std::string represent(); 
    void print();
    operation * getDefiningOp() const {return defop;}
    template<class opType>
    opType* getDefiningOp() const {return dynamic_cast<opType*>(getDefiningOp());}
    void setDefiningOp(operation *op){defop = op;} 
    
    // get the list of all ops using this value as a input;
    std::vector<operation*> getUsers();
    // this iid is used to build connection between the value and
    // dependence. The iid has to be the same in the value and 
    // the dependence containning this value.
    id_t getIID() const { return iid; }

    private:
    operation *defop = nullptr;
    const id_t iid = -1;
};

class valueRef{
    public:
    valueRef() = default;
    valueRef(value &val){ referTo(val); }
    ~valueRef() = default;
    
    void referTo( value &val){
        bInitiated = 1;
        defop = val.getDefiningOp();
        iid = val.getIID();
    }
    operation* getDefiningOp() const {return defop;}
    id_t getIID() const {return iid;}
    value & getValue();

    private:
    operation * defop = nullptr;
    bool bInitiated = 0;
    id_t iid;
};

class operation : public objInfo{
public : 
    operation() : objInfo(){ outputs.reserve(3); };
    virtual ~operation() = default;
    virtual std::string represent() = 0;
    virtual void printOp();
    void print();
    value& getValueByID(id_t id){
        return outputs[id];
    }
    valueRef* inputRefByValue(const value & v);
    void linkTo(operation *op){
        outgoings.insert(op); 
        op->incomings.insert(this);
    }
    void linkFrom(operation *op){
        op->outgoings.insert(this); 
        incomings.insert(op);
    }
    void breakLinkTo(operation *op){
        outgoings.erase(op);
        op->incomings.erase(this);
    }
    void breakLinkFrom(operation *op){
        op->outgoings.erase(this);
        incomings.erase(op);
    }

    // reduce link will reduce the link counts by 1 or remove it
    // if there is only one connection exists.
    void reduceLinkTo(operation *op){
        if(outgoings.count(op) == 0) return; 
        outgoings.erase(outgoings.find(op));
        op->incomings.erase(incomings.find(this));
    }
    void reduceLinkFrom(operation *op){
        if(incomings.count(op) == 0) return; 
        op->outgoings.erase(outgoings.find(this));
        incomings.erase(incomings.find(op));
    }
    // replace this operation by another operation.
    // the new operation must have the same output size as
    // the original one. 
    // These outputs are replaced by those from the new operation 
    // in the same order.
    // op old : input1,  ...        output1, output2, ...
    //                                 |        |
    // op new : input1, input2, ... output1, output2, ...
    void replaceBy(operation* new_op);
    
    // replace the n-th input by the value v. 
    // the connection to the original value will break
    // and link to the new operation owning v.
    void replaceInputValue(int n, value& v);
    template <typename... ARGS>
    void registInput(ARGS ...args)
    {
        // inputs are suppose to be value type.
        auto values = {args...};
        for (auto val : values)
        {
            inputs.push_back(valueRef(val));
            linkFrom(val.getDefiningOp());
        }
    }
    // register the input at the given position. Other inputs after 
    // that index will be push back by 1 pos.
    void registInputAt( value& val, int pos);
    
    // create a value as output from this op, the order is not 
    // changable as it related to the iid of that value.
    value& createValue();

    value& output(int n=0){return outputs[n];}
    value& input(int n=0) {return inputs[n].getValue();}
    size_t getInputSize() const;
    size_t getOutputSize() const;

    // assign trace id to the value created in this operation. 
    // The start of the trace id is specified by the argument n.
    void assignValueID(int& n);

    // drop all inputs to this operation, and remove all connects
    // associated to the op.
    void dropAllInputs();

    // detach the input edges from this operation and disconnect
    // the outputs from their users, this operation still connecting to
    // the output edges.
    // for example, detach v3 will leads to 
    //    v1                       v1
    //   / \                      /
    //  v2  v3    detach v3:     v2      v3
    //     /                              |
    //    v4                        v4
    //operation* detach();

    //void setTraceIDToOutput(context *ctx);
    int valueTraceIDStart = -1;

    bool isDependencyFullfilled();

    std::vector<value>& getOutputs() const {
        return const_cast<std::vector<value>&>(outputs);
    }

    std::multiset<operation*>& getOutgoings(){ return outgoings;}
    std::multiset<operation*>& getIncomings(){ return incomings;}

    void setActivation(bool a ){ bActive = a; }
    bool isActive(){return bActive; }

    void setExploration(bool a){ bExplored = a; }
    bool isExplored(){ return bExplored; }

    bool isRemovable(){ return bRemove; }
    void setRemovable(){ 
        bRemove = 1; }
    //void erase(){ detach(); bRemove = 1;}

    std::vector<value> getInputs(){
        std::vector<value> res;
        for(auto& ref : inputs){
            res.push_back(ref.getValue());
        }
        return res;
    }

    std::vector<valueRef>& getInputRefs(){ return inputs;}

    private:
    std::vector<valueRef> inputs;
    std::vector<value> outputs;
    // this function is used to determine if this operation contained
    // a region. If an op contained a region, it should override
    // this function.
    virtual graph* getSubgraph(){ return nullptr;}
    bool bActive = 1;
    bool bExplored = 0;

    // this is a member used to remove the operation efficiently. 
    // Should be used solely for removing process in graph.
    bool bRemove = 0;
    std::multiset<operation*> incomings;
    std::multiset<operation*> outgoings;
};

class graph {
public : 
    graph() = default;
    void print();
    inline void printOps();
    // A breadth-first walk function that is graph modification safe.
    // Call the callable at the begining of visiting each vertex.
    // The callable should return void.
    // The fn will be executed on each operation at most once. 
    // The operation ran by fn is marked as done. A operation will
    // got processed only if all inputs operations are done (All
    // dependences are processed).
    template<typename callable>
    void walk(callable && fn, bool checkDependency = 0, bool recycleInactive = 0, bool removeDisconnected = 0){
        std::queue<operation *> _vq;
        std::vector<operation *> vertice_buffer;
        for(auto op: entrances)
            _vq.push(op);
        while (_vq.size())
        {
            auto v = _vq.front();
            _vq.pop();
            if(!(v->isActive())){
                if(recycleInactive) v->setRemovable();
                continue;
            }
            if(v->isRemovable() || v->isExplored()) continue;
            v->setExploration(true);
            
            vertice_buffer.push_back(v);
            for(auto& vn : v->getOutgoings() )
            {
                if(checkDependency && !vn->isDependencyFullfilled()) continue;
                if (vn->isExplored()) continue;
                if(vn->isRemovable()) continue;
                _vq.push(vn);
            }
            fn(v);
        }

        // all the ops haven't been explored come from a disconnected graph
        // need to mark them as removable if we don't need them.
        if(removeDisconnected){
            for(auto & op : nodes ){
                if(op->isExplored()) continue;
                op->setRemovable();
            }
        }
        
        for (auto &v : vertice_buffer)
        {
            v->setExploration(false);
        }
        clean();
        return;
    }
    // add operation to this graph
    void addEntranceOp(operation* op){ entrances.push_back(op); }
    // regist the op in this graph into book, if this op has no
    // inputs, it will be added as an entrance op.
    void addOp(operation* op);

    void assignID(int n=0);

    // clean will remove all operations marked as removable;
    void clean();
    // entrances are the ops have no inputs

    // return how many operations graph contained
    int getNodeSize(){ return int(nodes.size()); }
    std::vector<operation*> & getNodeList(){return nodes;}
    private:
    std::vector<operation*> entrances;
    std::vector<operation*> nodes;
};

}

#endif