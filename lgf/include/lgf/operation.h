
#ifndef OPERATION_H_
#define OPERATION_H_
#include "type.h"
#include <unordered_set>
#include <queue>
#include <map>
#include <set>
#include <string>
#include "printer.h"
//#include "dgraph.h"
#include "global.h"
#include "exception.h"
#include <memory>
#include <algorithm>

// logic graph frameworks
namespace lgf{
class context;
class operation;
class dependence;
class value;
class valueRef;
class graph;

typedef size_t id_t;


//symbolic id;
using sid_t = std::string;
class objInfo {
    public: 
    objInfo() = default;
    objInfo(const objInfo & info) {
        traceID = info.getTraceID();
        sid=info.getSID();
    }
    objInfo(std::string id) {sid = id;}
    void setTraceID(int id_){traceID = id_;}
    int getTraceID() const {return traceID;}

    sid_t getSID() const {return sid;}
    void setSID(sid_t id){sid=id;}

    private:
    int traceID = -1;
    sid_t sid;
    //type_t type_;
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
    value(operation * op);
    value(operation *op, type_t type, std::string sid);

    virtual ~value() = default;
    virtual std::string represent(); 
    void print();
    void setType(type_t tp) {vtp = tp;}
    void setTypeID(const char * _id){vtp.setID(_id);}
    void setTypeID(std::string& _id){vtp.setID(_id);}
    type_t getType(){return vtp;}
    template<typename t>
    t& getType(){ return dynamic_cast<t>(vtp); }
    // get type representation;
    std::string getTR() const { 
        return vtp.represent(); }
    operation * getDefiningOp() const {return defop;}
    template<class opType>
    opType* getDefiningOp() const {return dynamic_cast<opType*>(getDefiningOp());}
    void setDefiningOp(operation *op){defop = op;} 

    void addUsesr(operation *op){
        if(std::find(users.begin(), users.end(), op)!=users.end()){
            std::cout<<"lgf::value::addUser Runtime Warning: user exists!"<<std::endl;
            return;
        }
        users.push_back(op);
    }
    void dropUser(operation *op);
    void dropUsers();
    template<typename t>
    void type_guard(){
        if(dynamic_cast<t>(vtp)) return;
        std::cerr<<"Value type mismatch!"<<std::endl;
        std::exit(EXIT_FAILURE); 
    }
    
    // get the list of all ops using this value as a input;
    std::vector<operation*> getUsers();

    private:
    operation *defop = nullptr;
    std::vector<operation*> users;
    type_t vtp;
};

class dependencyValue : public value {
    public:
    dependencyValue() = default;
    dependencyValue(operation * op) : value(op) { setSID("dummy"); }
    std::string represent() {
        return "";
    }
};

class operation : public objInfo{
public : 
    // the first output value is dependency value used to store 
    // the dependency inform that don't have value connections
    operation(std::string id="op", graph * g=nullptr) : 
    objInfo(id){ 
        auto ptr = std::make_unique<dependencyValue>(this);
        outputs.push_back(std::move(ptr));
        graph_ = g;
    };
    virtual ~operation() = default;
    virtual std::string represent() {
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<representInputs();
        return p.dump();
    }
    virtual std::string representInputs();
    virtual std::string representOutputs();
    virtual void print();

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
    void replaceInputValue(int n, value* v);
    template <typename... ARGS>
    void registerInput(ARGS ...args)
    {
        // inputs are suppose to be value type.
        auto values = {args...};
        for (auto val : values)
        {
            val->addUsesr(this);
            inputs.push_back(val);
        }
    }
    void registerInputs(std::vector<value*> &args){
        for(auto ptr : args){
            registerInput(ptr);
        }
    }
    // register the input at the given position. Other inputs after 
    // that index will be push back by 1 pos.
    void registerInputAt( value* val, int pos);
    
    // Attach an op means this op will depends on that one and
    // it is connected as an acceptor of the op.
    dependencyValue* getIndirectDependecyValue() { 
        return dynamic_cast<dependencyValue*>(outputs[0].get()); }
    void appendTo(operation *op){
        auto dep = op->getIndirectDependecyValue();
        this->registerInput(dep);
    }
    void dependOn(operation* op){
        this->registerInput(op->outputValue(0));
    }
    value* createValue();
    value* createValue(type_t& type, std::string sid);

    value* outputValue(int n=0){return outputs[n].get();}
    value* inputValue(int n=0) {return inputs[n];}
    size_t getInputSize() const;
    size_t getOutputSize() const;

    // assign trace id to the value created in this operation. 
    // The start of the trace id is specified by the argument n.
    virtual void assignValueID(int& n);

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

    std::vector<std::unique_ptr<value>>& getOutputs() const {
        return const_cast<std::vector<std::unique_ptr<value>>&>(outputs);
    }

    void setActivation(bool a ){ bActive = a; }
    bool isActive(){return bActive; }

    void setExploration(bool a){ bExplored = a; }
    bool isExplored(){ return bExplored; }

    bool isRemovable(){ return bRemove; }
    void setRemovable(){ 
        bRemove = 1; }
    //void erase(){ detach(); bRemove = 1;}

    graph* getParentGraph(){return graph_;}
    void setParentGraph(graph* g){ graph_ = g; }
    virtual bool verify() { return 0; }

    std::vector<value*>& getInputs(){ return inputs;}
    graph * expandToGraph();

    private:
    std::vector<value*> inputs;
    std::vector<std::unique_ptr<value>> outputs;
    // dependency stores the ops depending on this op.
    // Additional those ops don't have the value connecting to this op
    // other ops using the output from this op clearly depends on this op
    // but the inform is kept in output values so they are not kept here.
    std::vector<operation*> dependency;
    // this function is used to determine if this operation contained
    // a region. If an op contained a region, it should override
    // this function.
    //virtual graph* getSubgraph(){ return nullptr;}
    bool bActive = 1;
    bool bExplored = 0;

    // this is a member used to remove the operation efficiently. 
    // Should be used solely for removing process in graph.
    bool bRemove = 0;
    graph* graph_ = nullptr;
};

class graph : public operation{
    public : 
    class graphEntry : public operation {
        public:
        graphEntry(graph* g) : operation("", g){}
        virtual std::string represent() { return ""; }
        virtual void print() override {}
    };
    graph() = default;
    graph(std::string id, graph* pg = nullptr)
    : operation(id, pg)
    , entry(pg) {}
    virtual void print() override;
    virtual std::string represent() = 0;
    // A breadth-first walk function that is graph modification safe.
    // Call the callable at the begining of visiting each vertex.
    // The callable should return void.
    // The fn will be executed on each operation at most once. 
    // The operation ran by fn is marked as done. A operation will
    // got processed only if all inputs operations are done (All
    // dependences are processed).
    // notice that this walk skipped the entryOp so that we don't 
    // need to worry about the entry op got modified by accident.
    template<typename callable>
    void walk(callable && fn, bool checkDependency = 0, bool recycleInactive = 0, bool removeDisconnected = 0, bool deepWalk=0){
        std::queue<operation *> _vq;
        std::vector<operation *> vertice_buffer;
        vertice_buffer.reserve(getNodeSize());
        for(auto op: entry.getIndirectDependecyValue()->getUsers())
            _vq.push(op);
        _vq.push(&entry);
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
            for(auto i=0 ;i<v->getOutputSize(); i++){
                auto val = v->outputValue(i);
                for(auto vn : val->getUsers()){
                    if(checkDependency && !vn->isDependencyFullfilled()) continue;
                    if (vn->isExplored()) continue;
                    if(vn->isRemovable()) continue;
                    _vq.push(vn);
                }
            }
            if(deepWalk){
                if(auto g = dynamic_cast<graph*>(v)){
                    g->walk(fn, checkDependency, recycleInactive, removeDisconnected, 1);
                }
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
        for (auto v : vertice_buffer)
        {
            v->setExploration(false);
        }
        clean();
        return;
    }
    // add operation to this graph
    void attachToEntrance(operation* op){ 
        op->appendTo(dynamic_cast<operation*>(&entry)); }
    // regist the op in this graph into book, if this op has no
    // inputs, it will be added as an entrance op.
    void registerOp(operation* op);
    graph* getGraph() {return dynamic_cast<graph*>(this);}

    virtual void printGraph();

    void assignID(int n=0);

    // clean will remove all operations marked as removable;
    void clean();
    // entrances are the ops have no inputs
    operation&  getEntry(){ return entry; }
    virtual void verifyGraph() { 
        this->verify();
        walk([&](operation* op){
            THROW_WHEN(op->verify(), "Op verification failed!");
        }, 1);
    }

    // return how many operations graph contained
    int getNodeSize(){ return int(nodes.size()); }
    std::vector<operation*> & getNodeList(){return nodes;}
    private:
    std::vector<operation*> nodes;
    graphEntry entry;
    // how many operations contained in this graph
};

class canvas : public graph{
    public : 
    canvas() : graph("canvas"){}
    virtual std::string represent(){ return "";} 
};
//----------------------------------------

}

#endif