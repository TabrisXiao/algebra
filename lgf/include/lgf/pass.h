
#ifndef LGF_PASS_H_
#define LGF_PASS_H_
#include "operation.h"
#include "painter.h"
#include "utils.h"

namespace lgf{

class resultCode : public byteCode<int8_t>{
    public: 
    enum result: int8_t {
        default_result,
        success_result
    };
    resultCode(): byteCode() { value = 0; }
    resultCode(int8_t v): byteCode(int8_t(v)){}
    static resultCode success(){
        return resultCode(int8_t(resultCode::result::success_result));
    }
    static resultCode fail(){
        return resultCode();
    }

    static resultCode pass(){
        return resultCode(int8_t(resultCode::result::default_result));
    }
    
    bool isSuccess(){
        return check(success_result);
    }
};


class rewriterBase {
    public:
    rewriterBase() = default;
    virtual ~rewriterBase() = default;
    virtual resultCode execute( painter, operation * op) = 0;
    LGFContext* getContext(){ return ctx; }
    LGFContext *ctx = nullptr;
};

template<typename concreteOp>
class rewriter : public rewriterBase{
    public : rewriter() = default;
    virtual resultCode rewrite( painter, concreteOp *op) = 0;
    virtual resultCode execute( painter rewriter,operation* op) override final{
        if(auto cop = dynamic_cast<concreteOp*>(op))
        {
            auto sig = rewrite(rewriter, cop);
            return sig;
        }
        return resultCode::pass();
    }
};

class passBase {
public :
    passBase (const char * name) : _pass_name (name) {}

    // the return value is not defined yet.
    virtual resultCode run() = 0;

    graph * getGraph(){ return g; }
    void gotoGraph(graph *reg){ g = reg;}
    LGFContext * getContext(){ return ctx; }
    // addRewriter will create a rewriter using the arguments;
    template<typename T, typename ...ARGS>
    void addRewriter(ARGS...arg){ 
        auto ptr = std::make_unique<T>(arg...);
        ptr->ctx = ctx;
        rewriters.push_back(std::move(ptr));
    }

    resultCode applyRewriterOnce(painter &p, graph* g);
    resultCode applyRewriterGreedy(painter &p, graph* g);

    resultCode walkApplyRewriterOnce(painter &p, graph* g,bool deepwalk = 0);

    // Translation is a special method to apply rewriters,
    // It walk only once through a graph in the dependency order
    // and apply all the applicable rewriters to the ops.
    // So this method is only safe for the case that all rewriters
    // are order free.
    bool translation(painter &p, graph* g);

    public: 
    
    std::vector<std::unique_ptr<rewriterBase>> rewriters;
    std::string _pass_name;
    bool rewriteHappen = 0;
    LGFContext* ctx = nullptr;
    graph * g=nullptr;
};


class passManager{
    public : 
    passManager() = default;
    passManager(LGFContext* c, graph *op) {ctx = c, start = op;}
    void enablePrintAfterPass(){bPrintAfterPass = 1;}
    void enablePrintBeforePass(){bPrintBeforePass = 1;}

    void validation(graph* g);
    
    void redundantCheck(operation* op){
        bool canRemove = 1;
        for(auto & val: op->getOutputs() ){
            if(val->getUserSize()!=0) {
                canRemove = 0;
                break;
            }
        }
        if(canRemove){
            op->erase();
        }
    }
    
    void run(){
        if(bPrintInitialIR) 
        {   OSTREAM<<"\n------ Input "<<name<<" ------\n";
            start->assignID(0);
            start->print();
        }
        
        for(auto pass=passes.begin(); pass!=passes.end(); pass++){
            if(bPrintBeforePass){
                OSTREAM<<"------ IR before pass:  "<<(*pass).get()->_pass_name<<" ------\n";
                start->assignID(0);
                start->print();
                OSTREAM<<"\n";
            }

            (*pass)->run();
            //std::cout<<"pass: "<<(*pass)->_pass_name<<" finished."<<std::endl;
            validation(start);
            if(bPrintAfterPass){
                OSTREAM<<"------ IR after pass: "<<(*pass).get()->_pass_name<<" ------\n";
                start->assignID(0);
                start->print();
                OSTREAM<<"\n";
            }
        }
        if(bPrintFinalIR) 
        {   OSTREAM<<"\n------ IR after "<<name<<" ------\n";
            start->assignID(0);
            start->print();
        }
    }

    void addPass(std::unique_ptr<passBase> ps){
        ps->gotoGraph(dynamic_cast<graph*>(start));
        ps->ctx = ctx;
        passes.push_back(std::move(ps));
    }
    void addNormalizationPass();
    std::vector<std::unique_ptr<passBase>> passes;

    bool bPrintAfterPass = 0;
    bool bPrintBeforePass = 0;
    bool bPrintInitialIR = 0;
    bool bPrintFinalIR = 0;
    graph * start = nullptr;
    LGFContext* ctx = nullptr;
    std::string name = "";
};

}

#endif