
#ifndef LGF_PASS_H_
#define LGF_PASS_H_
#include "operation.h"
#include "painter.h"

namespace lgf{
class passBase {
public :
    passBase (const char * name) : _pass_name (name) {}

    // the return value is not defined yet.
    virtual bool run() = 0;

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

    bool applyRewriterOnce(painter &p, graph* g);
    int applyRewriterGreedy(painter &p, graph* g);

    bool walkApplyRewriterOnce(painter &p, graph* g,bool deepwalk = 0);

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
    void run(){
        if(bPrintInitialIR) 
        {   OSTREAM<<"\n------ Initial "<<name<<" ------\n";
            start->assignID(0);
            start->print();
        }
        if(bPrintAfterPass){
                OSTREAM<<"------ Init IR ------\n";
                start->assignID(0);
                start->print();
                OSTREAM<<"\n";
            }
        for(auto pass=passes.begin(); pass!=passes.end(); pass++){
            (*pass)->run();
            start->clean();
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