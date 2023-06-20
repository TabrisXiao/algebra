
#ifndef PASS_H_
#define PASS_H_
#include "operation.h"
#include "painter.h"
#include "lgf.h"

namespace lgf{

class passBase {
public :
    passBase (const char * name) : _pass_name (name) {}

    // the return value is not defined yet.
    virtual bool run() = 0;

    graph * getGraph(){ return g; }
    void gotoGraph(graph *reg){ g = reg;}
    std::string _pass_name;
    bool rewriteHappen = 0;
    graph * g=nullptr;
};

class passManager{
    public : 
    passManager(graph *op) {start = op;}
    void enablePrintAfterPass(){bPrintAfterPass = 1;}
    bool runPasses(){
        return 0;
    }
    void run(){
        for(auto pass=passes.begin(); pass!=passes.end(); pass++){
            (*pass).get()->run();
            if(bPrintAfterPass){
                OSTREAM<<"------ representation after pass: "<<(*pass).get()->_pass_name<<" ------\n";
                start->assignID(0);
                start->print();
                OSTREAM<<"\n";
            }
        }
    }

    void addPass(std::unique_ptr<passBase> ps){
        ps->gotoGraph(dynamic_cast<graph*>(start));
        passes.push_back(std::move(ps));
    }
    std::vector<std::unique_ptr<passBase>> passes;

    bool bPrintAfterPass = 0;
    bool bPrintBeforePass = 0;
    graph * start = nullptr;
};

}

#endif