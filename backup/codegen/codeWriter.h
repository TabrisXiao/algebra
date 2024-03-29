#ifndef CODEGEN_CODEWRITER_H
#define CODEGEN_CODEWRITER_H

#include "stream.h"
#include "lgf/operation.h"

namespace lgf{
namespace codegen{
class translationRuleBase {
    public:
    translationRuleBase() = default;
    virtual bool translate(cgstream &, lgf::operation *) = 0;
    virtual bool filter(lgf::operation*) = 0;
    virtual void enterGraphRule(cgstream &out, lgf::graph* reg){
        out.incrIndentLevel();
    }
    virtual void exitGraphRule(cgstream &out, lgf::graph* reg){
        out.decrIndentLevel();
    }

    virtual void writeHeader(cgstream &out, lgf::graph* reg){
    } 
    virtual void writeFinal(cgstream &out, lgf::graph* reg){
    } 
};

template<typename opty>
class translateRule : public translationRuleBase {
    public : 
    translateRule() = default;
    virtual bool filter(lgf::operation* op) {
        if(auto ty = dynamic_cast<opty*>(op)) return 1;
        return 0;
    }
    virtual bool translate(cgstream &out, lgf::operation *op){
        auto ptr = dynamic_cast<opty*>(op);
        return write(out, ptr);
    }

    virtual bool write(cgstream &out, opty *op) = 0;
};

class codeWriter {
    public: 
    codeWriter () = default;
    
    bool translateOp(cgstream &out, lgf::operation* op){
        bool check = 0;
        for(auto &w : rules){
            auto ptr = w.get();
            auto isop = ptr->filter(op);
            if(isop){
                auto res = ptr->translate(out, op);
            }
            check= isop || check;
        }
        THROW_WHEN(check==0, "Unkown write rule for Op: "+op->getSID()+".");
        if(auto g = op->expandToGraph()){
            translateGraph(out, g);
        }
    }

    void translateGraph(cgstream&out, lgf::graph *reg){
        auto rule = rules[0].get();
        rule->enterGraphRule(out, reg);
        reg->walk([&](lgf::operation* op){
            translateOp(out, op);
        });
       rule->exitGraphRule(out, reg);
    }

    void start(lgf::graph* reg){
        for(auto &w : rules){
            auto ptr = w.get();
            ptr->writeHeader(out, reg);
        }
    }

    void final(lgf::graph* reg){
        for(auto &w : rules){
            auto ptr = w.get();
            ptr->writeFinal(out, reg);
        }
    }

    void write(lgf::graph* reg){
        start(reg);
        reg->walk([&](lgf::operation* op){
            translateOp(out, op);
        });
        final(reg);
    }

    template<typename T, typename ...ARGS>
    void addTranslationRule(ARGS...arg){ 
        auto ptr = std::make_unique<T>(arg...);
        rules.push_back(std::move(ptr));
    }

    cgstream out;
    std::vector<std::unique_ptr<translationRuleBase>> rules;
};

} //namespace codegen
} //namespace lgf
    


#endif