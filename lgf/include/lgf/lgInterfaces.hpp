
#ifndef BUILTIN_INTERFACES_H
#define BUILTIN_INTERFACES_H
#include <unordered_map>
#include "painter.h"
#include "group.h"
#include "pass.h"

namespace lgf{
// for operation that inputs are commutable 
class commutable : public group<commutable>, public rewriter<group<commutable>> {
    public:
    commutable () = default;
    virtual bool rewrite(painter &rewriter, group<commutable> *origOp) override{
        auto op = dynamic_cast<operation*>(origOp);
        auto vec = op->getInEdges();
        std::sort(vec.begin(), vec.end(), [](value* a, value* b) {return a < b; });
        return 0;
    }
};

template<typename opType>
class DuplicatesRemover : public rewriter<opType>{
    public:
    DuplicatesRemover() = default;
    virtual bool rewrite(painter &rewriter, opType *origop) override{
        auto ctx = rewriter.getContext();
        auto origOp = dynamic_cast<operation*>(origop);
        auto code = origOp->represent(ctx);
        code = getExpressionCode(code);
        if(book.find(origOp)!=book.end()){
            return 0;
        }
        for(auto & pair : book){
            if(isEqual(code, pair.second)){
                rewriter.replaceOp(origOp, pair.first);
                return 1;
            }
        }
        book[origOp] = code;
    }
    std::string getExpressionCode(std::string &str){
        parser ps;
        ps.handle(str);
        std::string word = "1";
        while(word !="="){
            word = ps.nextWord();
        }
        return ps.getRestBuffer();
    }
    bool isEqual(std::string& code1, std::string &code2){
        if(code1 == code2) return 1;
        return 0;
    }
    std::unordered_map<operation*, std::string> book;
};

// // this type of pass is used specific for trait normalization if it has
// class normalizationPass : public passBase {
//     public:
//     normalizationPass() : passBase("normalization pass"){}
//     bool run() final{
//         graphModifier gm;
//         gm.addRewriter<commutable>();
//         gm.addRewriter<DuplicatesRemover<group<commutable>>>();
//         auto reg = getRegion();
//         gm.walkApplyOnce(reg);
//         return 0;
//     }
// };

// void createNormalizationPass(passManager &pm){
//     pm.addPass(std::make_unique<normalizationPass>());
// }
}
#endif