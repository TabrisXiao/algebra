
#ifndef MATH_AAB_OPS_H
#define MATH_AAB_OPS_H
#include "lgf/operation.h"
#include "libs/builtin/types.h"
#include "libs/AAB/types.h"
#include "lgf/group.h"
#include "pattern.h"

namespace lgf::AAB{

// ---------- mappingOp ----------
class mappingOp: public lgf::operation{
    public:
    mappingOp(std::string mapName) : operation(mapName){}
    template <typename... ARGS>
    mappingOp(std::string mapName, type_t tp, ARGS... args) : operation(mapName){
        createValue(tp, "");
        addArgument(args...);
    }
    template <typename... ARGS>
    void addArgument(ARGS ...args){
        auto values = {args...};
        for(auto & val : values){
            registerInput(val);
            narg++;
        }
    }
    void addArguments(std::vector<value*>& values){
        for(auto & val : values){
            addArgument(val);
        }
    }

    std::vector<value*> getArugments(){
        std::vector<value*> ret(narg);
        for(size_t i=0; i<narg; i++){
            ret[i] = argument(i);
        }
        return ret;
    }
    // template <typename... ARGS>
    // static mappingOp* build(lgf::LGFContext* ctx, std::string mapName, type_t tp, ARGS ...args){
    //     auto op = new mappingOp(mapName, args...);
    //     return op;
    // }
    size_t narg=0;
    size_t getArgNumber(){ return narg; }
    lgf::value* argument(size_t n){ return inputValue(n); }
};

// ---------- abstractMappingOp ----------
class abstractMappingOp : public mappingOp{
    public:
    abstractMappingOp(): mappingOp("AAB::AbstractMapping"){}
    template <typename... ARGS>
    abstractMappingOp(ARGS ...args): mappingOp("AAB::AbstractMapping", args...){}
    template <typename... ARGS>
    static abstractMappingOp* build(lgf::LGFContext* ctx, type_t tp, ARGS ...args){
        auto op = new abstractMappingOp(args...);
        op->createValue(tp, "");
        return op;
    }
    static abstractMappingOp* build(lgf::LGFContext* ctx, type_t tp){
        auto op = new abstractMappingOp();
        op->createValue(tp, "");
        return op;
    }
};

// ---------- addOp ----------
class addOp : public mappingOp, public normalizer
{
    public:
    addOp() : mappingOp("AAB::add") {}
    static addOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new addOp();
        op->addArgument(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0)->represent()<<" + "<<inputValue(1)->represent();
        return p.dump();
    }
    virtual resultCode rewrite(painter p, operation* op);
};

// ---------- negativeOp ----------
class negativeOp : public mappingOp, public normalizer
{
    public:
    public:
    negativeOp() : mappingOp("AAB::negative") {}
    static negativeOp* build(lgf::LGFContext* ctx, lgf::value* input){
        auto op = new negativeOp();
        op->addArgument(input);
        op->createValue(input->getType(), "");
        return op;
    }
    lgf::value* input(){ return inputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<input()->represent();
        return p.dump();
    }
    virtual resultCode rewrite(painter p, operation* op){
        if(auto neg = input()->getDefiningOp<negativeOp>()){
            p.setPaintPointAfter(op);
            op->replaceBy(neg->input()->getDefiningOp());
            return resultCode::success();
        }
        return resultCode::pass();
    }
};

// ---------- sumOp ----------
class sumOp : public mappingOp, public normalizer {
    public:
    sumOp() : mappingOp("AAB::sum") {}
    static sumOp* build(lgf::LGFContext* ctx, std::vector<value*>& vec){
        auto op = new sumOp();
        op->addArguments(vec);
        op->createValue(vec[0]->getType(), "");
        return op;
    }
    static sumOp* build(lgf::LGFContext* ctx, type_t type){
        auto op = new sumOp();
        op->createValue(type, "");
        return op;
    }
    template<typename ...ARGS>
    static sumOp* build(lgf::LGFContext* ctx, ARGS ... args ){
        auto op = new sumOp();
        op->addArgument(args...);
        op->createValue(op->inputValue(0)->getType(), "");
        return op;
    }
    lgf::value* input(int i=0){ return inputValue(i); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<representInputs();
        return p.dump();
    }

    virtual resultCode rewrite(painter p, operation* op);
};


// ---------- minusOp ----------
class minusOp : public mappingOp, public normalizer
{
    public:
    minusOp() : mappingOp("AAB::minus") {}
    static minusOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new minusOp();
        op->addArgument(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<lhs()->represent()<<" - "<<rhs()->represent();
        return p.dump();
    }
    virtual resultCode rewrite(painter p, operation* op){
        resultCode ret;
        p.setPaintPointAfter(op);
        auto neg = p.paint<negativeOp>(op->inputValue(1));
        auto add = p.replaceOp<addOp>(op, op->inputValue(0), neg->output());
        return resultCode::success();
    }
};

// ---------- multiplyOp ----------
class multiplyOp : public mappingOp, public normalizer
{
    public:
    multiplyOp() : mappingOp("AAB::multiply") {}
    static multiplyOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new multiplyOp();
        op->addArgument(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }

    virtual std::string represent(){
        printer p;
        //std::cout<<outputValue(1)<<std::endl;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0)->represent()<<" * "<<inputValue(1)->represent();
        return p.dump();
    }

    virtual resultCode rewrite(painter p, operation* op);
};

class directProductOp: public mappingOp {
    public:
    directProductOp(): mappingOp("AAB::directProduct"){};
    template<typename ...ARGS>
    static directProductOp* build(lgf::LGFContext* ctx, ARGS ... args ){
        auto op = new directProductOp();
        op->addArgument(args...);
        op->createValue();
        op->inferType(ctx);
        return op;
    }
    virtual void inferType(LGFContext* ctx) override {
        std::vector<type_t> types;
        for(auto &input : getInputs()){
            auto& type = input->getType();
            if(type.getSID() == sequenceType::sid){ 
                auto data = type.getDesc<sequenceDesc>()->getTypes();
                types.insert(types.end(), data.begin(), data.end());
            }else {
                types.push_back(type);
            }
        }
        auto seq = ctx->getType<sequenceType>(types);
        output()->setType(seq);
    }
};

// ---------- productOp ----------
class productOp : public mappingOp, public normalizer
{
    public:
    productOp() : mappingOp("AAB::product") {}
    static productOp* build(lgf::LGFContext* ctx, std::vector<value*>& vec){
        auto op = new productOp();
        op->addArguments(vec);
        op->createValue(vec[0]->getType(), "");
        return op;
    }
    static productOp* build(lgf::LGFContext* ctx, type_t type){
        auto op = new productOp();
        op->createValue(type, "");
        return op;
    }
    template<typename ...ARGS>
    static productOp* build(lgf::LGFContext* ctx, ARGS ... args ){
        auto op = new productOp();
        op->addArgument(args...);
        op->createValue(op->inputValue(0)->getType(), "");
        return op;
    }
    lgf::value* input(int i=0){ return inputValue(i); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<representInputs();
        return p.dump();
    }
    virtual void inferType(LGFContext* ctx) override {
        output()->setType(input(0)->getType());
    }

    virtual resultCode rewrite(painter p, operation* op);
};

class commutableProductOp: public mappingOp, public normalizer{
    public:
    commutableProductOp() : mappingOp("AAB::commutableProduct") {}
    static commutableProductOp* build(lgf::LGFContext* ctx, std::vector<value*>& vec){
        auto op = new commutableProductOp();
        op->addArguments(vec);
        op->createValue(vec[0]->getType(), "");
        return op;
    }
    static commutableProductOp* build(lgf::LGFContext* ctx, type_t type){
        auto op = new commutableProductOp();
        op->createValue(type, "");
        return op;
    }
    template<typename ...ARGS>
    static commutableProductOp* build(lgf::LGFContext* ctx, ARGS ... args ){
        auto op = new commutableProductOp();
        op->addArgument(args...);
        op->createValue(op->inputValue(0)->getType(), "");
        return op;
    }
    lgf::value* input(int i=0){ return inputValue(i); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<representInputs();
        return p.dump();
    }
    virtual void inferType(LGFContext* ctx) override {
        output()->setType(input(0)->getType());
    }

    virtual resultCode rewrite(painter p, operation* op);
};

class directProduct: public mappingOp {
    public:
    directProduct(): mappingOp("AAB::directProduct"){}
    template<typename ...ARGS>
    static productOp* build(lgf::LGFContext* ctx, ARGS ... args ){
        auto op = new productOp();
        op->addArgument(args...);
        op->createValue(op->inputValue(0)->getType(), "");
        return op;
    }
};

class tensorProductOp : public mappingOp{
    public:
    tensorProductOp() : mappingOp("AAB::tensorProduct"){}
    static tensorProductOp* build(lgf::LGFContext* ctx, value* lhs, value* rhs){
        auto op = new tensorProductOp();
        op->addArgument(lhs, rhs);
        op->createValue();
        op->inferType(ctx);
        return op;
    }
    virtual void inferType(LGFContext* ctx) override {
        tensorType lhs_tp = inputValue(0)->getType<tensorType>();
        tensorType rhs_tp = inputValue(1)->getType<tensorType>();
        if(lhs_tp.getElemType() != lhs_tp.getElemType()){
            throw std::runtime_error("tensorType: type mismatch");
        }
        std::vector<dim_t> dims(lhs_tp.getDims());
        std::vector<dim_t> rhs_dims(rhs_tp.getDims());
        dims.insert(dims.end(), rhs_dims.begin(), rhs_dims.end());
        auto tp = ctx->getType<tensorType>(lhs_tp.getElemType(), dims);
        output()->setType(tp);
     }
};

class contractionOp : public mappingOp {
     public:
     contractionOp() : mappingOp("AAB::contraction"){}
     static contractionOp* build(lgf::LGFContext* ctx, value* lhs, value* rhs, int index_lhs, int index_rhs){
        // in this case, it is like a tensor product + contraction
        auto op = new contractionOp();
        op->addArgument(lhs, rhs);
        op->idx_lhs = index_lhs;
        op->idx_rhs = index_rhs;
        op->createValue();
        op->inferType(ctx);
        return op;
     }
     static contractionOp* build(lgf::LGFContext* ctx, value* input, int index_lhs, int index_rhs){
        auto op = new contractionOp();
        op->addArgument(input);
        op->idx_lhs = index_lhs;
        op->idx_rhs = index_rhs;
        op->createValue();
        op->inferType(ctx);
     }
     void inferType1(LGFContext* ctx) {
        auto type = inputValue(0)->getType<tensorType>();
        std::vector<dim_t> dims(type.getDims());
        if( dims[idx_lhs] != dims[idx_rhs]){
            THROW("contractionOp: dimension mismatch");
        }
        dims.erase(dims.begin()+idx_lhs);
        dims.erase(dims.begin()+idx_rhs);
        auto tp = ctx->getType<tensorType>(type.getElemType(), dims);
        output()->setType(tp);
     }
     void inferType2(LGFContext* ctx) {
        // infer the type when two inputs are provided
        tensorType lhs_tp = inputValue(0)->getType<tensorType>();
        tensorType rhs_tp = inputValue(1)->getType<tensorType>();
        if(lhs_tp.getElemType() != lhs_tp.getElemType()){
            THROW("contractionOp: element type mismatch");
        }
        std::vector<dim_t> dims(lhs_tp.getDims());
        if( dims[idx_lhs] != rhs_tp.getDims()[idx_rhs]){
            THROW("contractionOp: dimension mismatch");
        }
        dims.erase(dims.begin()+idx_lhs);
        std::vector<dim_t> rhs_dims(rhs_tp.getDims());
        rhs_dims.erase(rhs_dims.begin()+idx_rhs);
        dims.insert(dims.end(), rhs_dims.begin(), rhs_dims.end());
        auto tp = ctx->getType<tensorType>(lhs_tp.getElemType(), dims);
        output()->setType(tp);
     }
     virtual void inferType(LGFContext* ctx) override {
        if(getArgNumber() == 1){
            inferType1(ctx);
        }
        else 
        {
            inferType2(ctx);
        }
        return;
     }
     int idx_lhs, idx_rhs;
 };

// ---------- inverseOp ----------
class inverseOp : public mappingOp, public normalizer
{
    public:
    inverseOp() : mappingOp("AAB::inverse") {}
    static inverseOp* build(lgf::LGFContext* ctx, lgf::value* input){
        auto op = new inverseOp();
        op->addArgument(input);
        op->createValue(input->getType(), "");
        std::cout<<"-- inverseOp::build"<<input->represent()<<" : "<<op->output()->represent()<<std::endl;
        return op;
    }
    lgf::value* input(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(1); }
    virtual resultCode rewrite(painter p, operation* op){
        // reduce the inverse(invers(x)) to x
        if(auto inv = input()->getDefiningOp<inverseOp>()){
            p.setPaintPointAfter(op);
            op->replaceBy(inv->input()->getDefiningOp());
            return resultCode::success();
        }
        return resultCode::pass();
    }
    virtual void inferType(LGFContext* ctx) override {
        output()->setType(input()->getType());
    }
};

// ---------- quotientOp ----------
class quotientOp : public mappingOp
{
    public:
    quotientOp() : mappingOp("AAB::quotient"){}
    static quotientOp* build(lgf::LGFContext *ctx, lgf::value* x, lgf::value* y){
        auto op = new quotientOp();
        op->addArgument(x, y);
        op->createValue(x->getType(), "");
        return op;
    }
    lgf::value* numerator(){ return inputValue(0); }
    lgf::value* denominator(){ return inputValue(1); }
};

class powerOp : public mappingOp
{
    public:
    powerOp() : mappingOp("AAB::power"){}
    static powerOp* build(lgf::LGFContext* ctx, lgf::value* x, lgf::value *y){
        auto op = new powerOp();
        op->addArgument(x, y);
        op->createValue(x->getType(), "");
        return op;
    }
    lgf::value* power(){ return inputValue(1); }
    lgf::value* x(){ return inputValue(0); }

};


class funcSineOp : public mappingOp{
    public:
    funcSineOp() :  mappingOp("AAB::sine"){}
    static funcSineOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcSineOp();
        op->addArgument(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class funcCosOp : public mappingOp{
    public:
    funcCosOp(): mappingOp("AAB::cos"){}
    static funcCosOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcCosOp();
        op->addArgument(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class permuteOp : public lgf::operation {
    public:
    permuteOp() : operation("AAB::permuteOp"){}
    static permuteOp* build(lgf::LGFContext* ctx, type_t type, value* input, value* from_, value* to_){
        auto op = new permuteOp();
        op->registerInput(input, from_, to_);
        op->createValue(type, "");
        return op;
    }
    value* input() {return inputValue(0);}
    value* from() {return inputValue(1);}
    value* to() { return inputValue(2); }
};

class distributeOp : public operation, public normalizer{
    public:
    distributeOp(): operation("AAB::distribute") {}
    static distributeOp* build(LGFContext* ctx, value *input){
        auto op = new distributeOp();
        op->registerInput(input);
        op->createValue(input->getType(), "");
        //op->setNontrivial();
        return op;
    }
    value *input(){ return inputValue(0); }
    
    resultCode distribute(painter p, value* val, operation* user){
        auto op = val->getDefiningOp();
        resultCode result;
        operation* newop=nullptr;
        if(auto root = dynamic_cast<multiplyOp*>(op)){
            auto lhs = root->inputValue(0);
            auto rhs = root->inputValue(1);
            if(logicResult::success() == rDistributivePattern<multiplyOp,  addOp>(op)){
                auto addop = rhs->getDefiningOp<addOp>();
                p.setPaintPointAfter(root);
                auto addl = p.paint<multiplyOp>(lhs, addop->lhs());
                auto addr = p.paint<multiplyOp>(lhs, addop->rhs());
                addop->output()->disconnectOp(root);
                newop = p.paint<addOp>(addl->output(), addr->output());
                user->replaceInputValueBy(val, newop->outputValue(1));
                result.add(resultCode::success());
                p.getGraph()->clean();
            }else if(
                logicResult::success() == lDistributivePattern<multiplyOp,  addOp>(op)
            ){
                auto addop = lhs->getDefiningOp<addOp>();
                p.setPaintPointAfter(root);
                auto addl = p.paint<multiplyOp>(addop->lhs(), rhs);
                auto addr = p.paint<multiplyOp>(addop->rhs(), rhs);
                addop->output()->disconnectOp(root);
                newop = p.paint<addOp>(addl->output(), addr->output());
                user->replaceInputValueBy(val, newop->outputValue(1));
                result.add(resultCode::success());
                p.getGraph()->clean();
            } else {
                newop = root;
            }
            if(result.isSuccess()){
                root->replaceBy(newop);
                root->erase();
            }
        }else {
            newop = dynamic_cast<addOp*>(op);
        }
        if(newop){
            result.add(distribute(p, newop->inputValue(0), newop));
            result.add(distribute(p, newop->inputValue(1), newop));
        }
        return result;
    }

    resultCode matchCheck(painter p, operation* );
    void transform(painter p, productOp*, std::vector<value*>::iterator&, sumOp*);
    virtual resultCode rewrite(painter p, operation* op);
};

class associateOp : public operation, public normalizer {
    public:
    associateOp() : operation("AAB::associate"){}
    static associateOp* build(LGFContext* ctx, value* input, value* target){
        auto op = new associateOp();
        op->registerInput(input);
        op->createValue(input->getType(), "");
        return op;
    }

    resultCode transformIfEqual(painter p, operation* op, value* target, int hand){
        int other = hand == 1 ? 0 : 1;
        if(op->outputValue(1) == target){
            p.setPaintPointAfter(op);
            p.replaceOp<cstDeclOp>(op, 1);
            op->erase();
            return resultCode::success();
        } else if( auto mulop = dynamic_cast<multiplyOp*>(op)){
            if(mulop->inputValue(hand)==target){
                
            }
        }
        return resultCode::fail();
    }

    resultCode associate(painter p, operation* op, int l = 0, int r = 1){
        resultCode result;
        if(auto addop = dynamic_cast<addOp*>(op)){
            value* val1 = addop->inputValue(l);
            value* val2 = addop->inputValue(r);
            if(auto multi = val1->getDefiningOp<multiplyOp>()){
                val1 = multi->inputValue(l);
                
            }
        }
    }
    virtual resultCode rewrite(painter p, operation* op){
        return resultCode::pass();
    }
};

class partialDifferentiateOp : public mappingOp {
    public:
    partialDifferentiateOp() : mappingOp("AAB::PartialDifferentiate"){}
    static partialDifferentiateOp* build(LGFContext* ctx, value* func, value* var){
        auto op = new partialDifferentiateOp();
        op->addArgument(func, var);
        op->createValue(func->getType(), "");
        return op;
    }
    value* func(){ return inputValue(0); }
    value* var(){ return inputValue(1); }
};

class differentiateOp : public mappingOp {
    public:
    differentiateOp() : mappingOp("AAB::differentiate"){}
    static differentiateOp* build(LGFContext* ctx, value* input, value* target){
        auto op = new differentiateOp();
        op->addArgument(input, target);
        op->createValue(input->getType(), "");
        return op;
    }
    value* input(){ return inputValue(0); }
    value* target(){ return inputValue(1); }
};

class exponentialOp : public mappingOp {
    public:
    exponentialOp() : mappingOp("AAB::exponential"){}
    static exponentialOp* build(LGFContext* ctx, value* input, value* power){
        auto op = new exponentialOp();
        op->addArgument(input, power);
        op->createValue(input->getType(), "");
        return op;
    }
    value* input(){
        return inputValue(0);
    }
    void inferType(LGFContext* ctx) override{
        // using the input type as output type
        output()->setType(input()->getType()); 
    }
};


// class factorOp : public operation, public normalizer{
//     public:
//     factorOp() : operation("AAB::factor") {}
//     static factorOp* build(lgf::LGFContext* ctx, lgf::value* exp, lgf::value* target){
//         auto op = new factorOp();
//         op->registerInput(exp, target);
//         op->createValue(exp->getType(), "");
//         return op;
//     }
//     lgf::value* exp() { return inputValue(0); }
//     lgf::value* target() { return inputValue(1); }
//     lgf::value* output(){ return outputValue(1); }

//     struct factorInfo {
//         int count = 0;
//         lgf::value* ret=nullptr;
//     };

//     virtual std::string represent(){
//         printer p;
//         p<<representOutputs()<<" = "<<getSID() <<" : "<<target()->represent()<<" out of "<<exp()->represent();
//         return p.dump();
//     }

//     factorInfo factorImpl(painter& p, value* nodeValue, value* target){
//         factorInfo info;
//         // check if current value is the target
//         if(target == nodeValue ){
//             return info;
//         }
        
//         if(auto multiop = nodeValue->getDefiningOp<multiplyOp>()){
//             auto linfo = factorImpl(p, multiop->lhs(), target);
//             auto rinfo = factorImpl(p, multiop->rhs(), target);
//             bool removeOp = 0;
//             info.count = rinfo.count+linfo.count;
//             // check if lhs is the target
//             if(!linfo.ret ){
//                 info.count++;
//                 info.ret = rinfo.ret;
//                 removeOp = 1;
//             }
//             // check if rhs is the target
//             if(!rinfo.ret){
//                 info.count++;
//                 info.ret = linfo.ret;
//                 removeOp = 1;
//             }
            
//             if( info.ret ){
//                 multiop->replaceBy(info.ret->getDefiningOp());
//             } else if(removeOp) {
//                 multiop->erase();
//             } else {
//                 info.ret = multiop->output();
//             }
//             return info;
//         }else if(auto addop = nodeValue->getDefiningOp<addOp>()){
//             auto linfo = factorImpl(p, addop->lhs(), target);
//             auto rinfo = factorImpl(p, addop->rhs(), target);
//             info.count = linfo.count;
//             bool rebuild = 0;
//             value* nlhs, *nrhs;
//             if( info.count > rinfo.count) info.count = rinfo.count;
//             if(linfo.count > info.count) {
//                 rebuild = 1;
//             }
//         }
//         info.ret = nodeValue;
//         return info;
//     }
//     virtual resultCode rewrite(painter p, operation* op){
//         auto fop = dynamic_cast<factorOp*>(op);
//         auto exp = fop->exp();                                                                                                  
//         auto target = fop->target();
//         auto info = factorImpl(p, exp, target);
//         p.setPaintPointAt(op);
//         if(info.count==1){
//             p.replaceOp<multiplyOp>(fop, target, info.ret);
//         } else if(info.count>1){
//             auto cst = p.paint<cstDeclOp>(info.count);
//             auto power = p.paint<powerOp>(target, cst->output());
//             p.replaceOp<multiplyOp>(fop, power->output(), info.ret);
//         } else {
//             fop->replaceBy(exp->getDefiningOp());
//         }
//         return 0;
//     }
// };

}
#endif