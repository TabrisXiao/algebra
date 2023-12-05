
#ifndef MATH_AAB_OPS_H
#define MATH_AAB_OPS_H
#include "lgf/operation.h"
#include "lgf/group.h"
#include "types.h"

namespace lgf::LinearAlg{

// class unitMatrixDeclare : public lgf::operation, public normalizer{
//     public:
//     unitMatrixDeclare() = default;
//     static unitMatrixDeclare* build(lgf::LGFContext* ctx, int n){
//         auto op = new unitMatrixDeclare();
//         op->createValue(ctx->getType<tensor_t>(ctx->getType<float_t>(), {n,n}), "");
//         return op;
//     }

//     virtual resultCode rewrite(painter p, operation *op)
// };

class contractOp : public lgf::operation{
    public:
    contractOp() = default;
    static contractOp* build(lgf::LGFContext* ctx, lgf::value* lhs, int lhsN, lgf::value* rhs, int rhsN){
        auto op = new contractOp();
        op->registerInput(lhs, rhs);
        op->lhsAxis = lhsN;
        op->rhsAxis = rhsN;
        op->createValue(op->inferType(ctx,lhs,rhs),"");
        return op;
    }
    type_t inferType(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value *rhs){
        lhs->type_check<tensor_t>();
        rhs->type_check<tensor_t>();
        auto& lhsType = lhs->getType<tensor_t>();
        auto& rhsType = lhs->getType<tensor_t>();
        if(lhsType.isAbstract()) return lhs->getType();
        if(rhsType.isAbstract()) return rhs->getType();
        auto &lhShape = lhsType.shape();
        auto &rhShape = rhsType.shape();
        if(lhShape.size()> lhsAxis || rhShape.size()> rhsAxis){
            THROW("contract axis excceeded the tensor shape size!");
        }
        CHECK_VALUE(lhShape[lhsAxis], rhShape[rhsAxis], "The dimension along the contract dimension has to be the same!");
        //combining two shapes into the output shape.
        std::vector<int> newShape(rhShape.size()+lhShape.size()-1);
        int n=0;
        for(auto iter = lhShape.begin(); iter !=lhShape.end(); iter++){
            if(iter == lhShape.begin()+lhsAxis) continue;
            newShape[n++]=*iter;
        }
        for(auto iter = rhShape.begin(); iter !=rhShape.end(); iter++){
            if(iter == rhShape.begin()+rhsAxis) continue;
            newShape[n++]=*iter;
        }
        return ctx->getType<tensor_t>(lhsType.getElemType(), newShape);
    }
    int lhsAxis, rhsAxis;
};

class determinant : public lgf::operation {
    public:
    determinant () = default;
    static determinant* build(lgf::LGFContext*, lgf::value *input){
        input->type_guard<tensor_t>();
        auto op = new determinant();
        op->setSID("linearAlg::determinant");
        op->createValue(input->getType<tensor_t>().getElemType(), "");
        return op;
    }
};

}

#endif