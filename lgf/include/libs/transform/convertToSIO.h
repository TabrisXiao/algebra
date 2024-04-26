

#ifndef TRANSFORM_CONVERT_TO_SIO_H
#define TRANSFORM_CONVERT_TO_SIO_H

#include "libs/builtin/ops.h"
#include "libs/algebra/algebra.h"
#include "libs/sio/sio.h"
#include "lgf/pass.h"

namespace lgf::transform
{

    template <typename origOp, typename targetOp>
    class convertToSIORewriter : public rewriter<origOp>
    {
    public:
        convertToSIORewriter(std::string funcName = "") : funcName(funcName){};
        virtual valueDesc *convert_desc(LGFContext *ctx, valueDesc *desc)
        {
            return desc;
        }
        virtual resultCode rewrite(painter &p, origOp *op)
        {
            auto newop = p.isomorphic_rewrite<origOp, targetOp>(op);
            newop->set_value_desc(convert_desc(p.get_context(), op->get_value_desc()));
            if (auto funcop = dynamic_cast<sio::funcOp *>(newop))
            {
                funcop->setFuncName(funcName);
            }
            return resultCode::success();
        }
        std::string funcName = "";
    };

    class convertCstToSIO : public rewriter<cstDeclOp>
    {
    public:
        convertCstToSIO() = default;
        virtual resultCode rewrite(painter &p, cstDeclOp *op)
        {
            sid_t number;
            if (auto f32d = op->get_data_attr()->dyn_cast<float32Data>())
            {
                number = f32d->represent_data();
            }
            else if (auto i32d = op->get_data_attr()->dyn_cast<int32Data>())
            {
                number = i32d->represent_data();
            }
            std::cout << op->represent() << std::endl;
            p.replace_op<lgf::sio::numberOp>(op, op->get_value_desc(), number);
            return resultCode::success();
        };
    };

    class convertToSIOPass : public passBase
    {
    public:
        convertToSIOPass() : passBase("convertToSIOPass") {}
        virtual resultCode run() final
        {
            painter p(get_graph());
            add_rewriter<convertCstToSIO>();
            add_rewriter<convertToSIORewriter<lgf::declOp, sio::symbolOp>>();
            add_rewriter<convertToSIORewriter<lgf::sumOp, sio::sumOp>>();
            add_rewriter<convertToSIORewriter<lgf::inverseOp, sio::inverseOp>>();
            add_rewriter<convertToSIORewriter<negativeOp, sio::negativeOp>>();
            add_rewriter<convertToSIORewriter<productOp, sio::scalarProductOp>>();
            add_rewriter<convertToSIORewriter<funcCosOp, sio::funcOp>>("cos");
            add_rewriter<convertToSIORewriter<funcSineOp, sio::funcOp>>("sin");
            add_rewriter<convertToSIORewriter<funcExpOp, sio::funcOp>>("exp");
            add_rewriter<convertToSIORewriter<partialDifferentiateOp, sio::partialD>>();
            add_rewriter<convertToSIORewriter<differentiateOp, sio::differentialOp>>();
            return apply_rewriter_greedy(p, get_graph());
        }
    };

    std::unique_ptr<passBase> createConvertToSIOPass()
    {
        return std::make_unique<convertToSIOPass>();
    }
}

#endif // TRANSFORM_CONVERT_TO_SIO_H