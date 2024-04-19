

#ifndef TRANSFORM_CONVERT_TO_SIO_H
#define TRANSFORM_CONVERT_TO_SIO_H

#include "libs/Builtin/ops.h"
#include "libs/algebra/ops.h"
#include "libs/SIO/ops.h"
#include "lgf/pass.h"

namespace lgf::transform
{

    template <typename origOp, typename targetOp>
    class convertToSIORewriter : public rewriter<origOp>
    {
    public:
        convertToSIORewriter(std::string funcName = "") : funcName(funcName){};
        virtual valueDesc *convert_desc(valueDesc *desc) { return desc; }
        virtual resultCode rewrite(painter &p, origOp *op)
        {
            auto newop = p.isomorphic_rewrite<origOp, targetOp>(op);
            newop->set_value_desc(convert_desc(op->get_value_desc()));
            if (auto funcop = dynamic_cast<SIO::funcOp *>(newop))
            {
                funcop->setFuncName(funcName);
            }
            return resultCode::success();
        }
        std::string funcName = "";
    };

    class convertToSIOPass : public passBase
    {
    public:
        convertToSIOPass() : passBase("convertToSIOPass") {}
        virtual resultCode run() final
        {
            painter p(get_graph());
            add_rewriter<convertToSIORewriter<lgf::declOp, SIO::symbolOp>>();
            add_rewriter<convertToSIORewriter<lgf::sumOp, SIO::sumOp>>();
            add_rewriter<convertToSIORewriter<negativeOp, SIO::negativeOp>>();
            add_rewriter<convertToSIORewriter<productOp, SIO::scalarProductOp>>();
            add_rewriter<convertToSIORewriter<funcCosOp, SIO::funcOp>>("cos");
            add_rewriter<convertToSIORewriter<funcSineOp, SIO::funcOp>>("sin");
            add_rewriter<convertToSIORewriter<partialDifferentiateOp, SIO::partialD>>();
            add_rewriter<convertToSIORewriter<differentiateOp, SIO::differentialOp>>();
            return apply_rewriter_greedy(p, get_graph());
        }
    };

    std::unique_ptr<passBase> createConvertToSIOPass()
    {
        return std::make_unique<convertToSIOPass>();
    }
}

#endif // TRANSFORM_CONVERT_TO_SIO_H