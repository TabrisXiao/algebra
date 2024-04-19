

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
        virtual type_t convertType(type_t type) { return type; }
        virtual resultCode rewrite(painter p, origOp *op)
        {
            auto newop = p.isomorphicRewrite<origOp, targetOp>(op);
            newop->outputValue(0)->setType(convertType(op->outputValue(0)->getType()));
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
            painter p(getContext());
            addRewriter<convertToSIORewriter<declOp, SIO::symbolOp>>();
            addRewriter<convertToSIORewriter<AAB::sumOp, SIO::sumOp>>();
            addRewriter<convertToSIORewriter<AAB::negativeOp, SIO::negativeOp>>();
            addRewriter<convertToSIORewriter<AAB::productOp, SIO::scalarProductOp>>();
            addRewriter<convertToSIORewriter<AAB::commutableProductOp, SIO::scalarProductOp>>();
            addRewriter<convertToSIORewriter<funcCosOp, SIO::funcOp>>("cos");
            addRewriter<convertToSIORewriter<funcSineOp, SIO::funcOp>>("sin");
            addRewriter<convertToSIORewriter<partialDifferentiateOp, SIO::partialD>>();
            return applyRewriterGreedy(p, getGraph());
        }
    };

    std::unique_ptr<passBase> createConvertToSIOPass()
    {
        return std::make_unique<convertToSIOPass>();
    }
}

#endif // TRANSFORM_CONVERT_TO_SIO_H