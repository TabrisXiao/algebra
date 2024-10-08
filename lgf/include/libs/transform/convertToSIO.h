

#ifndef TRANSFORM_CONVERT_TO_SIO_H
#define TRANSFORM_CONVERT_TO_SIO_H

#include "libs/builtin/ops.h"
#include "libs/math/algebra/algebra.h"
#include "libs/sio/sio.h"
#include "lgf/pass.h"

namespace lgf::transform
{

    template <typename origOp, typename targetOp>
    class convertToSIORewriter : public rewriter<origOp>
    {
    public:
        convertToSIORewriter(std::string funcName = "") : funcName(funcName){};
        virtual descriptor convert_desc(LGFContext *ctx, descriptor desc)
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
            auto desc = op->get_value_desc();
            // if (desc->dyn_cast<unitDesc>())
            // {
            //     auto elem = desc->dyn_cast<unitDesc>()->get_elem_desc();
            //     if (elem->is<math::realNumber, varDesc>())
            //     {
            //         p.replace_op<lgf::sio::numberOp>(op, op->get_value_desc(), "1");
            //         return resultCode::success();
            //     }
            // }
            // else if (desc->dyn_cast<zeroDesc>())
            // {
            //     auto elem = desc->dyn_cast<zeroDesc>()->get_elem_desc();
            //     if (elem->is<math::realNumber, varDesc>())
            //     {
            //         p.replace_op<lgf::sio::numberOp>(op, op->get_value_desc(), "0");
            //         return resultCode::success();
            //     }
            // }
            // else
            if (desc.is<math::realNumber>())
            {
                auto real_data = op->get_data_attr();
                p.replace_op<lgf::sio::numberOp>(op, op->get_value_desc(), real_data.represent());
                return resultCode::success();
            }
            else if (!op->get_data_attr().is_null())
            {
                sid_t number;
                if (auto f32d = op->get_data_attr().dyn_cast<math::realNumberData>())
                {
                    number = f32d->represent();
                }
                else if (auto i32d = op->get_data_attr().dyn_cast<math::realNumberData>())
                {
                    number = i32d->represent();
                }
                p.replace_op<lgf::sio::numberOp>(op, op->get_value_desc(), number);
                return resultCode::success();
            }
            return resultCode::pass();
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
            add_rewriter<convertToSIORewriter<lgf::math::sumOp, sio::sumOp>>();
            add_rewriter<convertToSIORewriter<lgf::math::inverseOp, sio::inverseOp>>();
            add_rewriter<convertToSIORewriter<math::negativeOp, sio::negativeOp>>();
            add_rewriter<convertToSIORewriter<math::productOp, sio::scalarProductOp>>();
            add_rewriter<convertToSIORewriter<math::funcCosOp, sio::funcOp>>("cos");
            add_rewriter<convertToSIORewriter<math::funcSineOp, sio::funcOp>>("sin");
            add_rewriter<convertToSIORewriter<math::funcExponentationOp, sio::funcOp>>("exp");
            add_rewriter<convertToSIORewriter<math::funcLogarithmOp, sio::funcOp>>("log");
            add_rewriter<convertToSIORewriter<math::partialDifferentiateOp, sio::partialD>>();
            add_rewriter<convertToSIORewriter<math::differentiateOp, sio::differentialOp>>();

            return apply_rewriter_greedy(p, get_graph());
        }
    };

    std::unique_ptr<passBase> createConvertToSIOPass()
    {
        return std::make_unique<convertToSIOPass>();
    }
}

#endif // TRANSFORM_CONVERT_TO_SIO_H