
#ifndef TRANSFORM_ALGEBRA_SIMPLIFICATION_PASS
#define TRANSFORM_ALGEBRA_SIMPLIFICATION_PASS
#include "libs/builtin/passes.h"
#include "libs/math/algebra/ops.h"
namespace lgf::math
{

    class zeroRewriter : public rewriter<cstDeclOp>
    {
    public:
        zeroRewriter() = default;
        virtual resultCode rewrite(painter &p, cstDeclOp *op);
    };

    class unitRewriter : public rewriter<cstDeclOp>
    {
    public:
        unitRewriter() = default;
        virtual resultCode rewrite(painter &p, cstDeclOp *op);
    };
} // namespace lgf
#endif
