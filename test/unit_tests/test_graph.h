
#include "unit_test_frame.h"
#include "lgf/operation.h"
#include "lgf/painter.h"
#include "lgf/LGFContext.h"
#include "libs/Builtin/ops.h"

using namespace lgf;
namespace test_body{
// this tests targets to tests painter class on various graphs
class test_painter : public test_wrapper{
    public:
    test_painter() {test_id = "painter test";};
    bool run() {
        LGFContext ctx;
        canvas c;
        painter builder(&ctx);
        builder.gotoGraph(&c);
        auto module = builder.paint<moduleOp>("");
        auto x = builder.paint<declOp>(ctx.getType<variable>());
        auto y = builder.paint<declOp>(ctx.getType<variable>());
        auto y1 = builder.paint<updateOp>(x->output(), y->output());
        auto cst = builder.replaceOp<declOp>(x, ctx.getType<intType>());
        TEST_CHECK_VALUE(x->output()->getUserSize(), 0, "replaceOp failure");
        TEST_CHECK_VALUE(y->output()->getUserSize(), 1, "replaceOp failure");
        TEST_CHECK_VALUE(cst->output()->getUserSize(), 1, "replaceOp failure");

        y1->replaceInputValueBy(cst->output(), x->output());
        TEST_CHECK_VALUE(x->output()->getUserSize(), 1, "replaceOp failure");
        TEST_CHECK_VALUE(cst->output()->getUserSize(), 0, "replaceOp failure");
        c.assignID();
        c.print();
        return isfail;
    }
};
} // namespace test_body