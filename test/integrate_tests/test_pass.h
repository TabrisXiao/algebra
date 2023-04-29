#include "unit_test_frame.h"
#include "lgf/operation.h"
#include "aab/ops.h"
#include "lgf/painter.h"
#include "lgf/lgOps.h"
#include "lgf/pass.h"
#include "aab/passes.h"

using namespace lgf;
namespace test_body{
class test_pass : public test_wrapper{
    
    public:
    test_pass() {test_id = "pass test";};
    bool run() {
        painter builder;
        auto module = builder.createOp<moduleOp>();
        builder.gotoGraph(module->getSubgraph());
        auto varx = builder.createOp<defOp>("x");
        auto vary = builder.createOp<defOp>("y");
        auto xy = builder.createOp<multiplyOp>(varx->output(), vary->output());
        auto yx = builder.createOp<multiplyOp>(vary->output(), varx->output());
        auto xx = builder.createOp<multiplyOp>(varx->output(), varx->output());
        auto yy = builder.createOp<multiplyOp>(vary->output(), vary->output());
        auto add1 = builder.createOp<addOp>(xx->output(), yy->output());
        auto add2 = builder.createOp<addOp>(yx->output(), xy->output());
        auto add3 = builder.createOp<addOp>(add1->output(), add2->output());
        module->assignID(0);
        module->print();
        passManager pm(module);
        pm.enablePrintAfterPass();
        createConvertAddToSumPass(pm);
        createFuseSumOpPassPass(pm);
        createLhsAssociatePass(pm);
        pm.run();
        return 0;
    }
};
};
