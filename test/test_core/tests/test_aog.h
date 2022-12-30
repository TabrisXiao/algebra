
#include "ops.h"
using namespace aog;
void test_aog(){
    
    context ctx;
    defOp def1(&ctx, "Symbolic"), def2(&ctx, "Symbolic");
    multiplyOp xx(&ctx, def1.output(), def1.output());
    multiplyOp xy(&ctx, def1.output(), def2.output());
    multiplyOp yx(&ctx, def2.output(), def1.output());
    multiplyOp yy(&ctx, def2.output(), def2.output());
    addOp op1(&ctx, xx.output(), xy.output());
    addOp op2(&ctx, yy.output(), yx.output());
    addOp op3(&ctx, op1.output(), op2.output());
    ctx.print();
}