
#include "ops.h"
using namespace aog;
void test_aog(){
    
    context ctx;
    defOp def1(&ctx), def2(&ctx), def3(&ctx);
    addOp op1(&ctx, def1.output(), def2.output());
    multiplyOp op2(&ctx, op1.output(), def3.output());
    ctx.print();
}