
#include "frame.h"

using namespace MC;
void run_test_ops(){
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<MC::AA::AADialect>();
    ops op("test", &ctx);
    op.init();
    // the test for simplify the operations into (x+y)^2
    // here we first declare x, y
    // and make the x^2+xy+yx+y^2
    variable x = op.declVar("integer");
    variable y = op.declVar("integer");
    variable xy = op.multiply(x,y);
    variable yx = op.multiply(y,x);
    variable xx = op.multiply(x,x);
    variable yy = op.multiply(y,y);
    variable x2y2 = op.add(xx,yy);
    variable xy2 = op.add(yx,xy);
    variable xpy2 = op.add(x2y2,xy2);
    op.dump();

    //TODO: add lowering pass to simplify the operations to (x+y)^2
}