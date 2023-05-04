#include "unit_test_frame.h"
#include "codegen/ast.h"
#include "lgf/painter.h"
#include <string>

using namespace lgf;
namespace test_body{

class test_codegen : public test_wrapper{
    
    public:
    test_codegen() {test_id = "operation test";};
    bool run() {
        graph reg;
        painter pntr(&reg);
        pntr.createOp<defClass>("node");
        pntr.createOp<declVar>("node");
        reg.assignID();
        reg.print();
        return 0;
    }
};
};
