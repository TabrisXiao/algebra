
#pragma once

#include "unit_test_frame.h"
#include "libs/Builtin/Builtin.h"
#include "lgf/painter.h"

namespace test_body{

class test_builtin : public test_wrapper{
    public:
    test_builtin() {test_id = "builtin lib test";};
    bool run(){
        moduleOp module;
        painter p(&module);
        auto ctx = p.get_context();
        auto intV = ctx->get_desc<intValue>();
        auto data = lgf::intData(3);
        auto cst = p.paint<lgf::cstDeclOp>(intV, &data);
        auto x = p.paint<lgf::declOp>(intV);
        auto assign = p.paint<lgf::updateOp>(x, cst);
        auto ret = p.paint<lgf::returnOp>(assign);
        module.print();
        return 0;
    }
};
} // namespace test_qft