
#pragma once

#include "unit_test_frame.h"
#include "libs/interface/interface.h"

using namespace lgi;

namespace test_body{

class test_interface : public test_wrapper{
    public:
    test_interface() {test_id = "interface test";};
    bool run(){
        variable x;
        auto y = function::cos(x);
        auto z = 3+x;
        auto w = x * y;
        auto a = function::d(y, x, 2);
        a.check();
        canvas::get().get_pass_manager().set_log_level(2);
        canvas::get().compile();
        return 0;
    }
};
} // namespace test_qft