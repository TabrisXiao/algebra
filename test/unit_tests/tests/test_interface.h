
#pragma once

#include "unit_test_frame.h"
#include "libs/interface/function.h"

using namespace lgi;

namespace test_body{

class test_interface : public test_wrapper{
    public:
    test_interface() {test_id = "interface test";};
    bool run(){
        function::realNumber x;
        auto y = x;
        auto z = x + y;
        auto w = x * y;
        canvas::get().print();
        return 0;
    }
    
};
} // namespace test_qft