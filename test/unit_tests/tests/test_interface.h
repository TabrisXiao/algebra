
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
        auto y = x;
        auto z = 3+x;
        auto w = x * y;
        canvas::get().print();
        return 0;
    }
    
};
} // namespace test_qft