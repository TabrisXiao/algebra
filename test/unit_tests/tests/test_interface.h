
#pragma once

#include "unit_test_frame.h"
#include "libs/interface/variable.h"
#include "libs/interface/function.h"
#include "libs/interface/stat.h"

using namespace lgi;

namespace test_body{

class test_interface : public test_wrapper{
    public:
    test_interface() {test_id = "interface test";};
    bool run(){
        var x;
        auto y = x;
        auto z = x + y;
        auto w = x * y;
        lgi::stat::normalVariable s;
        canvas::get().print();
        return 0;
    }
    
};
} // namespace test_qft