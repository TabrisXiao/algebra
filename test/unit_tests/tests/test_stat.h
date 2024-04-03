
#pragma once

#include "unit_test_frame.h"
#include "libs/interface/function.h"
#include "libs/interface/stat.h"

using namespace lgi;

namespace test_body{

class test_stat : public test_wrapper{
    public:
    test_stat() {test_id = "stat lib test";};
    bool run(){
        lgi::stat::normalVariable x;
        auto y = function::exp(x);
        canvas::get().print();
        return 0;
    }
};
} // namespace test_qft