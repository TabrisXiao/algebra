#include <iostream>
#include "unit_test_frame.h"
#include "tests\test_qft.h"
//#include "test_codegen.h"

int main(){
    using namespace test_body;
    unit_test_frame frame;
    //frame.add_test<test_operation>();
    //frame.add_test<test_codegen>();
    frame.add_test<test_qft>();
    frame.run_all_test();
    return 0;
}