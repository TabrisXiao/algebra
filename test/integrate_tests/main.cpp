#include <iostream>
#include "unit_test_frame.h"
#include "test_compile.h"

using namespace test_body;
int main(){
    unit_test_frame frame;
    frame.add_test<test_compile>();
    frame.run_all_test();
    return 0;
}