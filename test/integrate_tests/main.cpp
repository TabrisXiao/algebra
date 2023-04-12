#include <iostream>
#include "unit_test_frame.h"
#include "test_pass.h"

using namespace test_body;
int main(){
    unit_test_frame frame;
    frame.add_test<test_pass>();
    frame.run_all_test();
    return 0;
}