#include <iostream>
#include "unit_test_frame.h"
#include "test_operation.h"
#include "test_codegen.h"
#include "test_ast.h"

using namespace test_body;
int main(){
    unit_test_frame frame;
    //frame.add_test<test_operation>();
    frame.add_test<test_codegen>();
    frame.add_test<test_ast>();
    frame.run_all_test();
    return 0;
}