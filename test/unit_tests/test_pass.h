#include "unit_test_frame.h"
#include "operation.h"
#include "ops.h"

using namespace aog;
namespace test_body{
class test_pass : public test_wrapper{
    
    public:
    test_pass() {test_id = "pass test";};
    bool run() {
        
        return 0;
    }
};
};
