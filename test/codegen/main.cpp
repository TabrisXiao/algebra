

#include "codegen/codegen.h"

int main()
{
    using namespace lgf::codegen;

    const std::string test = "./test/codegen/test.input";
    lgfOpCodeGen gen;

    gen.run(test);
}