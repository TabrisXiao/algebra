
#ifndef LGF_MAIN_HEADER
#define LGF_MAIN_HEADER

#include "types.h"
#include "ops.h"
#include "LGFContext.h"
#include "LGFModule.h"
#include "pass.h"

namespace lgf{

class lgfModule : public LGFModule {
    public:
    lgfModule() = default;
    virtual void registerTypes() final {
        REGISTER_TYPE(variable, "variable");
        REGISTER_TYPE(intType, "int");
        REGISTER_TYPE(doubleType, "double");
        REGISTER_TYPE(listType, "list");
    }
    virtual void pipeline(passManager& pm){}
};

}

#endif