
#ifndef LGF_MODULE_H
#define LGF_MODULE_H

#include "painter.h"

namespace lgf{

class LGFModule{
    public:
    LGFModule() = default;
    virtual void registerTypes() {}
    virtual void registerInterface() {}
    virtual void registerInitPass() {}
};

}

#endif