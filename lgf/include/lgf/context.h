
#ifndef LGFCONTEXT_H_
#define LGFCONTEXT_H_

#include "value.h"
#include <memory>
#include <vector>
#include "attribute.h"

namespace lgf
{

    class LGFContext
    {
    public:
        LGFContext() = default;
        ~LGFContext() = default;
        LGFContext(LGFContext &) = delete;
    };

}

#endif