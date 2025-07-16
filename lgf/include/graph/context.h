#ifndef LGF_CONTEXT_H_
#define LGF_CONTEXT_H_
#include "operation.h"
#include <memory>

namespace lgf
{
    class context : public operation
    {
    public:
        context() : operation("context")
        {
            add_region();
        }
        virtual std::string represent() override
        {
            return get_name() + " " + get_region(0)->represent();
        }
        virtual std::unique_ptr<operation> copy() override
        {
            return clone<context>();
        }
    };

} // namespace lgf

#endif