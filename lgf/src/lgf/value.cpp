
#include "lgf/value.h"
#include "lgf/node.h"

namespace lgf
{

    //---------------------------------------------------
    std::string value::represent()
    {
        printer p;
        if (!desc)
            return p.dump();
        p << get_sid() << " " << desc->represent();
        return p.dump();
    }

} // namespace lgf