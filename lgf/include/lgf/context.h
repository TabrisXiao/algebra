
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

        template <typename T, typename... Args>
        T *get_data_attr(Args... args)
        {
            auto attr = std::make_unique<T>(args...);
            auto ptr = attr.get();
            data.push_back(std::move(attr));
            return ptr;
        }

        std::vector<std::unique_ptr<dataAttr>> data;
    };

}

#endif