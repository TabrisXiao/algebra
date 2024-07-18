#include "lgf/code.h"

namespace lgf
{
    byteCodeHandle byteCodeStorage::init_code(csize_t c_size)
    {
        auto h = append_data(0, c_size);
        if (h == 64)
            throw std::runtime_error("xbyteCode::init_code: data overflow!");
        return byteCodeHandle(h, c_size, this);
    }
} // namespace lgf