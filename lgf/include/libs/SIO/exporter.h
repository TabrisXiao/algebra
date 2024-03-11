

#ifndef LIBS_SIO_EXPORT_H
#define LIBS_SIO_EXPORT_H

#include "lgf/global.h"
#include "lgf/operation.h"

namespace lgf::SIO{
class export2Txt {
    public:
    export2Txt(graph* g_) : g(g_) {}

    std::string process(value* val );

    void run(graph* entry);
    void run(){run(g);}
    
    global::stream& os = global::stream::getInstance();
    graph *g = nullptr;
};
} // namespace lgf::SIO

#endif