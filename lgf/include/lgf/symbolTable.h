
#ifndef LGF_SYMBOLTABLE_H
#define LGF_SYMBOLTABLE_H
#include <map>
#include <string>

namespace lgf{
template<typename info>
class symbolTable{
    public:
    symbolTable() = default;

    std::map<std::string, info> stbl;
};
}


#endif