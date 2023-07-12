
#ifndef COMPILER_SYMBOLICTABLE_H
#define COMPILER_SYMBOLICTABLE_H
#include <map>
#include <string>
#include <memory>

namespace lgf::compiler{

template<typename handle>
class symbolTable {
    public:
    symbolTable() = default;
    std::unique_ptr<handle> get(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return std::unique_ptr<handle>();
        return *itr;
    }
    bool check(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return 0;
        return 1;
    }
    template<typename ...ARGS>
    void addSymbol(std::string key, ARGS ... args){
        auto is_valid_key = !check(key);
        auto msg = " The key: "+key+" is already exists!";
        CHECK_CONDITION(is_valid_key, msg);
        table[key] = std::make_unique<handle>(args...);
    }

    std::map<std::string, std::unique_ptr<handle>> table;
};

}

#endif