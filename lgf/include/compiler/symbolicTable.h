
#ifndef COMPILER_SYMBOLICTABLE_H
#define COMPILER_SYMBOLICTABLE_H
#include <map>
#include <string>
#include <memory>

namespace lgfc{

template<typename handle>
class symbolTable {
    public:
    symbolTable() = default;
    const handle* get(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second);
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
        table[key] = handle(args...);
    }

    std::map<std::string, handle> table;
};

}

#endif