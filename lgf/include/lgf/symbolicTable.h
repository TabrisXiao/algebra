
#ifndef COMPILER_SYMBOLICTABLE_H
#define COMPILER_SYMBOLICTABLE_H
#include <map>
#include <string>
#include <memory>

namespace lgf{

template<typename meta>
class symbolTable {
    public:
    symbolTable() = default;
    meta* find(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second);
    }
    bool has(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return 0;
        return 1;
    }
    void addEntry(std::string key, meta entry){
        THROW_WHEN(has(key) ,"The key: "+key+" already exists!");
        table[key] = entry;
    }
    template<typename ...ARGS>
    void addEntry(std::string key, ARGS ... args){
        addEntry(key, meta(args...));
    }

    std::map<std::string, meta> table;
};

}

#endif