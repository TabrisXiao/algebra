#ifndef LGF_SYMBOLTABLE_H
#define LGF_SYMBOLTABLE_H
#include <map>
#include <string>
#include <memory>
#include "lgf/exception.h"

namespace lgf{

namespace codegen{

template<typename handle>
class symbolTable {
    public:
    symbolTable() = default;
    std::unique_ptr<handle>  get(std::string key){
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
    void addInfo(std::string key, ARGS ... args){
        auto is_valid_key = !check(key);
        auto msg = " The key: "+key+" is already exists!";
        CHECK_CONDITION(is_valid_key, msg);
        table[key] = std::make_unique<handle>(args...);
    }

    std::map<std::string, std::unique_ptr<handle>> table;
};

enum info_type_e {
    info_func,
    info_variable,
    info_struct,
    info_template
};

class sketchTypeInfo {
public:
    sketchTypeInfo() = delete;
    sketchTypeInfo(info_type_e & t)
    : t_(t) {}
    
    info_type_e t_;
};

}
}

#endif