
#ifndef LGF_TYPETABLE_H
#define LGF_TYPETABLE_H
#include "symbolicTable.h"
#include <functional>

namespace lgf {
class type_t;
class liteParser;
class LGFContext;
class typeTable {
    public:
    using paser_func_t = std::function<type_t(liteParser &, LGFContext*)>;
    class typeInfo {
        public:
        typeInfo() = default;
        paser_func_t parser;
    };
    typeTable(typeTable &) = delete;
    typeTable(typeTable &&) = delete;
    ~typeTable() { delete instance; }
    static typeTable& get(){
        if(!instance) instance = new typeTable();
        return *instance;
    }
    void registerType(std::string id, paser_func_t& func){
        typeInfo info;
        info.parser = func;
        table.addEntry(id, info);
    }
    template<typename type>
    void registerType(std::string id){
        typeInfo info;
        info.parser = &type::parse;
        table.addEntry(id, info);
    }
    const paser_func_t& findParser(std::string key){
        return table.find(key)->parser;
    }
    protected:
    typeTable() = default;
    inline static typeTable *instance=nullptr;
    symbolTable<typeInfo> table;
};
}
#define REGISTER_TYPE(TYPE, ID)\
    typeTable::get().registerType<TYPE>(ID)
#endif