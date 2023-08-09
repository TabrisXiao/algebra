
#ifndef COMPILER_SYMBOLICTABLE_H
#define COMPILER_SYMBOLICTABLE_H
#include <map>
#include <string>
#include <memory>
#include <queue>

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

template<typename meta>
class nestedSymbolicTable{
    public: 
    nestedSymbolicTable() = default;
    virtual ~nestedSymbolicTable(){}
    nestedSymbolicTable(std::string key): id(key){}
    nestedSymbolicTable<meta>* findTable(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second);
    }
    meta* getData(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second.data);
    }
    nestedSymbolicTable<meta>* addTable(std::string key){
        if(auto ptr = findTable(key)) return ptr;
        table[key] = nestedSymbolicTable<meta>(key);
        return &(table[key]);
    }
    nestedSymbolicTable<meta>* findTable(std::queue<std::string>& keys){
        while(keys.size()){
            auto key = keys.pop();
            if(auto ptr = findTable(key))
                return ptr->findTable(keys);
            return nullptr;
        }
    }
    
    std::string id;
    meta data;
    std::map<std::string, nestedSymbolicTable<meta>> table;
};

}

#endif