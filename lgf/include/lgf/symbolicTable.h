
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
    nestedSymbolicTable(std::string key, meta d): id(key), data(d){}
    nestedSymbolicTable<meta>* findTable(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second);
    }
    meta* getData(){
        return &data;
    }
    meta* getData(std::string key){
        auto itr = table.find(key);
        if( itr == table.end()) return nullptr;
        return &((*itr).second.data);
    }
    nestedSymbolicTable<meta>* addTable(std::string key, meta data){
        if(auto ptr = findTable(key)) return ptr;
        table[key] = nestedSymbolicTable<meta>(key, data);
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
    nestedSymbolicTable<meta>* findNestTable(std::vector<std::string> path, int i=0){
        if(i >= path.size()) return this;
        auto ptr = findTable(path[i]);
        THROW_WHEN(!ptr, "Can't find the module: "+path[i]);
        i++;
        return ptr->findNestTable(path, i);
    }
    
    std::string id;
    meta data;
    std::map<std::string, nestedSymbolicTable<meta>> table;
};

}

#endif