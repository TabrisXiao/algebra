
#ifndef EDSL_SYMBOLTABLE_H_
#define EDSL_SYMBOLTABLE_H_

#include "config.h"
#include <unordered_map>
#include "llvm/ADT/StringRef.h"
namespace MC{
class symbolTable{
    public:
    //the symbolTable should be a singleton;
    symbolTable(symbolTable &) = delete;
    void operator=(const symbolTable &) = delete;
    static symbolTable *getInstance(){
        if(gSymbolTable != nullptr)
            return gSymbolTable;
        gSymbolTable = new symbolTable();
        return gSymbolTable;
    }
    bool registerSymbol(std::string id){
        if(!lookup(id)){
            _table[id]=true;
            nid++;
            return SUCCESS;
        }
        else return FAIL;
    }
    void eraseSymbol(std::string id){
        if(lookup(id)){
            _table[id]=false;
            nid--;
        }
        return;
    }
    bool lookup(std::string id){
        if(_table.find(id)==_table.end()) return FAIL;
        return SUCCESS;
    }
    
    protected:
    symbolTable(){}
    std::unordered_map<std::string, bool> _table;
    uint64_t nid = 0;
    inline static symbolTable *gSymbolTable= nullptr;
};
}


#endif