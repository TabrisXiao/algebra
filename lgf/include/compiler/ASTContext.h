
#ifndef COMPILER_CONTEXT_H
#define COMPILER_CONTEXT_H
#include "lgf/symbolicTable.h"
#include "lgf/operation.h"
#include "utils.h"
#include <queue>
#include <stack>

namespace lgf::compiler{

struct idinfo {
    //category can be:
    // type : a type name
    // func : a id of a function
    // arg  : argument of a function
    // var  : a variable id
    std::string category;
    location loc;
    std::string type;
    value * handle= nullptr;
};

struct moduleInfo {
    std::string name;
    nestedSymbolicTable<moduleInfo>* parent = nullptr;
    symbolTable<idinfo> ids;
};

class ASTContext : public nestedSymbolicTable<moduleInfo> {
    public:
    ASTContext() {
    }
    nestedSymbolicTable<moduleInfo>* findModule(std::string name){
        return findTable(name);
    } 
    // void addSymbolInfoToScope(std::string name, std::queue<std::string> scope_path, idinfo info){
    //     auto scp = root_scope.findScope(scope_path);
    //     scp.addSymbolInfo(name, info);
    // }

    void addSymbolInfoToCurrentScope(std::string name, idinfo info){
        module->getData()->ids.addEntry(name, info);
    }
    idinfo * findSymbolInfoInCurrentModule(std::string name){
        return module->getData()->ids.find(name);
    }
    // void moveToSubscope(std::string name){
    //     current_scope = &(current_scope->findScope(name));
    // }
    void moveToSubmodule(std::string name){
        if(auto ptr = findModule(name))
            module = ptr;
        THROW_WHEN(true, "The submodule: "+name+" doesn't exists!");
        return;
    }
    void moveToParentModule(){
        module = module->getData()->parent;
    }
    // scope<idinfo>* getScope(std::queue<std::string> path){
    //     return &(current_scope->findScope(path));
    // }
    // void createScopeAndEnter(std::string name){
    //     current_scope->registerScope(name);
    //     current_scope = &(current_scope->findScope(name));
    // }
    void createSubmoduleAndEnter(std::string name){
        module = module->addTable(name, {name, module});
    }
    
    unsigned int module_id = 0;
    nestedSymbolicTable<moduleInfo>* module = this;
};

}

#endif