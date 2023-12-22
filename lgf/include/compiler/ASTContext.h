
#ifndef COMPILER_CONTEXT_H
#define COMPILER_CONTEXT_H
#include "lgf/symbolicTable.h"
#include "lgf/operation.h"
#include "lgf/liteParser.h"
#include "utils.h"
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
    bool initialized = false;
    value * handle= nullptr;
};


struct moduleInfo {
    std::string name;
    nestedSymbolicTable<moduleInfo>* parent = nullptr;
    symbolTable<idinfo> ids;
    value* ref = nullptr;
};

class ASTContext : public nestedSymbolicTable<moduleInfo> {
    public:
    ASTContext() {
    }
    nestedSymbolicTable<moduleInfo>* findL1Module(std::string name){
        return findTable(name);
    }
    nestedSymbolicTable<moduleInfo>* findModule(std::string name){
        // module name has the form:
        // parent::child::module
        liteParser lp(name);
        auto id = lp.parseIdentifier();
        if(lp.isEOF()){
            return module->findTable(id);
        }
        module = module->findTable(id);
        return findModule(lp.getBuffer());
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
    idinfo * findSymbol(std::string id, std::string path){
        if(auto m = findModule(path)){
            if(auto info = m->getData()->ids.find(id))
                return info;
        }
        return nullptr;
    }
    std::string getCurrentModuleName(){
        return module->getData()->name;
    }
    bool hasSymbol(std::string name){
        if(auto ptr = findSymbolInfoInCurrentModule(name)) return 1;
        return 0;
    }
    // void moveToSubscope(std::string name){
    //     current_scope = &(current_scope->findScope(name));
    // }
    void moveToSubmodule(std::string name){
        if(auto ptr = module->findTable(name)){
            module = ptr;
            return;
        }
        THROW_WHEN(true, "The submodule: "+name+" doesn't exists!");
        return;
    }
    void moveTo(std::vector<std::string>& path, int n=0){
        if(n>= path.size()) return;
        if(n==0) resetModulePtr();
        moveToSubmodule(path[n]);
        n++;
        return moveTo(path, n);
    }
    void moveToParentModule(){
        module = module->getData()->parent;
        //abs_path.pop_back();
    }
    // scope<idinfo>* getScope(std::queue<std::string> path){
    //     return &(current_scope->findScope(path));
    // }
    // void createScopeAndEnter(std::string name){
    //     current_scope->registerScope(name);
    //     current_scope = &(current_scope->findScope(name));
    // }
    void createSubmoduleAndEnter(std::string name){
        if(name.empty()) return;
        liteParser lp(name);
        auto id = lp.parseIdentifier();
        module = module->addTable(name, {name, module});
        if(!lp.isEOF()){
            lp.parseDot();
            return createSubmoduleAndEnter(lp.getBuffer());
        }
        //abs_path.push_back(name);
    }
    void resetModulePtr(){
        module = this;
        //abs_path.clear();
    }
    void deleteCurrentModule(){
        auto name = module->getData()->name;
        moveToParentModule();
        auto iter = module->table.find(name);
        module->table.erase(iter);
    }
    
    unsigned int module_id = 0;
    nestedSymbolicTable<moduleInfo>* module = this;
    //std::vector<std::string> abs_path;
};

}

#endif