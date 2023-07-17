
#ifndef COMPILER_CONTEXT_H
#define COMPILER_CONTEXT_H
#include "symbolicTable.h"
#include "utils.h"
#include <queue>

namespace lgf::compiler{

class context {
    public:
    struct idinfo {
        //category can be:
        // type : a type name
        // func : a id of a function
        // class: a id of a class object
        // variable: a variable id
        std::string category;
        location loc;
        std::string extra="";
    };
    class scope{
        public:
        scope() = default;
        scope(scope* p) : parent(p) {}
        idinfo* findSymbolInfo(std::string id){
            return stbl.find(id);
        }
        void addSymbolInfo(std::string id, idinfo info){
            stbl.addEntry(id, info);
        }
        bool hasSymbolInfo(std::string id){
            return stbl.has(id);
        }
        scope& findScope(std::string name) {
            // parse scope string
            auto & iter = subscope.find(name);
            THROW_WHEN(iter == subscope.end(), "Scope name: \'"+name+"\' doesn't     exist!");
            return (*iter).second;
        }
        scope& findScope(std::queue<std::string>& path){
            if(path.size() == 0 ) return *this;
            auto &_scope = subscope.find(path.front());
            path.pop();
            return (*_scope).second.findScope(path);
        }
        bool hasScope(std::string name){
            if(subscope.find(name) == subscope.end()) return 0;
            return 1;
        }
        void registerScope(std::string name){
            if(hasScope(name)) return;
            subscope.insert(std::make_pair(name, scope(this)));
        }
        symbolTable<idinfo> stbl;
        scope* parent = nullptr;
        std::map<std::string, scope> subscope;
    };

    context() { current_scope = &root_scope; }
    scope& findScope(std::string name){
        return root_scope.findScope(name);
    }
    void addSymbolInfoToScope(std::string name, std::queue<std::string> scope_path, idinfo info){
        auto scp = root_scope.findScope(scope_path);
        scp.addSymbolInfo(name, info);
    }

    void addSymbolInfoToCurrentScope(std::string name, idinfo info){
        current_scope->addSymbolInfo(name,info);
    }
    void moveToSubscope(std::string name){
        current_scope = &(current_scope->findScope(name));
    }
    void moveToSubscope(std::queue<std::string> path){
        current_scope = &(current_scope->findScope(path));
    }
    void moveToParentScope(){
        current_scope = current_scope->parent;
    }
    
    scope* current_scope = nullptr;
    scope root_scope;
};
}

#endif