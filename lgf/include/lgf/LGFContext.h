
#ifndef LGFCONTEXT_H_
#define LGFCONTEXT_H_

#include "type.h"
#include "symbolicTable.h"
#include "typeTable.h"
#include <memory>
#include <vector>
#include "LGFModule.h"
//#include "moduleTable.h"

namespace lgf{

class LGFContext {
    public: 
    using mTable = symbolTable<std::unique_ptr<LGFModule>>;
    using regionTable = nestedSymbolicTable<mTable>;
    LGFContext() = default;
    template<typename tp>
    tp getOrCreateDesc(std::unique_ptr<descriptor>&& imp){
        tp t;
        t.desc = imp.get();
        if(auto ptr = findTypeDesc(imp.get())){
            t.desc = ptr;
            return t;
        }
        descs.push_back(std::move(imp));
        return t;
    }
    template<typename tp, typename ...ARGS>
    tp getType(ARGS ...args){
        auto ptr = std::make_unique<tp::desc_t>(tp::sid, args...);  
        return getOrCreateDesc<tp>(std::move(ptr));
    }
    template<typename tp>
    tp getType(){
        auto ptr = std::make_unique<tp::desc_t>(tp::sid);
        return getOrCreateDesc<tp>(std::move(ptr));
    }
    
    descriptor* findTypeDesc(descriptor *ptr){
        for(auto & tp : descs){
            if( ptr->getSID() != tp->getSID() ) continue;
            if( ptr->represent() == tp->represent() ) return tp.get();
        }
        return nullptr;
    }
    template<typename module_t>
    module_t* registerModule(std::string name){
        // the name of module should be like
        // region.subregion.module_name
        liteParser parser.loadBuffer(name);
        std::string id = parser.parseIdentifier();
        auto* rptr = root_region.createOrGetTable(id);
        while(parser.getCurToken()==int('.')){
            parser.parseDot();
            id = parser.parseIdentifier();
            if(parser.getCurToken()!=int('.')) break;
            rptr=rptr->createOrGetTable(id);
        }
        if(auto module = rptr->getData()->find(id)) return module.get();
        return rptr->getData()->addEntry(id, module_t());
    }
    template<typename type>
    void registerType(std::string id){
        tptable.registerType<type>(id);
    }
    typeTable& getTypeTable() { return tptable; }
    type_t parseTypeStr(std::string str){
        return tptable.parseTypeStr(this, str);
    }
    

    std::vector<std::unique_ptr<descriptor>> descs;
    regionTable root_region;
    typeTable tptable;
};

}

#endif