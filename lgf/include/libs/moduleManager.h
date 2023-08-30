
#ifndef LGF_MODULETABLE_H
#define LGF_MODULETABLE_H
#include "lgf/symbolicTable.h"
#include "lgf/exception.h"
#include <functional>
#include "Builtin/Builtin.h"
#include "LinearAlg/LinearAlg.h"
#include "lgf/LGFContext.h"

namespace lgf {
class moduleManager {
    public:
    using type_reg_func_t = std::function<void(LGFContext*)>;
    struct moduleInfo {
        type_reg_func_t type_reg_f;
    };
    moduleManager(moduleManager &) = delete;
    moduleManager(moduleManager &&) = delete;
    ~moduleManager() { delete _mtble; }
    static moduleManager &get(){
        if (_mtble != nullptr)
            return *_mtble;
        _mtble = new moduleManager();
        return *_mtble;
    }
    moduleInfo* findInfo(std::string id){
        return table.find(id);
    }

    void loadInternalModule(std::string mid, LGFContext *ctx){
        auto entry = _mtble->findInfo(mid);
        THROW_WHEN(entry==nullptr, "The module: "+mid+" is unknown!");
        entry->type_reg_f(ctx);
    }

    protected:
    moduleManager(){
        table.addEntry("Builtin", {&LGFBaseModule::registerTypes});
        table.addEntry("LinearAlg", {&LinearAlgModule::registerTypes});
    }
    inline static moduleManager *_mtble=nullptr;
    symbolTable<moduleInfo> table;
};
}
#endif