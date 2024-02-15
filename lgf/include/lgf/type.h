
#ifndef LGF_TYPE_H_
#define LGF_TYPE_H_
#include "liteParser.h"
#include <iostream>
#include <string>

namespace lgf{
class LGFContext;
class LGFModule;
#define THROW_WHEN(condition, msg) \
    if (condition) { \
        std::cerr<<"Runtime Error: " __FILE__ ":"<< std::to_string(__LINE__)<< ": "<<msg<<"\n"; \
        std::exit(EXIT_FAILURE); \
    }
template<typename storageType>
class typeMarker : private byteCode<storageType> {
    public:
    typeMarker(uint8_t size): byteCode(), size_max(size) {}
    bool is(uint8_t code) { 
        THROW_WHEN(code > size_max, "typeMarker: code out of range");
        return ((1<<code) & value) != 0 ; }
    void mark(uint8_t code ) { 
        THROW_WHEN(code > size_max, "typeMarker: code out of range");
        value |= 1<<code; 
    }
    void combine(storageType& val) { value |= val; }
    uint8_t size_max=64;
    storageType value=0;
};

class typeImpl{
    public:
    typeImpl(std::string sid) : id(sid){}
    virtual std::string represent(){
        return id;
    }
    LGFModule* getModule(){ return module; }
    std::string getSID(){ return id;}
    // the module that this type belongs to
    LGFModule* module=nullptr;
    std::string id;
    
};

class type_t {
    public:
    type_t () = default;
    type_t (typeImpl* ptr){ impl=ptr; }
    ~type_t() = default;
    // for a new type_t, it need to have a static function build like
    // static std::unque_ptr<typeImpl> createImpl(args...)
    void setID(std::string id_){ impl->id= id_;}
    std::string getSID() {return impl->id;}
    virtual std::string represent() const {
        if(impl) return impl->represent();
        return "Null";
    }

    bool operator==(const type_t& other){
        return this->represent() == other.represent();
    }
    typeImpl* getImpl(){ return impl; }

    template<typename T>
    T* getImplAs(){
        return dynamic_cast<T*>(impl);
    }

    typeImpl* impl = nullptr;
};

}

#endif