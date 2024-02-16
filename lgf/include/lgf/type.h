
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

class descriptor {
    public:
    descriptor() = default;
    descriptor(std::string id_): id(id_){}
    std::string getSID(){ return id; }
    void setSID(std::string id_){ id = id_; }
    virtual std::string represent() const {
        return id;
    }
    std::string id;
};

class type_t {
    public:
    type_t () = default;
    type_t (type_t& tp){ desc = tp.desc; }
    type_t (descriptor* desc_): desc(desc_){}
    ~type_t() = default;
    
    void setSID(std::string id_){ desc->id= id_;}
    std::string getSID() {return desc->id;}
    virtual std::string represent() const {
        if(desc) return desc->represent();
        return "";
    }
    bool operator==(const type_t& other){
        return this->represent() == other.represent();
    }
    descriptor* getDesc(){ return desc; }

    template<typename T>
    T* getDesc(){
        return dynamic_cast<T*>(desc);
    }

    descriptor* desc = nullptr;
};


}

#endif