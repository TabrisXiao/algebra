
#ifndef LGF_TYPE_H_
#define LGF_TYPE_H_
#include "exception.h"
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

class objectAbstract {
    public:
    objectAbstract(std::string sid): id(sid){}
    virtual std::string represent() const = 0;
    std::string getSID(){ return id;}
    void setSID(std::string sid){ id = sid;}
    std::string id;
};

class objectType {
    public:
    objectType() = default;
    objectType(objectType& other) {
        abstract = other.abstract;
    }
    objectType(objectAbstract *ptr) { abstract = ptr;}
    
    virtual std::string represent() const {
        if(abstract) return abstract->represent();
        THROW("Null objectAbstract")
        return "";
    }
    bool operator==(const objectType& other){
        return this->represent() == other.represent();
    }
    void setSID(std::string sid){
        abstract->setSID(sid);
    }

    objectAbstract* getAbstract(){ return abstract; }

    template<typename T>
    T* getAbstract(){
        return dynamic_cast<T*>(abstract);
    }

    objectAbstract* abstract = nullptr;
};

}

#endif