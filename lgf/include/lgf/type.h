
#ifndef LGF_TYPE_H_
#define LGF_TYPE_H_
#include "liteParser.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace lgf{
class LGFContext;
class LGFModule;
#define THROW_WHEN(condition, msg) \
    if (condition) { \
        std::cerr<<"Runtime Error: " __FILE__ ":"<< std::to_string(__LINE__)<< ": "<<msg<<"\n"; \
        std::exit(EXIT_FAILURE); \
    }
    
template<typename storageType>
class typeMarker : private bitCode<storageType> {
    public:
    typeMarker(uint8_t size): bitCode(), size_max(size) {}
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

class trait {
    public:
    trait() = default;
    trait(const std::string id_): id(id_){}
    std::string getSID() const { return id; }
    virtual std::string represent() const {
        return id;
    }
    const std::string id;
};

class descriptor {
    public:
    descriptor() = default;
    descriptor(const std::string id_): id(id_){}
    void setSID(std::string sid_){ id = sid_; }
    std::string getSID() const { return id; }
    virtual std::string representType() const { return id; }
    std::string represent() const {
        return representType()+representTraits();
    }
    std::string representTraits() const {
        std::string result = "";
        if(traits.empty()) return result;
        result = " traits = {";
        for(auto& trait: traits){
            result += trait.second->represent() + ", ";
        }
        result.pop_back();
        result.pop_back();
        result += "}";
        return result;
    }
    template<typename trait_t, typename ...ARGS>
    bool addTrait(ARGS ...args){
        auto it = traits.find(trait_t().id);
        if(it != traits.end()) return 1;
        traits[trait_t().id] = std::make_unique<trait_t>(args...);
        return 0;
    }
    template<typename trait_t>
    trait* getTrait(){
        auto it = traits.find(trait_t().id);
        if(it != traits.end()) return it->second.get();
        return nullptr;
    }
    std::map<std::string, std::unique_ptr<trait>> traits;
    private:
    std::string id;
};

class type_t {
    public:
    using desc_t = descriptor;
    static inline const std::string id;
    type_t () = default;
    type_t (const type_t& tp) { desc = tp.desc; }
    type_t (descriptor* desc_): desc(desc_){}
    virtual ~type_t() = default;
    
    std::string getSID() const {return desc->getSID();}
    std::string represent() const {
        if(desc) return desc->represent();
        return "";
    }
    std::string representType() const {
        if(desc) return desc->representType();
        return "";
    }
    bool operator==(const type_t& other){
        return this->representType() == other.representType();
    }
    bool operator!=(const type_t& other){
        return this->representType() != other.representType();
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