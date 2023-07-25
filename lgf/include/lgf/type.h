
#ifndef LGF_TYPE_H_
#define LGF_TYPE_H_
#include "liteParser.h"
#include <iostream>
#include <string>

namespace lgf{
class LGFContext;
class typeImpl{
    public:
    typeImpl(std::string sid) : id(sid){}
    virtual std::string represent(){
        return id;
    }
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
        return impl->represent();
    }
    bool operator==(const type_t& other){
        return this->represent() == other.represent();
    }
    typeImpl* getImpl(){ return impl; }

    typeImpl* impl = nullptr;
};

}

#endif