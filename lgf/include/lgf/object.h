
#ifndef LGF_OBJECT_H_
#define LGF_OBJECT_H_
#include <iostream>
#include <string>

namespace lgf{

//symbolic id;
using sid_t = std::string;

class graphObject {
    public:
    graphObject() = default;
    graphObject(sid_t id) : sid(id) {}
    std::string get_sid() const { return sid; }
    void set_sid(sid_t id) { sid = id; }
    bool set_sid_if_null(sid_t id){
        if(sid.empty()) {
            sid=id;
            return 1;
        }
        return 0;
    }
    void set_nid(uint64_t id){
        nid = id;
    }
    uint64_t get_nid() const { return nid; }
    virtual sid_t represent() = 0;
    protected:
    sid_t sid="";
    uint64_t nid = 0;
};

}

#endif