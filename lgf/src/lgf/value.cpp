
#include "lgf/value.h"
#include "lgf/node.h"

namespace lgf{

value::~value() {
    deprecate();
}
//---------------------------------------------------

void value::deprecate(){
    for(auto user : users){
        user->dlink_value(this);
    }
    users.clear();
}
//---------------------------------------------------

void value::remove_user(node* user){
    auto iter = std::find(users.begin(), users.end(), user);
    (*iter)->dlink_value(this);
    if(iter!=users.end()) users.erase(iter);
}
//---------------------------------------------------

std::string value::represent() {
    printer p;
    if(!desc) return p.dump();
    p<<get_sid()<<" "<<desc->represent();
    return p.dump();
}
//---------------------------------------------------
}// namespace lgf