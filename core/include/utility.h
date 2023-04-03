
#ifndef CORE_UTILITIES_H_
#define CORE_UTILITIES_H_
#include <vector>
#include <iostream>


namespace utility{
struct Indent
{
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

void indent(int n, std::ostream &os);

// remove all objects that equal to the given value from the vector
template<typename T>
void remove_value_from_vector(T value, std::vector<T> & vec){
    auto iter = std::find(vec.begin(), vec.end(), value);
    while(iter!=vec.end()){
        vec.erase(iter);
        iter = std::find(vec.begin(), vec.end(), value);
    }
}

// check if the value exists in the vector, if not, push_back it.
template<typename T>
void check_push_back(T obj, std::vector<T> &vec){
    auto iter = std::find(vec.begin(), vec.end(), obj);
    if(iter==vec.end()) vec.push_back(obj);
}
}


#endif