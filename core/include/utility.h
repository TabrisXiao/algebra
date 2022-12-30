
#ifndef CORE_UTILITIES_H_
#define CORE_UTILITIES_H_

namespace utility{
struct Indent
{
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

void indent(int n, std::ostream &os){
    for (int i = 0; i < n; i++)
        os << "  ";
}
}

#endif