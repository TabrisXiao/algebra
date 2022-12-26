
#ifndef CORE_UTILITIES_H
#define CORE_UTILITIES_H

struct Indent
{
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

#endif