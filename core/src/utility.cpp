
#include "utility.h"

void utility::indent(int n, std::ostream &os){
    for (int i = 0; i < n; i++)
        os << "  ";
}