
#include "dialect/AA/dialect.hpp"

void MC::registerDialect(mlir::DialectRegistery register)
{
    register.insert<MC::AA::AADialect>();
}