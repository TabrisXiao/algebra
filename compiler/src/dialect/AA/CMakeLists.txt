file(GLOB_RECURSE srcs "*.cpp")
set(headers 
    ${PROJECT_SOURCE_DIR}/compiler/include/dialect
    ${PROJECT_BINARY_DIR}/compiler/include/dialect
    )
add_mlir_dialect_library(MLIRAADialectLib
    ${srcs}

    DEPENDS
    AADialectIncGen

    LINK_LIBS PUBLIC
	MLIRIR
    MLIRInferTypeOpInterface
)
target_include_directories(MLIRAADialectLib PUBLIC ${headers})