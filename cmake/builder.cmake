include(CMakeParseArguments)

function(create_cpp_library)
# TYPE: STATIC
# COPT: The option for compiler
    cmake_parse_arguments(
        RULE
        "PUBLIC;ALWAYSLINK;TESTONLY;WHOLEARCHIVE"
        "NAME;TYPE"
        "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES;PROPS;PRIVATE_DEPS;COMPONENTS"
        ${ARGN})
    
    add_library(${RULE_NAME} ${RULE_TYPE} "")
    
    target_sources(
        ${RULE_NAME}
        PRIVATE
        ${RULE_SRCS}
        ${RULE_HDRS}
    )

    target_include_directories(${RULE_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${RULE_INCLUDES}>
    )
    target_compile_options(${RULE_NAME}
        PRIVATE
        ${RULE_COPTS}
    )
    # Install header files
    install(
        TARGETS ${RULE_NAME}
        DESTINATION ${INSTALL_LIBRARY_DIR}
    )
    file(RELATIVE_PATH _HDR_INSTALL_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    foreach(_COMPONENT ${RULE_COMPONENTS})
        install(
            FILES ${RULE_HDRS}
            DESTINATION "include/${_HDR_INSTALL_PATH}"
            COMPONENT ${_COMPONENT}
        )
    endforeach()

endfunction() # create_cpp_library

function(create_cpp_binary)
    cmake_parse_arguments(
        RULE
        "TESTONLY"
        "NAME;OUT"
        "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
        ${ARGN})
    
    add_executable(${RULE_NAME} "")
    if(RULE_SRCS)
        target_sources(${RULE_NAME}
        PRIVATE
            ${RULE_SRCS}
        )
    endif()
    if(RULE_OUT)
        set_target_properties(${RULE_NAME} PROPERTIES OUTPUT_NAME "${RULE_OUT}")
    else()
        set_target_properties(${RULE_NAME} PROPERTIES OUTPUT_NAME "${RULE_NAME}")
    endif()

    target_compile_options(
        ${RULE_NAME}
        PRIVATE
        ${RULE_COPTS}
    )
    target_link_options(
        ${RULE_NAME}
        PRIVATE
        ${RULE_LINKOPTS}
    )
    target_link_libraries(${RULE_NAME} 
        PUBLIC 
        ${RULE_DEPS})
    target_compile_options(
        ${RULE_NAME}
        PRIVATE
        ${RULE_COPTS}
    )

    # setup the output for debugging
    set_target_properties( ${RULE_NAME}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"
    )

    # collect all the binaries and libraries to the pre-defined directory
    install(
        TARGETS ${RULE_NAME}
        COMPONENT ${RULE_NAME}
        RUNTIME DESTINATION "${INSTALL_RUNTIME_DIR}"
        LIBRARY DESTINATION "${INSTALL_LIBRARY_DIR}"
        ARCHIVE DESTINATION "${INSTALL_ARCHIVE_DIR}"
    )
endfunction()

function(import_cpp_library)
cmake_parse_arguments(
    RULE
    "TESTONLY"
    "NAME;TYPE"
    "LIBRARY;INTERFACE;IMPORT"
    ${ARGN})
# TYPE: STATIC or SHARED
# LIB: the path to the static library (for STATIC) or the runtime library (for SHARED)
# IMPORT: the path to import library. It is only needed for SHARED library
    add_library(${RULE_NAME} ${RULE_TYPE} IMPORTED)

    set_target_properties(
        ${RULE_NAME} 
        PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES 
        ${RULE_INTERFACE})
    set_target_properties(
        ${RULE_NAME} 
        PROPERTIES 
        IMPORTED_LOCATION 
        ${RULE_LIBRARY})
    if(${RULE_TYPE} STREQUAL "SHARED")
        if(RULE_IMPORT)
            set_target_properties(
                ${RULE_NAME} 
                PROPERTIES 
                IMPORTED_IMPLIB 
                ${RULE_IMPORT})
        endif()
    endif()

    foreach(_DEP ${RULE_DEPS})
        add_dependencies(
            ${RULE_NAME}
            ${_DEP}
            ${INSTALL_RUNTIME_DIR}
        )
    endforeach()

    file(COPY ${RULE_LIBRARY}
        DESTINATION "${CMAKE_BINARY_DIR}/bin"
    )
    # collects the .dll to the bin folder so that executor can find them on RUNTIME
    install(
        FILES ${RULE_LIBRARY}
        DESTINATION ${INSTALL_RUNTIME_DIR}
    )
endfunction()