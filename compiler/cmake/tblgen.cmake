
# MLIR/LLVM Table Generator utility function. These functions are inspired from mlir cmake modules (mlir/cmake/modules/AddMLIR.cmake)
function(generate_tblgen_command)
# PROJECT: either llmv or mlir
# MODE: -gen-dialect-decls  -gen-dialect-defs etc.
# SOURCE: the source file input to tblgen
# OUTPUT: the output folder from the tblgen
# EXTRA_INCLUDE : includes from other IRs
    cmake_parse_arguments(
        tblgen
        ""
        "PROJECT;OUTPUT;MODE"
        "EXTRA_INCLUDE"
        ${ARGN})
    set(ofn ${tblgen_OUTPUT})
    if(NOT ${tblgen_PROJECT}_TABLEGEN_EXE)
        message(FATAL_ERROR "${tblgen_PROJECT}_TABLEGEN_EXE not set")
    endif()
    if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
        set(tblgen_SOURCE_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    else()
        set(tblgen_SOURCE_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
    endif()

    # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
    if(CMAKE_GENERATOR MATCHES "Ninja")
        # Make output path relative to build.ninja, assuming located on
        # ${CMAKE_BINARY_DIR}.
        # CMake emits build targets as relative paths but Ninja doesn't identify
        # absolute path (in *.d) as relative path (in build.ninja)
        # Note that tblgen is executed on ${CMAKE_BINARY_DIR} as working directory.
        file(RELATIVE_PATH ofn_rel
            ${CMAKE_BINARY_DIR} ${ofn})
        set(additional_cmdline
            -o ${ofn_rel}
            -d ${ofn_rel}.d
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPFILE ${ofn}.d
            )
        set(local_tds)
        set(global_tds)
    else()
        file(GLOB local_tds "*.td")
        file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
        set(additional_cmdline
            -o ${ofn}
            )
    endif()

    if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio has problems with llvm-tblgen's native --write-if-changed
    # behavior. Since it doesn't do restat optimizations anyway, just don't
    # pass --write-if-changed there.
        set(tblgen_change_flag)
    else()
        set(tblgen_change_flag "--write-if-changed")
    endif()

    get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
    list(APPEND tblgen_includes ${tblgen_EXTRA_INCLUDES})
    #get_directory_property(tblgen_includes ${tblgen_PROJECT}-tblgen INCLUDE_DIRECTORIES)
    #list(APPEND tblgen_includes ${tblgen_EXTRA_INCLUDE})
    # Filter out empty items before prepending each entry with -I
    list(REMOVE_ITEM tblgen_includes "")
    list(TRANSFORM tblgen_includes PREPEND -I)

    set(tablegen_exe ${${tblgen_PROJECT}_TABLEGEN_EXE})
    set(tablegen_depends ${${tblgen_PROJECT}_TABLEGEN_TARGET} ${tablegen_exe})
    # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
    # (both the target and the file) to have .inc files rebuilt on
    # a tablegen change, as cmake does not propagate file-level dependencies
    # of custom targets. See the following ticket for more information:
    # https://cmake.org/Bug/view.php?id=15858
    # The dependency on both, the target and the file, produces the same
    # dependency twice in the result file when
    # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
    # but lets us having smaller and cleaner code here.
    message("------------" ${tblgen_MODE})
    add_custom_command(OUTPUT ${ofn}
        COMMAND ${tablegen_exe} ${tblgen_MODE} -I${CMAKE_CURRENT_SOURCE_DIR}
        ${tblgen_includes}
        ${LLVM_TABLEGEN_FLAGS}
        ${tblgen_SOURCE_ABSOLUTE}
        ${tblgen_change_flag}
        ${additional_cmdline}
        # The file in LLVM_TARGET_DEFINITIONS may be not in the current
        # directory and local_tds may not contain it, so we must
        # explicitly list it here:
        DEPENDS ${tablegen_exe}
        ${local_tds} ${global_tds}
        ${tblgen_SOURCE_ABSOLUTE}
        ${LLVM_TARGET_DEPENDS}
        COMMENT "Building ${ofn}..."
    )
    # `make clean' must remove all those generated files:
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${ofn}       PARENT_SCOPE)
    set_source_files_properties(
        ${ofn} PROPERTIES
        GENERATED 1)
endfunction()

function(add_tblgen_command)
    cmake_parse_arguments(
        tblgen
        ""
        "SRCS;OUTPUT;PROJECT;MODE"
        "EXTRA"
        ${ARGN})
    # set up the absolute path for source and output
    if(IS_ABSOLUTE ${tblgen_SRCS})
        set(tblgen_SOURCE_ABSOLUTE ${tblgen_SRCS})
    else()
        set(tblgen_SOURCE_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${tblgen_SRCS})
    endif()

    if(IS_ABSOLUTE ${tblgen_OUTPUT})
        set(tblgen_OUTPUT ${tblgen_OUTPUT})
    else()
        set(tblgen_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${tblgen_OUTPUT})
    endif()

    # make the output folder
    get_filename_component(OUTPUT_FOLDER ${tblgen_OUTPUT} DIRECTORY)
    file(MAKE_DIRECTORY ${OUTPUT_FOLDER})

    # sequence the EXTRA include together
    set(args ${tblgen_EXTRA})
    foreach(hdrs IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS TBLGEN_INCLUDES)
        list(APPEND args -I ${hdrs})
    endforeach()

    set(LLVM_TARGET_DEFINITIONS ${tblgen_SRCS})
    set(TABLEGEN_OUTPUT)

    generate_tblgen_command(
        OUTPUT ${tblgen_OUTPUT}
        MODE ${tblgen_MODE}
        PROJECT ${tblgen_PROJECT}
        EXTRA_INCLUDE ${args}
        )
    set(TBLGEN_OUTPUT_FILE ${TBLGEN_OUTPUT_FILE} ${TABLEGEN_OUTPUT} PARENT_SCOPE)
endfunction()

macro(add_dialect dialect)
    add_tblgen_command(
        PROJECT MLIR
        MODE -gen-dialect-decls
        EXTRA -dialect=dialect
        SRCS ${tblgen_src_base}/dialect/${dialect}/op.td
        OUTPUT ${tblgen_hdrs_output}/dialect/${dialect}/generated/dialect.hpp.inc
    )
    add_tblgen_command(
        PROJECT MLIR
        MODE -gen-dialect-defs
        EXTRA -dialect=${dialect}
        SRCS ${tblgen_src_base}/dialect/${dialect}/op.td
        OUTPUT ${tblgen_hdrs_output}/dialect/${dialect}/generated/dialect.cpp.inc
    )
    add_tblgen_command(
        PROJECT MLIR
        MODE "-gen-op-decls"
        EXTRA "-dialect=${dialect}"
        SRCS ${tblgen_src_base}/dialect/${dialect}/op.td
        OUTPUT ${tblgen_hdrs_output}/dialect/${dialect}/generated/op.hpp.inc
    )
    add_tblgen_command(
        PROJECT MLIR
        MODE "-gen-op-defs"
        EXTRA "-dialect=${dialect}"
        SRCS ${tblgen_src_base}/dialect/${dialect}/op.td
        OUTPUT ${tblgen_hdrs_output}/dialect/${dialect}/generated/op.cpp.inc
    )
    # add_tblgen_command(
    #     PROJECT MLIR
    #     MODE "-gen-dialect-doc"
    #     EXTRA "-dialect=${dialect}"
    #     SRCS ${tblgen_src_base}/dialect/${dialect}/op.td
    #     OUTPUT ${tblgen_docs_output}/dialect/${dialect}/${dialect}.md
    # )
    set(TBLGEN_OUTPUT_FILE ${TBLGEN_OUTPUT_FILE} ${TBLGEN_OUTPUT} PARENT_SCOPE)
endmacro()
