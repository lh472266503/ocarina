function(ocarina_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(ocarina-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(ocarina-backend-${name} PRIVATE
            ocarina-ast
            ocarina-runtime
            ocarina-compile)
    add_dependencies(ocarina-backend ocarina-backend-${name})
    set_target_properties(ocarina-backend-${name} PROPERTIES
            UNITY_BUILD OFF
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            DEBUG_POSTFIX "")
endfunction()

add_library(ocarina-backend INTERFACE)

add_subdirectory(cuda)

#target_link_libraries(ocarina-backend INTERFACE
#        ocarina-backend-cuda)