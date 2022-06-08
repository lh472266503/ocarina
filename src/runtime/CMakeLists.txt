file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-runtime SHARED ${SOURCE_FILES})
set_target_properties(ocarina-runtime PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_compile_definitions(ocarina-runtime PRIVATE OC_RUNTIME_EXPORT_DLL=1)
target_link_libraries(ocarina-runtime PUBLIC ocarina-ast)