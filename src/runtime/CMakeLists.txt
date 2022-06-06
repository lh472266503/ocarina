file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(katana-runtime SHARED ${SOURCE_FILES})
set_target_properties(katana-runtime PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_compile_definitions(katana-runtime PRIVATE KTN_RUNTIME_EXPORT_DLL=1)
target_link_libraries(katana-runtime PUBLIC katana-ast)