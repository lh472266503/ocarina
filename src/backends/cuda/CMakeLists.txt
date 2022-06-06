file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(nano-backends-cuda SHARED ${SOURCE_FILES})
set_target_properties(nano-backends-cuda PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_compile_definitions(nano-backends-cuda PRIVATE NN_BACKENDS_EXPORT_DLL=1)
target_link_libraries(nano-backends-cuda PUBLIC nano-core)