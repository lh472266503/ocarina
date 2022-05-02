file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(sycamore-core SHARED ${SOURCE_FILES})
target_link_libraries(sycamore-core PUBLIC sycamore-ext)
target_compile_definitions(sycamore-core PRIVATE SCM_CORE_EXPORT_DLL=1)
set_target_properties(sycamore-core PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)