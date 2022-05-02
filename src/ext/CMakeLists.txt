
add_library(sycamore-ext SHARED empty.cpp)
set_target_properties(sycamore-ext PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

#add_library(xxhash INTERFACE)
#target_include_directories(xxhash INTERFACE xxHash)
#target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
#set_target_properties(xxhash PROPERTIES
#        UNITY_BUILD OFF)
#
#target_link_libraries(sycamore-ext PUBLIC xxhash)