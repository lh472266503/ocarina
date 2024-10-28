
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

pybind11_add_module(ocarina_api ${SOURCE_FILES})
target_link_libraries(ocarina_api PUBLIC EASTL
        ocarina-util
        ocarina-math
        ocarina-core)