
include_directories(ext/spdlog/include)
include_directories(ext/xxHash)
include_directories(ext/fmt/include)
include_directories(ext/EASTL/include)
include_directories(src)

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(ast)
add_subdirectory(dsl)
add_subdirectory(rt)
add_subdirectory(generator)
add_subdirectory(runtime)
add_subdirectory(backends)

add_library(ocarina-all INTERFACE)
target_link_libraries(ocarina-all INTERFACE
        ocarina-ext
        ocarina-dsl
        ocarina-ast
        ocarina-rt
        ocarina-generator
        ocarina-runtime
        ocarina-core)

add_executable(ocarina main.cpp)
target_link_libraries(ocarina ocarina-all)

add_subdirectory(tests)

set_target_properties(ocarina PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

