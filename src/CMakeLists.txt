
#include_directories(ext/spdlog/include)
include_directories(ext/stblib/stb)
include_directories(ext/fmt/include)
include_directories(ext/EASTL/include)
include_directories(ext/imgui/glad)
include_directories(ext/imgui)
include_directories(src)

add_library(ocarina-include INTERFACE)
target_include_directories(ocarina-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(ast)
add_subdirectory(dsl)
add_subdirectory(generator)
add_subdirectory(rhi)
add_subdirectory(backends)
add_subdirectory(util)
add_subdirectory(math)
add_subdirectory(gui)

add_library(ocarina-all INTERFACE)
target_link_libraries(ocarina-all INTERFACE
        ocarina-ext
        ocarina-dsl
        ocarina-ast
        ocarina-generator
        ocarina-backend
        ocarina-rhi
        ocarina-util
        ocarina-math
        ocarina-core)

add_dependencies(ocarina-all ocarina-gui)

add_library(ocarina SHARED main.cpp)
target_link_libraries(ocarina ocarina-all)


option(OCARINA_BUILD_TESTS "Build tests" TRUE)

if(${OCARINA_BUILD_TESTS})
    add_subdirectory(tests)
endif()

set_target_properties(ocarina PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

