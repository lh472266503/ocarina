
include_directories(ext/spdlog/include)
include_directories(ext/stblib/stb)
include_directories(ext/xxHash)
include_directories(ext/fmt/include)
include_directories(ext/EASTL/include)
include_directories(ext/imgui/glad)
include_directories(ext/imgui)
include_directories(src)

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
        ocarina-gui
        ocarina-core)

add_dependencies(ocarina-all ocarina-gui)

add_executable(ocarina main.cpp)
target_link_libraries(ocarina ocarina-all)

add_subdirectory(tests)

set_target_properties(ocarina PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

