
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
add_subdirectory(compile)
add_subdirectory(runtime)
add_subdirectory(backends)

add_library(nano-all INTERFACE)
target_link_libraries(nano-all INTERFACE
        nano-ext
        nano-dsl
        nano-ast
        nano-rt
        nano-compile
        nano-runtime
        nano-core)

add_executable(nano main.cpp)
target_link_libraries(nano nano-all)

add_subdirectory(tests)

set_target_properties(nano PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

