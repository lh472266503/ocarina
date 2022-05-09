
include_directories(ext/spdlog/include)
include_directories(ext/xxHash)
include_directories(ext/fmt/include)
include_directories(src)

add_subdirectory(ext)
add_subdirectory(core)
add_subdirectory(ast)
add_subdirectory(dsl)
add_subdirectory(tests)

add_library(katana-all INTERFACE)
target_link_libraries(katana-all INTERFACE
        katana-ext
        katana-ast
        katana-core)

add_executable(katana main.cpp)
target_link_libraries(katana katana-all)

set_target_properties(katana PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

