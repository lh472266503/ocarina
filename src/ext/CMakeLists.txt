
add_library(ocarina-ext SHARED empty.cpp)
set_target_properties(ocarina-ext PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE xxHash)
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
set_target_properties(xxhash PROPERTIES
        UNITY_BUILD OFF)

target_link_libraries(ocarina-ext PUBLIC xxhash)

set(FMT_OS OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt)
target_compile_definitions(fmt-header-only INTERFACE
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)

target_link_libraries(ocarina-ext PUBLIC fmt::fmt-header-only)

add_subdirectory(EASTL)
target_link_libraries(ocarina-ext PUBLIC EASTL)

add_subdirectory(stblib)
add_subdirectory(tinyexr)


add_subdirectory(glfw)
add_subdirectory(imgui)
add_library(glfw_config INTERFACE)
target_link_libraries(glfw_config INTERFACE glfw)

add_library(ocarina-gui-ext INTERFACE)
target_link_libraries(ocarina-gui-ext INTERFACE glfw_config ocarina-imgui)


target_link_libraries(ocarina-ext PUBLIC
        ocarina-ext-stb
        ocarina-ext-tinyexr
        )