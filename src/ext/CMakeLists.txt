
add_library(nano-ext SHARED empty.cpp)
set_target_properties(nano-ext PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE xxHash)
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
set_target_properties(xxhash PROPERTIES
        UNITY_BUILD OFF)

target_link_libraries(nano-ext PUBLIC xxhash)

set(FMT_OS OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt)
target_compile_definitions(fmt-header-only INTERFACE
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)

target_link_libraries(nano-ext PUBLIC fmt::fmt-header-only)

add_subdirectory(EASTL)
target_link_libraries(nano-ext PUBLIC EASTL)