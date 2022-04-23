file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

include_directories("${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include")

add_executable(Sycamore ${SOURCE_FILES})
