
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

pybind11_add_module(ocapi ${SOURCE_FILES})
target_link_libraries(ocapi PUBLIC EASTL
        ocarina-util
        ocarina-math
        ocarina-core)

add_custom_command(
        TARGET ocapi
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copy pyd"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND python.exe ${CMAKE_SOURCE_DIR}/src/scripts/copy_pyd.py
        COMMAND ${CMAKE_COMMAND} -E echo "Complete copy!"
)
