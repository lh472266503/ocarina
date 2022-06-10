
add_executable(test-callable test_callable.cpp)
target_link_libraries(test-callable ocarina-all)

set_target_properties(test-callable PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_executable(test-parsetype test_parse_type.cpp)
target_link_libraries(test-parsetype ocarina-all)

set_target_properties(test-parsetype PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON)