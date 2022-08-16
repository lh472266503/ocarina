

function(ocarina_add_test name)
    cmake_parse_arguments(TEST "" "" "SOURCES" ${ARGN})
    add_executable(${name}  ${TEST_SOURCES})
    target_link_libraries(${name} ocarina-all)
    set_target_properties(${name} PROPERTIES
            PUBLIC_HEADER "${HEADER_FILES}"
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            WINDOWS_EXPORT_ALL_SYMBOLS ON)
endfunction()


ocarina_add_test(test-callable SOURCES test_callable.cpp)
ocarina_add_test(test-parsetype SOURCES test_parse_type.cpp)
ocarina_add_test(test-kernel SOURCES test_kernel.cpp)
ocarina_add_test(test-imageio SOURCES test_image_io.cpp)
ocarina_add_test(test-accel SOURCES test_accel.cpp)
ocarina_add_test(test-gui SOURCES test_gui.cpp)
