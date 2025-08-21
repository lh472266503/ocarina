cmake_minimum_required(VERSION 3.22)

set(PYTHON_PATH  "DEFAULT_VALUE_IF_NOTHING_ELSE_SET")
set(CUSTOM_PYTHON_PATH "" CACHE PATH "Optional: Specify a custom path to the Python executable if not found automatically.")

find_program(PYTHON_EXECUTABLE NAMES python python3 python2
                             PATHS ENV PATH)

if (PYTHON_EXECUTABLE)
    message(STATUS "Found Python executable: ${PYTHON_EXECUTABLE}")
    set(PYTHON_PATH "${PYTHON_EXECUTABLE}")
else()
    message(STATUS "Python executable not found in standard paths.")
    if (CUSTOM_PYTHON_PATH) # Check if the user provided a custom path
        message(STATUS "Using custom Python path from argument: ${CUSTOM_PYTHON_PATH}")
        set(PYTHON_PATH "${CUSTOM_PYTHON_PATH}")
    else()
        message(FATAL_ERROR "No custom Python path provided.")
    endif()
endif()

project(ocarina)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/")

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

add_subdirectory(src)
