add_subdirectory(cuda)

add_library(ocarina-backends INTERFACE)
target_link_libraries(ocarina-backends INTERFACE
        ocarina-backends-cuda)