add_subdirectory(cuda)

add_library(nano-backends INTERFACE)
target_link_libraries(nano-backends INTERFACE
        nano-backends-cuda)