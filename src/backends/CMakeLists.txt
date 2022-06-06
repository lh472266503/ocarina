add_subdirectory(cuda)

add_library(katana-backends INTERFACE)
target_link_libraries(katana-backends INTERFACE
        katana-backends-cuda)