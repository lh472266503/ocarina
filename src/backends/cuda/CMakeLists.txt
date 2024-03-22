
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    file(GLOB backend_headers "embed/*.h")

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/cuda)
    add_custom_target(copy_cuda_headers ALL DEPENDS ${backend_headers})

    file(GLOB_RECURSE HEADER_FILES *.h*)
    file(GLOB_RECURSE SOURCE_FILES *.c*)
    find_package(OptiX REQUIRED VERSION 7)
    include_directories(${OptiX_INCLUDE})
    ocarina_add_backend(cuda SOURCES ${SOURCE_FILES})
    target_compile_definitions(ocarina-backend-cuda PRIVATE OPTIX_INCLUDE=${OptiX_INCLUDE})
    target_link_libraries(ocarina-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)

    add_custom_command(TARGET copy_cuda_headers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying cuda header files..."
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/cuda
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${backend_headers} ${CMAKE_BINARY_DIR}/bin/cuda
    )

endif ()

