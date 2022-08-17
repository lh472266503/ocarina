
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    file(GLOB_RECURSE HEADER_FILES *.h*)
    file(GLOB_RECURSE SOURCE_FILES *.c*)
    find_package(OptiX REQUIRED VERSION 7)
    include_directories(${OptiX_INCLUDE})
    ocarina_add_backend(cuda SOURCES ${SOURCE_FILES})
    target_compile_definitions(ocarina-backend-cuda PRIVATE OPTIX_INCLUDE=${OptiX_INCLUDE})
    target_link_libraries(ocarina-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)

endif ()

