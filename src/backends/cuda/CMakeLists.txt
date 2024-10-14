
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)

    file(GLOB backend_headers "builtin/*.h")

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/cuda)
    add_custom_target(copy_cuda_headers ALL DEPENDS ${backend_headers})
    include_directories(${CUDA_INCLUDE_DIRS})
    file(GLOB_RECURSE HEADER_FILES *.h*)
    file(GLOB_RECURSE SOURCE_FILES *.c*)
    find_package(OptiX REQUIRED VERSION 7)
    include_directories(${OptiX_INCLUDE})
    ocarina_add_backend(cuda SOURCES ${HEADER_FILES} ${SOURCE_FILES})
    target_compile_definitions(ocarina-backend-cuda PRIVATE OPTIX_INCLUDE=${OptiX_INCLUDE})
    target_link_libraries(ocarina-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)

    if (CMAKE_GENERATOR MATCHES "Visual Studio")
    add_custom_command(TARGET copy_cuda_headers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Current IDE is ${CMAKE_GENERATOR}, current Config is ${CMAKE_CFG_INTDIR}"
            COMMAND ${CMAKE_COMMAND} -E echo "Copying cuda header files to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}${CMAKE_CFG_INTDIR}"
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/cuda
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${backend_headers} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/cuda
            COMMAND ${CMAKE_COMMAND} -E echo "Complete copy!")
    else()
    add_custom_command(TARGET copy_cuda_headers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying cuda header files to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${backend_headers} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda
            COMMAND ${CMAKE_COMMAND} -E echo "Complete copy!"
    )
    endif()
endif ()

