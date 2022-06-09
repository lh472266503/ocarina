
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    file(GLOB_RECURSE HEADER_FILES *.h*)
    file(GLOB_RECURSE SOURCE_FILES *.c*)

#    add_library(ocarina-backend-cuda SHARED ${SOURCE_FILES})
#    set_target_properties(ocarina-backend-cuda PROPERTIES
#            PUBLIC_HEADER "${HEADER_FILES}"
#            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
#            WINDOWS_EXPORT_ALL_SYMBOLS ON)
#    target_compile_definitions(ocarina-backend-cuda PRIVATE OC_BACKENDS_EXPORT_DLL=1)
#    target_link_libraries(ocarina-backend-cuda PUBLIC ocarina-core)

    ocarina_add_backend(cuda SOURCES ${SOURCE_FILES})
    target_link_libraries(ocarina-backend-cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif ()

