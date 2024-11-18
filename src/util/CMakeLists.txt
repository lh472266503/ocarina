file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-util SHARED ${SOURCE_FILES})
set_target_properties(ocarina-util PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        FOLDER "ocarina")
target_compile_definitions(ocarina-util PRIVATE OC_UTIL_EXPORT_DLL=1)
target_link_libraries(ocarina-util PUBLIC ocarina-ext ocarina-core)