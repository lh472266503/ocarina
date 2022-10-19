file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-windows INTERFACE)

function(ocarina_add_window_plugin name)

    cmake_parse_arguments(PLUGIN "" "CATEGORY" "SOURCES" ${ARGN})
    set(lib_name ocarina-${PLUGIN_CATEGORY}-${name})
    add_library(${lib_name} MODULE ${PLUGIN_SOURCES})
    target_link_libraries(${lib_name} PRIVATE ocarina-core)
    set(category ocarina-${PLUGIN_CATEGORY}s)
    add_dependencies(${category} ${lib_name})

    set_target_properties(${lib_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            UNITY_BUILD OFF
            DEBUG_POSTFIX "")
endfunction()

ocarina_add_window_plugin(gl CATEGORY window SOURCES window.cpp)

#add_library(ocarina-gui MODULE ${SOURCE_FILES})
#set_target_properties(ocarina-gui PROPERTIES
#        UNITY_BUILD OFF
#        PUBLIC_HEADER "${HEADER_FILES}"
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
#        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#        WINDOWS_EXPORT_ALL_SYMBOLS ON)
#
#
#target_compile_definitions(ocarina-gui PRIVATE OC_GUI_EXPORT_DLL=1)
#target_link_libraries(ocarina-gui PRIVATE ocarina-core ocarina-gui-ext)