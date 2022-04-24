file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)


add_library(sycamore-ast SHARED ${SOURCE_FILES})
target_compile_definitions(sycamore-ast PRIVATE SCM_AST_EXPORT_DLL=1)
set_target_properties(sycamore-ast PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)