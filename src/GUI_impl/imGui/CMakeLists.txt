file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

ocarina_add_window_plugin(imGui CATEGORY GUI_impl SOURCES ${SOURCE_FILES})
