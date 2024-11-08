
#include "glfw_window.h"
#include "sdl_window.h"

namespace ocarina {
OC_EXPORT_API ocarina::Window *create(const char *name, ocarina::uint2 initial_size, WindowLibrary library, bool resizable) {
    if (library == WindowLibrary::GLFW)
        return ocarina::new_with_allocator<ocarina::GLWindow>(name, initial_size, resizable);
    else
        return ocarina::new_with_allocator<ocarina::SDLWindow>(name, initial_size, resizable);
}

OC_EXPORT_API void destroy(ocarina::Window *ptr) {
    ocarina::delete_with_allocator(ptr);
}

}// namespace ocarina