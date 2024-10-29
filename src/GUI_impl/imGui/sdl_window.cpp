//
// Created by Zero on 2022/8/16.
//

#include "sdl_window.h"
#include "widgets.h"
#include "ext/imgui/gizmo/ImGuizmo.h"
#include "core/logging.h"
#include "rhi/common.h"
//#include <ext/SDL/include/SDL3/SDL_syswm.h>

namespace ocarina {


void SDLWindow::init_widgets() noexcept {
    widgets_ = make_unique<ImGuiWidgets>(this);
}

void SDLWindow::init(const char *name, uint2 initial_size, bool resizable) noexcept {
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
    flags |= resizable ? SDL_WINDOW_RESIZABLE : 0;
    handle_ = SDL_CreateWindow(name, initial_size.x, initial_size.y, flags);
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(handle_), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    window_handle_ = (uint64_t)hwnd;

}

SDLWindow::SDLWindow(const char *name, uint2 initial_size, bool resizable) noexcept
    : Window(resizable) {
    init(name, initial_size, resizable);
}

SDLWindow::~SDLWindow() noexcept {
    //glfwMakeContextCurrent(handle_);
    //texture_.reset();
    //widgets_.reset();
    //ImGui_ImplOpenGL3_Shutdown();
    //ImGui_ImplGlfw_Shutdown();
    //ImGui::DestroyContext();
    SDL_DestroyWindow(handle_);
}

uint2 SDLWindow::size() const noexcept {
    auto width = 0;
    auto height = 0;
    //glfwGetWindowSize(handle_, &width, &height);
    SDL_GetWindowSize(handle_, &width, &height);
    return make_uint2(
        static_cast<uint>(width),
        static_cast<uint>(height));
}

bool SDLWindow::should_close() const noexcept {
    return false;//SDL_ShouldMinimizeOnFocusLoss(handle_);//glfwWindowShouldClose(handle_);
}

void SDLWindow::set_background(const uchar4 *pixels, uint2 size) noexcept {
    //if (texture_ == nullptr) { texture_ = ocarina::make_unique<GLTexture>(); }
    //texture_->load(pixels, size);
}

void SDLWindow::set_background(const float4 *pixels, uint2 size) noexcept {
    //if (texture_ == nullptr) {
    //    texture_ = ocarina::make_unique<GLTexture>();
    //}
    //texture_->load(pixels, size);
}

void SDLWindow::gen_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept {
    //CHECK_GL(glGenBuffers(1, addressof(handle)));
    //CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, handle));
    //CHECK_GL(glBufferData(GL_ARRAY_BUFFER, size_in_byte,
    //                      nullptr, GL_STREAM_DRAW));
    //CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0u));
}

void SDLWindow::bind_buffer(ocarina::uint &handle, ocarina::uint size_in_byte) const noexcept {
    //if (handle == 0) {
    //    gen_buffer(handle, size_in_byte);
    //}
    //CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, handle));
}

void SDLWindow::set_background(const Buffer<ocarina::float4> &buffer, ocarina::uint2 size) noexcept {
    //if (texture_ == nullptr) {
    //    texture_ = ocarina::make_unique<GLTexture>();
    //}
    //texture_->bind();
    //bind_buffer(buffer.gl_handle(), buffer.size_in_byte());
}

void SDLWindow::unbind_buffer(ocarina::uint &handle) const noexcept {
    CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void SDLWindow::set_should_close() noexcept {
    //glfwSetWindowShouldClose(handle_, true);
}

void SDLWindow::_begin_frame() noexcept {
    if (!should_close()) {
        //glfwMakeContextCurrent(handle_);
        //glfwPollEvents();
        //ImGui_ImplOpenGL3_NewFrame();
        //ImGui_ImplGlfw_NewFrame();
        //ImGui::NewFrame();
        //ImGuizmo::BeginFrame();
        Window::_begin_frame();
    }
}

void SDLWindow::_end_frame() noexcept {
    if (!should_close()) {
        Window::_end_frame();
        // background
        //if (texture_ != nullptr) {
        //    ImVec2 background_size{
        //        static_cast<float>(texture_->size().x),
        //        static_cast<float>(texture_->size().y)};
        //    ImGui::GetBackgroundDrawList()->AddImage(
        //        reinterpret_cast<ImTextureID>(static_cast<uint64_t>(texture_->handle())), {}, background_size);
        //}
        //// rendering
        //ImGui::Render();
        //int display_w, display_h;
        //glfwGetFramebufferSize(handle_, &display_w, &display_h);
        //glViewport(0, 0, display_w, display_h);
        //glClearColor(clear_color_.x, clear_color_.y, clear_color_.z, clear_color_.w);
        //glClear(GL_COLOR_BUFFER_BIT);
        //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        //glfwSwapBuffers(handle_);
    }
}

void SDLWindow::set_size(uint2 size) noexcept {
    if (resizable_) {
        SDL_SetWindowSize(handle_, static_cast<int>(size.x), static_cast<int>(size.y));
    } else {
        OC_WARNING("Ignoring resize on non-resizable window.");
    }
}

}// namespace ocarina

OC_EXPORT_API ocarina::SDLWindow *create_sdlwindow(const char *name, ocarina::uint2 initial_size, bool resizable) {
    return ocarina::new_with_allocator<ocarina::SDLWindow>(name, initial_size, resizable);
}

OC_EXPORT_API void destroy_sdlwindow(ocarina::SDLWindow *ptr) {
    ocarina::delete_with_allocator(ptr);
}