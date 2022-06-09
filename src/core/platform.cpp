//
// Created by Zero on 09/06/2022.
//

#include "platform.h"
#include <windows.h>
#include <dbghelp.h>
#include "fmt/format.h"
#include "core/logging.h"

namespace ocarina {

namespace detail {
[[nodiscard]] ocarina::string win32_last_error_message() {
    void *buffer = nullptr;
    auto err_code = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&buffer,
        0, nullptr);
    ocarina::string err_msg{fmt::format("{} (code = 0x{:x}).", static_cast<char *>(buffer), err_code)};
    LocalFree(buffer);
    return err_msg;
}
}// namespace detail

void *dynamic_module_load(const fs::path &path) noexcept {
    auto path_string = path.string();
    auto module = LoadLibraryA(path_string.c_str());
    if (module == nullptr) [[unlikely]] {
        OC_ERROR(
            "Failed to load dynamic module '{}', reason: {}.",
            path_string, detail::win32_last_error_message());
    }
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) {
        FreeLibrary(reinterpret_cast<HMODULE>(handle));
    }
}

void *dynamic_module_find_symbol(void *handle, ocarina::string_view name_view) noexcept {
    static thread_local ocarina::string name;
    name = name_view;
    OC_INFO_FORMAT("Loading dynamic symbol: {}.", name);
    auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str());
    if (symbol == nullptr) [[unlikely]] {
        OC_INFO_FORMAT("Failed to load symbol '{}', reason: {}.",
                       name, detail::win32_last_error_message());
    }
    return reinterpret_cast<void *>(symbol);
}

}// namespace ocarina