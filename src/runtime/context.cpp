//
// Created by Zero on 06/06/2022.
//

#include "context.h"
#include "core/logging.h"
#include "dynamic_module.h"
#include "device.h"

namespace ocarina {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
    ocarina::unique_ptr<Device> device;
    ocarina::vector<DynamicModule> loaded_modules;
};

namespace detail {
[[nodiscard]] fs::path create_runtime_directory(fs::path path) noexcept {
    path = fs::canonical(path);
    if (fs::is_directory(path)) {
        return path;
    }
    return fs::canonical(path.parent_path());
}

bool create_directory_if_necessary(const fs::path &path) {
    if (fs::exists(path)) {
        return false;
    }
    try {
        fs::create_directory(path);
        OC_INFO("create folder ", path.string());
    } catch (fs::filesystem_error &e) {
        OC_WARNING("Failed to create folder ", path, ", reason: ", e.what());
    }
    return true;
}

}// namespace detail

Context::Context(const fs::path &path, string_view cache_dir) noexcept
    : _impl(std::move(ocarina::make_unique<Impl>())) {
    _impl->runtime_directory = detail::create_runtime_directory(path);
    _impl->cache_directory = runtime_directory() / cache_dir;
    detail::create_directory_if_necessary(cache_directory());
}

Context::~Context() noexcept {
    OC_INFO("context was destructed !");
}

const fs::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}

const fs::path &Context::cache_directory() const noexcept {
    return _impl->cache_directory;
}

void Context::load_module_function(const fs::path &path, ocarina::string_view module_name) {

}

void Context::init_device(ocarina::string_view backend_name) noexcept {

}

Device *Context::device() noexcept {
    return _impl->device.get();
}

const Device *Context::device() const noexcept {
    return _impl->device.get();
}

}// namespace ocarina