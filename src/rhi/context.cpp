//
// Created by Zero on 06/06/2022.
//

#include "context.h"
#include "core/logging.h"
#include "device.h"
#include "windows/gl.h"

#ifdef NDEBUG

#include "context_impl.h"

#endif

namespace ocarina {

#ifdef _MSC_VER
static constexpr string_view backend_prefix = "ocarina-backend-";
static constexpr string_view window_lib_prefix = "ocarina-window-";
#else
static constexpr string_view window_lib_prefix = "libocarina-backend-";
static constexpr string_view window_lib_name = "libocarina-window-";
#endif

namespace detail {
[[nodiscard]] fs::path create_rhi_directory(fs::path path) noexcept {
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

[[nodiscard]] ocarina::string backend_full_name(const string &name) {
    return string(backend_prefix) + name;
}

[[nodiscard]] string window_name(const string &name) {
    return string(window_lib_prefix) + name;
}

}// namespace detail

Context::Context(const fs::path &path, string_view cache_dir)
    : _impl(std::move(ocarina::make_unique<Impl>())) {
    _impl->runtime_directory = detail::create_rhi_directory(path);
    _impl->cache_directory = runtime_directory() / cache_dir;
    DynamicModule::add_search_path(runtime_directory());
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

void Context::write_cache(const string &fn, const string &text) const noexcept {
    std::ofstream fs;
    fs.open((cache_directory() / fn).c_str());
    fs << text;
    fs.close();
}

string Context::read_cache(const string &fn) const noexcept {
    std::ifstream fst;
    fst.open((cache_directory() / fn).c_str());
    std::stringstream buffer;
    buffer << fst.rdbuf();
    return buffer.str();
}

void Context::clear_cache() const noexcept {
    if (fs::exists(_impl->cache_directory)) {
        fs::remove_all(_impl->cache_directory);
    }
}

bool Context::is_exist_cache(const string &fn) const noexcept {
    if (!_impl->use_cache) {
        return false;
    }
    auto path = cache_directory() / fn;
    return fs::exists(path);
}

const DynamicModule *Context::obtain_module(const string &module_name) noexcept {
    auto iter = _impl->modules.find(module_name);
    DynamicModule *ret = nullptr;
    if (iter == _impl->modules.cend()) {
        DynamicModule d(module_name);
        _impl->modules.insert(std::make_pair(string(module_name), std::move(d)));
        ret = &_impl->modules.at(module_name);
    } else {
        ret = &iter->second;
    }
    return ret;
}

Device Context::create_device(const string &backend_name) noexcept {
    auto d = obtain_module(dynamic_module_name(detail::backend_full_name(backend_name)));
    auto create_device = reinterpret_cast<Device::Creator *>(d->function_ptr("create"));
    auto destroy_func = reinterpret_cast<Device::Deleter *>(d->function_ptr("destroy"));
    return Device{Device::Handle{create_device(this), destroy_func}};
}

Window::Wrapper Context::create_window(const char *name, uint2 initial_size, const char *type, bool resizable) {
    auto d = obtain_module(dynamic_module_name(detail::window_name(type)));
    auto create_window = reinterpret_cast<Window::Creator *>(d->function_ptr("create"));
    auto destroy_func = reinterpret_cast<Window::Deleter *>(d->function_ptr("destroy"));
    return Window::Wrapper(create_window(name, initial_size, resizable), destroy_func);
}

}// namespace ocarina