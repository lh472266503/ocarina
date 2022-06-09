//
// Created by Zero on 06/06/2022.
//

#include "context.h"
#include "core/logging.h"
#include "dynamic_module.h"
#include "device.h"
#include "core/platform.h"

namespace ocarina {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
    ocarina::unique_ptr<Device> device;
    ocarina::map<string, DynamicModule> modules;
};

static constexpr string_view backend_prefix = "ocarina-backend-";

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

[[nodiscard]] ocarina::string backend_full_name(const string &name) {
    return string(backend_prefix) + name;
}

}// namespace detail

Context::Context(const fs::path &path, string_view cache_dir)
    : _impl(std::move(ocarina::make_unique<Impl>())) {
    _impl->runtime_directory = detail::create_runtime_directory(path);
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

const DynamicModule *Context::obtain_module(const string& module_name) noexcept {
    auto iter = _impl->modules.find(module_name);
    DynamicModule* ret = nullptr;
    if (iter == _impl->modules.cend()) {
        DynamicModule d(module_name);
        _impl->modules.insert(std::make_pair(string(module_name), std::move(d)));
        ret = &_impl->modules.at(module_name);
    } else {
        ret = &iter->second;
    }
    return ret;
}

void Context::init_device(const ocarina::string &backend_name) noexcept {
    auto d = obtain_module(dynamic_module_name(detail::backend_full_name(backend_name)));
    auto create_device = reinterpret_cast<Device::Creator*>(d->function_ptr("create"));
    _impl->device.reset(create_device(this));
}

Device *Context::device() noexcept {
    return _impl->device.get();
}

const Device *Context::device() const noexcept {
    return _impl->device.get();
}

}// namespace ocarina