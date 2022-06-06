//
// Created by Zero on 06/06/2022.
//

#include "context.h"
#include "core/logging.h"
#include "dynamic_module.h"

namespace nano {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
    nano::vector<DynamicModule> loaded_modules;
};

Context::Context(const fs::path &program) noexcept
    : _impl(std::move(nano::make_unique<Impl>())) {
}

Context::~Context() noexcept {
    NN_INFO("context was destructed !");
}
const fs::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}
const fs::path &Context::cache_directory() const noexcept {
    return _impl->cache_directory;
}
void Context::load_module_function(const fs::path &path, nano::string_view module_name) {
}
}// namespace nano