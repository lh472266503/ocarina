//
// Created by Zero on 06/06/2022.
//

#include "context.h"
#include "core/logging.h"
#include "dynamic_module.h"

namespace ocarina {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
    ocarina::vector<DynamicModule> loaded_modules;
};

Context::Context(const fs::path &program) noexcept
    : _impl(std::move(ocarina::make_unique<Impl>())) {
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
}// namespace ocarina