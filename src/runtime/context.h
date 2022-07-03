//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"

namespace ocarina {
class Device;
class DynamicModule;
class Context final : public concepts::Noncopyable{
private:
    struct Impl;
    ocarina::unique_ptr<Impl> _impl;
public:
    explicit Context(const fs::path &path, string_view cache_dir = ".cache");
    ~Context() noexcept;
    [[nodiscard]] const fs::path &runtime_directory() const noexcept;
    [[nodiscard]] const fs::path &cache_directory() const noexcept;
    const DynamicModule *obtain_module(const string& module_name) noexcept;
    void init_device(const ocarina::string& backend_name) noexcept;
    [[nodiscard]] const Device *device() const noexcept;
    [[nodiscard]] Device *device() noexcept;
};

}// namespace ocarina

#ifndef NDEBUG

#include "context_impl.h"

#endif