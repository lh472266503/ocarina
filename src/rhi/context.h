//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "window.h"

namespace ocarina {
class Device;
class DynamicModule;
class Context : public concepts::Noncopyable {
private:
    struct Impl;
    ocarina::unique_ptr<Impl> _impl;

public:
    explicit Context(const fs::path &path, string_view cache_dir = ".cache");
    virtual ~Context() noexcept;
    [[nodiscard]] const fs::path &runtime_directory() const noexcept;
    [[nodiscard]] const fs::path &cache_directory() const noexcept;
    void write_cache(const string &fn, const string &text) const noexcept;
    [[nodiscard]] string read_cache(const string &fn) const noexcept;
    void clear_cache() const noexcept;
    [[nodiscard]] bool is_exist_cache(const string &fn) const noexcept;
    const DynamicModule *obtain_module(const string &module_name) noexcept;
    [[nodiscard]] Device create_device(const string &backend_name) noexcept;
    [[nodiscard]] Window::Handle create_window(const char *name, uint2 initial_size, const char *type = "gl", bool resizable = false);
};

}// namespace ocarina

#ifndef NDEBUG

#include "context_impl.h"

#endif