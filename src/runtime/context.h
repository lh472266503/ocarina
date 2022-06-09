//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"

namespace ocarina {
class Device;
class Context final : public concepts::Noncopyable{
private:
    struct Impl;
    ocarina::unique_ptr<Impl> _impl;

public:
    explicit Context(const fs::path &path, string_view cache_dir = ".cache");
    ~Context() noexcept;
    void load_module(const fs::path &path, ocarina::string_view module_name);
    [[nodiscard]] const fs::path &runtime_directory() const noexcept;
    [[nodiscard]] const fs::path &cache_directory() const noexcept;
    void init_device(ocarina::string_view backend_name) noexcept;
    [[nodiscard]] const Device *device() const noexcept;
    [[nodiscard]] Device *device() noexcept;
};
}// namespace ocarina