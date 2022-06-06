//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "device.h"

namespace nano {
class Context final : public concepts::Noncopyable{
private:
    struct Impl;
    nano::unique_ptr<Impl> _impl;
public:
    explicit Context(const fs::path &program) noexcept;
    ~Context() noexcept;
    void load_module_function(const fs::path &path, nano::string_view module_name);
    [[nodiscard]] const fs::path &runtime_directory() const noexcept;
    [[nodiscard]] const fs::path &cache_directory() const noexcept;
    [[nodiscard]] Device create_device(nano::string_view backend_name,
                                       nano::string_view property = "{}") noexcept;
};
}// namespace nano