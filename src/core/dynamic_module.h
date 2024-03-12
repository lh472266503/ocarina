//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"

namespace ocarina {
class DynamicModule : public concepts::Noncopyable {
private:
    void *_handle{};
    static ocarina::vector<fs::path> &_search_path();

public:
    static void add_search_path(fs::path path) noexcept;
    static void remove_search_path(fs::path path) noexcept;
    static void clear_search_path() noexcept;
    explicit DynamicModule(const string &name) noexcept;
    OC_MAKE_MEMBER_GETTER(handle, )
    DynamicModule(fs::path path, const string &name) noexcept;
    [[nodiscard]] void *function_ptr(const string &func_name) const noexcept;
};
}// namespace ocarina