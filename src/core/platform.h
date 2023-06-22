//
// Created by Zero on 09/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace ocarina {
[[nodiscard]] OC_CORE_API void *dynamic_module_load(const fs::path &path) noexcept;
void dynamic_module_destroy(void *handle) noexcept;
[[nodiscard]] OC_CORE_API void *dynamic_module_find_symbol(void *handle,
                                                           ocarina::string_view name) noexcept;
[[nodiscard]] OC_CORE_API ocarina::string dynamic_module_name(ocarina::string_view name) noexcept;


struct TraceItem {
    string module;
    uint64_t address;
    string symbol;
    size_t offset;
};

[[nodiscard]] OC_NEVER_INLINE vector<TraceItem> backtrace() noexcept;

[[nodiscard]] OC_NEVER_INLINE string backtrace_string() noexcept;

}// namespace ocarina