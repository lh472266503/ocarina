//
// Created by Zero on 09/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace ocarina {
[[nodiscard]] void *dynamic_module_load(const fs::path &path) noexcept;
void dynamic_module_destroy(void *handle) noexcept;
}// namespace ocarina