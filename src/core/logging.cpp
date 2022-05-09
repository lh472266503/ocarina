//
// Created by Zero on 24/04/2022.
//

#include "logging.h"

namespace katana {
inline namespace core {
spdlog::logger &logger() noexcept {
    static auto ret = spdlog::stdout_color_mt("console");
    return *ret;
}
}
}// namespace katana::core