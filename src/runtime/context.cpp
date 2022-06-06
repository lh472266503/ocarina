//
// Created by Zero on 06/06/2022.
//

#include "context.h"

namespace katana {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
};
Context::Context(const fs::path &program) noexcept {
}
}