//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"

namespace katana {
class DynamicModule : public concepts::Noncopyable{
public:
    using handle_type = void *;

private:
    handle_type _handle{};

public:
};
}// namespace katana