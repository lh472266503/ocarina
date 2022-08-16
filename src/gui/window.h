//
// Created by Zero on 2022/8/16.
//

#pragma once

//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
#include "core/stl.h"
#include "core/basic_types.h"

namespace ocarina {
class Window {
public:
    using MouseButtonCallback = ocarina::function<void(int /* button */, int /* action */, float2 /* (x, y) */)>;
    using CursorPositionCallback = ocarina::function<void(float2 /* (x, y) */)>;
    using WindowSizeCallback = ocarina::function<void(uint2 /* (width, height) */)>;
    using KeyCallback = ocarina::function<void(int /* key */, int /* action */)>;
    using ScrollCallback = ocarina::function<void(float2 /* (dx, dy) */)>;


};
}// namespace ocarina