//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/util.h"

namespace ocarina {

class Window;

using WindowCreator = Window *(const char *name, uint2 initial_size, bool resizable);
using WindowDeleter = void(Window *);
using WindowWrapper = ocarina::unique_ptr<Window, WindowDeleter *>;

}// namespace ocarina