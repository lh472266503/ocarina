//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "GUI/widgets.h"

namespace ocarina {

class ImGuiWidgets : public Widgets {
public:
    void init() noexcept override;
    void push_window(const char *label) noexcept override;
    void pop_window() noexcept override;

    void text(const char *format, ...) noexcept override;
};

}// namespace ocarina