//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "GUI/widgets.h"

namespace ocarina {

class ImGuiWidgets : public Widgets {
public:
    void init() noexcept override;
    void push_window(std::string_view label) noexcept override;
    void pop_window() noexcept override;

    void text(std::string_view format, ...) noexcept override;
};

}// namespace ocarina