//
// Created by Zero on 10/07/2022.
//

#pragma once

#include "core/pool.h"

namespace ocarina {
class Command;

class CommandQueue {

private:
    ocarina::vector<Command *> _commands{};

public:
    CommandQueue() = default;
    ~CommandQueue() { recycle(); }
    template<typename... Args>
    auto push_back(Args &&...args) {
        return _commands.push_back(OC_FORWARD(args)...);
    }
    template<typename... Args>
    auto emplace_back(Args &&...args) {
        return _commands.emplace_back(OC_FORWARD(args)...);
    }
    [[nodiscard]] auto begin() const noexcept { return _commands.begin(); }
    [[nodiscard]] auto begin() noexcept { return _commands.begin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.end(); }
    [[nodiscard]] auto end() noexcept { return _commands.end(); }
    void pop_back();
    void clear() noexcept;
    void recycle() noexcept;
};

}// namespace ocarina