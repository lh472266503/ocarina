//
// Created by Zero on 10/07/2022.
//

#pragma once

#include "core/pool.h"
#include "graphics_descriptions.h"

namespace ocarina {
class Command;
class CommandVisitor;

class OC_RHI_API CommandList : public ocarina::vector<Command *> {
public:
    using Super = ocarina::vector<Command *>;
    using Super::Super;
    CommandList &operator<<(Command *command) noexcept;
    CommandList &operator<<(const vector<Command *> &commands) noexcept;
    CommandList &operator<<(std::function<void()> func) noexcept;
    void accept(CommandVisitor &visitor) const noexcept;
    void recycle() noexcept;
};

class OC_RHI_API CommandQueue {

private:
    CommandList commands_{};
    QueueType queueType_;

public:
    CommandQueue() = default;
    CommandQueue(QueueType queueType) : queueType_(queueType) {}
    ~CommandQueue() { recycle(); }
    template<typename... Args>
    auto push_back(Args &&...args) {
        return commands_.push_back(OC_FORWARD(args)...);
    }
    template<typename... Args>
    auto emplace_back(Args &&...args) {
        return commands_.emplace_back(OC_FORWARD(args)...);
    }
    [[nodiscard]] auto begin() const noexcept { return commands_.begin(); }
    [[nodiscard]] auto begin() noexcept { return commands_.begin(); }
    [[nodiscard]] auto end() const noexcept { return commands_.end(); }
    [[nodiscard]] auto end() noexcept { return commands_.end(); }
    void pop_back();
    void clear() noexcept;
    void recycle() noexcept;
};

}// namespace ocarina