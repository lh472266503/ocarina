//
// Created by Zero on 10/07/2022.
//

#include "command_queue.h"
#include "command.h"

namespace ocarina {

CommandList &CommandList::operator<<(ocarina::Command *command) noexcept {
    push_back(command);
    return *this;
}

CommandList &CommandList::operator<<(const vector<Command *> &commands) noexcept {
    append(*this, commands);
    return *this;
}

void CommandQueue::recycle() noexcept {
    for (Command *command : _commands) {
        command->recycle();
    }
}

void CommandQueue::pop_back() {
    _commands.pop_back();
}

void CommandQueue::clear() noexcept{
    recycle();
    _commands.clear();
}

}// namespace ocarina