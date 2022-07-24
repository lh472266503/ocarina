//
// Created by Zero on 10/07/2022.
//

#include "command_queue.h"
#include "command.h"

namespace ocarina {

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