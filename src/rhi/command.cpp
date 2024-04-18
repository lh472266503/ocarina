//
// Created by Zero on 10/07/2022.
//

#include "command.h"

#include <utility>
#include "resources/shader.h"

namespace ocarina {

OC_COMMAND_POOL_FUNCTIONS_IMPL

ShaderDispatchCommand::ShaderDispatchCommand(handle_ty entry, SP<ArgumentList> argument_list, uint3 dim)
    : Command(true), entry_(entry),
      argument_list_(ocarina::move(argument_list)), dispatch_dim_(dim) {
}

span<void *> ShaderDispatchCommand::args() noexcept {
    return argument_list_->ptr();
}

span<const std::byte> ShaderDispatchCommand::argument_data() noexcept {
    return argument_list_->argument_data();
}

size_t ShaderDispatchCommand::params_size() noexcept {
    return structure_size(argument_list_->blocks());
}

}// namespace ocarina