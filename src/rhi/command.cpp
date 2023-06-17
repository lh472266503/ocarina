//
// Created by Zero on 10/07/2022.
//

#include "command.h"

#include <utility>
#include "resources/shader.h"

namespace ocarina {

OC_COMMAND_POOL_FUNCTIONS_IMPL

ShaderDispatchCommand::ShaderDispatchCommand(handle_ty entry, SP<ArgumentList> argument_list, uint3 dim)
    : Command(true), _entry(entry),
      _argument_list(ocarina::move(argument_list)), _dispatch_dim(dim) {
}

span<void *> ShaderDispatchCommand::args() noexcept {
    return _argument_list->ptr();
}

span<const MemoryBlock> ShaderDispatchCommand::params() noexcept {
    return _argument_list->params();
}

size_t ShaderDispatchCommand::params_size() noexcept {
    return structure_size(params());
}

}// namespace ocarina