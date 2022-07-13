//
// Created by Zero on 06/07/2022.
//

#include "stream.h"
#include "device.h"

namespace ocarina {
Stream::Stream(Device::Impl *device)
    : Resource(device, Tag::STREAM,
               device->create_stream()) {}

Stream &Stream::operator<<(Command *command) noexcept {
    impl()->add_command(command);
    return *this;
}

void Stream::commit(const Commit &commit) noexcept {
    impl()->commit(commit);
}

Stream &Stream::operator<<(const Commit &cmt) noexcept {
    commit(cmt);
    return *this;
}
}// namespace ocarina