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

void Stream::commit() noexcept {
    impl()->commit();
}

Stream &Stream::operator<<(Commit) noexcept {
    commit();
    return *this;
}
}// namespace ocarina