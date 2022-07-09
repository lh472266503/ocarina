//
// Created by Zero on 06/07/2022.
//

#include "stream.h"
#include "device.h"
namespace ocarina {
Stream::Stream(Device::Impl *device) : Resource(device, Tag::STREAM,
                                                device->create_stream()) {}
}// namespace ocarina