//
// Created by Zero on 06/07/2022.
//

#pragma once

#include "resource.h"
#include "device.h"
namespace ocarina {

class Stream : public Resource {
private:

public:
    Stream(Device::Impl *device, handle_ty handle)
        : Resource(device, Tag::STREAM, handle) {}

};

}// namespace ocarina