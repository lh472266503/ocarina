//
// Created by Zero on 06/06/2022.
//


#include "device.h"

namespace nano {

Device::Device() {
    _impl = nano::make_unique<Impl>();
}
}