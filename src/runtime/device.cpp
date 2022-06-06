//
// Created by Zero on 06/06/2022.
//


#include "device.h"

namespace katana {

Device::Device() {
    _impl = katana::make_unique<Impl>();
}
}