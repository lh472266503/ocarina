//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"

namespace ocarina {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

}// namespace ocarina