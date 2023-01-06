//
// Created by Zero on 12/01/2023.
//

#include "bindless_array.h"

namespace ocarina {

BindlessArray::BindlessArray(Device::Impl *device)
    : RHIResource(device, Tag::BINDLESS_ARRAY,
                  device->create_bindless_array()) {}

}// namespace ocarina