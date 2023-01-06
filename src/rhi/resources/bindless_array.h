//
// Created by Zero on 03/11/2022.
//

#pragma once

#include "resource.h"
#include "dsl/type_trait.h"
#include "rhi/command.h"
#include "rhi/device.h"

namespace ocarina {

class BindlessArray : public RHIResource {
public:


    class Impl {
    public:
        [[nodiscard]] virtual size_t size() const noexcept = 0;
        [[nodiscard]] virtual size_t alignment() const noexcept = 0;
    };

private:
    size_t _slot_num{65536};

public:
    explicit BindlessArray(Device::Impl *device);
};

}// namespace ocarina