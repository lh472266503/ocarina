//
// Created by Zero on 03/11/2022.
//

#pragma once

#include "resource.h"
#include "dsl/type_trait.h"
#include "rhi/command.h"

namespace ocarina {

class BindlessArray : public RHIResource {
private:
    size_t _slot_num{65536};

private:
    friend class Device;
    void _emplace_buffer(size_t index, uint64_t handle, size_t offset_bytes) noexcept;

public:
    BindlessArray(Device::Impl *device, size_t size) noexcept;

    template<typename T>
    requires is_buffer_or_view_v<T>
    void emplace(uint index, T &&t) noexcept {

    }
};

}