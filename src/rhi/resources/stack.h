//
// Created by Zero on 2024/9/23.
//

#include <ast/function.h>
#include "rhi/resources/managed.h"
#include "rhi/resources/byte_buffer.h"
#include <dsl/var.h>
#include <dsl/builtin.h>
#include <dsl/operators.h>
#include "rhi/device.h"

namespace ocarina {

template<typename T, AccessMode mode>
class Stack : public ByteBuffer {
public:
    using element_type = T;
    static constexpr AccessMode access_mode = mode;
    using Super = ByteBuffer;

public:
    explicit Stack(Device::Impl *device, uint size, string name = "stack")
        : ByteBuffer(device, size * sizeof(T) + sizeof(uint), name) {}
    [[nodiscard]] Super &super() noexcept { return *this; }
    [[nodiscard]] uint capacity() const noexcept {
        return (super().size() - sizeof(uint)) / sizeof(T);
    }
};

}// namespace ocarina