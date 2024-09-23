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

enum AccessMode {
    AOS,
    SOA
};

template<typename T, AccessMode mode = AOS>
class Stack : public ByteBuffer {
public:
    using element_type = T;
    static constexpr AccessMode access_mode = mode;
    using Super = ByteBuffer;

private:
    string name_;
    uint capacity_;

public:
    explicit Stack(uint size, string name = "stack")
        : capacity_(size), name_(std::move(name)) {}
    OC_MAKE_MEMBER_GETTER(capacity, )
    [[nodiscard]] Super &super() noexcept {
        return *this;
    }
    void init(Device &device) noexcept {
        super() = device.create_byte_buffer(sizeof(T) * capacity_ + sizeof(capacity_), name_);
    }
};

}// namespace ocarina