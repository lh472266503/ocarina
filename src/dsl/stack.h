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
class Stack {
public:
    using element_type = T;
    static constexpr AccessMode access_mode = mode;

private:
    string name_;
    uint size_;

    /// store elements, last four elements store count
    ByteBuffer byte_buffer_;

public:
    explicit Stack(uint size, string name = "stack")
        : size_(size), name_(std::move(name)) {}
    OC_MAKE_MEMBER_GETTER(size, )
    void init(Device &device) noexcept {
        byte_buffer_ = device.create_byte_buffer(sizeof(T) * size_ + sizeof(size_), name_);
    }
    
};

}// namespace ocarina