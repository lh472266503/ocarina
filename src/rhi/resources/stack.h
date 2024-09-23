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
    static_assert(is_valid_buffer_element_v<T>);
    using element_type = T;
    static constexpr uint stride = sizeof(T);
    static constexpr AccessMode access_mode = mode;
    using Super = ByteBuffer;

public:
    explicit Stack(Device::Impl *device, uint size, string name = "stack")
        : ByteBuffer(device, size * sizeof(T) + sizeof(uint), name) {}
    [[nodiscard]] Super &super() noexcept { return *this; }
    [[nodiscard]] uint capacity() const noexcept {
        return (super().size() - sizeof(uint)) / stride;
    }

    [[nodiscard]] BufferUploadCommand *clear(bool async = true) noexcept {
        static uint val = 0;
        auto cmd = view(size() - sizeof(uint), sizeof(uint)).upload(&val, async);
        return cmd;
    }

    [[nodiscard]] uint host_count() const noexcept {
        uint val = 0;
        auto cmd = view(size() - sizeof(uint), sizeof(uint)).download(&val, false);
        cmd->accept(*device_->command_visitor());
        return val;
    }

    void clear_immediately() noexcept {
        clear(false)->accept(*device_->command_visitor());
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> count() const noexcept {
        const auto expr = make_expr<ByteBuffer>(expression());
        return load_as<Size>(expr.size() - sizeof(uint));
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> &count() noexcept {
        auto expr = make_expr<ByteBuffer>(expression());
        return load_as<Size>(expr.size() - sizeof(uint));
    }

    template<typename Arg, typename Size = uint>
    requires std::is_same_v<T, remove_device_t<Arg>>
    void push_back(const Arg &arg) noexcept {
        auto expr = make_expr<ByteBuffer>(expression());
        Var<Size> index = atomic_add(count(), 1);
        store(index * stride, arg);
    }

    template<typename Index, typename Size = uint>
    [[nodiscard]] Var<T> at(const Index &index) const noexcept {
        Var<Size> offset = index * sizeof(T);
        return load_as<T>(offset);
    }

    template<typename Index, typename Size = uint>
    [[nodiscard]] Var<T> &at(const Index &index) noexcept {
        Var<Size> offset = index * sizeof(T);
        return load_as<T>(offset);
    }
};

}// namespace ocarina