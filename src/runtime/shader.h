//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "device.h"
#include "buffer.h"
#include "stream.h"
#include "command.h"
#include "core/concepts.h"
#include "core/util.h"

namespace ocarina {

class ArgumentList {
private:
    static constexpr auto Size = 200;
    ocarina::vector<void *> _args;
    ocarina::array<std::byte, Size> _argument_data{};
    size_t _cursor{};

private:
    template<typename T>
    requires concepts::basic<T>
    void _encode_basic(T &&arg) noexcept {
        _cursor = mem_offset(_cursor, alignof(T));
        auto dst_ptr = _argument_data.data() + _cursor;
        _cursor += sizeof(T);
        OC_ASSERT(_cursor < Size);
        std::memcpy(dst_ptr, &arg, sizeof(T));
        _args.push_back(dst_ptr);
    }

    template<typename T>
    void _encode_buffer(Buffer<T> &buffer) noexcept {
        handle_ty handle = buffer.handle();
        _cursor = mem_offset(_cursor, alignof(handle_ty));
//        auto dst_ptr = _argument_data.data() + _cursor;
//        _cursor += sizeof(handle_ty);
//        OC_ASSERT(_cursor < Size);
//        std::memcpy(dst_ptr, &handle, sizeof(handle_ty));
//        cout << *reinterpret_cast<handle_ty*>(dst_ptr) << "----------\n";
        _args.push_back(buffer.handle_address());
        int i = 0;
    }

public:
    ArgumentList() = default;
    [[nodiscard]] span<void *> ptr() noexcept { return _args; }
    [[nodiscard]] size_t num() const noexcept { return _args.size(); }

    template<typename T>
    ArgumentList &operator<<(T &&arg) {
        if constexpr (concepts::basic<T>) {
            _encode_basic(OC_FORWARD(arg));
        } else if constexpr (is_buffer_v<T>) {
            _encode_buffer(OC_FORWARD(arg));
        }
        return *this;
    }
};

template<typename T>
class Shader {
    static_assert(always_false_v<T>);
};

template<typename... Args>
class Shader<void(Args...)> final : public Resource {
public:
    using signature = typename detail::canonical_signature_t<void(Args...)>;

private:
    ShaderTag _shader_tag;
    ArgumentList _argument_list;

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : Resource(device, SHADER,
                   device->create_shader(function)),
          _shader_tag(tag) {}

    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) {
        return ShaderDispatchCommand::create(handle(), _argument_list.ptr(), make_uint3(x, y, z));
    }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) { return dispatch(dim.x, dim.y, dim.z); }

    template<typename... A>
    requires std::is_invocable_v<signature, A...>
    Shader &operator()(A &&...args) noexcept {
        (_argument_list << ... << OC_FORWARD(args));
        return *this;
    }
};

}// namespace ocarina