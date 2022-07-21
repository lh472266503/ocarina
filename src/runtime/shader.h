//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "device.h"
#include "resource.h"
#include "stream.h"
#include "command.h"
#include "core/concepts.h"
#include "core/util.h"

namespace ocarina {

class ArgumentList {
private:
    ocarina::vector<void *> _args;
    ocarina::vector<std::byte> _argument_data;

public:
    ArgumentList() = default;
    [[nodiscard]] span<const void *const> ptr() const noexcept { return _args; }
    [[nodiscard]] size_t num() const noexcept { return _args.size(); }

    template<typename T>
    void push_back(T &&arg) noexcept {
        size_t offset = mem_offset(_argument_data.size(), alignof(T));
        _argument_data.resize(offset + sizeof(T));
        auto dst_ptr = _argument_data.data() + offset;
        std::memcpy(dst_ptr, &arg, sizeof(T));
        _args.push_back(dst_ptr);
    }

    template<typename T>
    requires concepts::basic<T>
        ArgumentList &operator<<(T &&arg) {
        push_back(arg);
        return *this;
    }
};

template<typename... Args>
class Shader final : public Resource {
public:
    using signature = typename detail::canonical_signature_t<void(Args...)>;

    class Impl {
    public:
        virtual void launch(handle_ty stream) noexcept = 0;
    };

private:
    ShaderTag _shader_tag;
    ArgumentList _argument_list;

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : Resource(device, SHADER,
                   device->create_shader(function)),
          _shader_tag(tag) {}

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }

    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) {
        return ShaderDispatchCommand::create(_argument_list.ptr(), uint3(x, y, z));
    }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) { return dispatch(dim.x, dim.y, dim.z); }

    Shader &operator()(Args &&...args) noexcept {
        (_argument_list << ... << OC_FORWARD(args));
        return *this;
    }
};

}// namespace ocarina