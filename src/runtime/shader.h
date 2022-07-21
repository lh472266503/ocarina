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
    [[nodiscard]] span<void *const> ptr() const noexcept { return _args; }
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

template<typename T = int>
class Shader {
public:
    class Impl {
    public:
        virtual void launch(handle_ty stream) noexcept = 0;
    };
    static_assert(std::is_integral_v<T>);
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

    [[nodiscard]] Shader<>::Impl *impl() noexcept { return reinterpret_cast<Shader<>::Impl *>(_handle); }
    [[nodiscard]] const Shader<>::Impl *impl() const noexcept { return reinterpret_cast<const Shader<>::Impl *>(_handle); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) {
        return new ShaderDispatchCommand(_argument_list.ptr(), uint3());
//        return ShaderDispatchCommand::create(_argument_list.ptr(), uint3(x, y, z));
    }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) { return dispatch(dim.x, dim.y, dim.z); }

    template<typename... A>
    requires std::is_invocable_v<signature, expr_value_t<A>...>
        Shader &operator()(A &&...args) noexcept {
        (_argument_list << ... << OC_FORWARD(args));
        return *this;
    }
};

}// namespace ocarina