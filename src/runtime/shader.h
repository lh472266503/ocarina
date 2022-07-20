//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "device.h"
#include "resource.h"
#include "stream.h"
#include "command.h"

namespace ocarina {

class ShaderInvoke {
private:
    ocarina::vector<const void *> _args;

public:
    explicit ShaderInvoke(vector<const void *> &&args) : _args(std::move(args)) {}
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1);
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) { return dispatch(dim.x, dim.y, dim.z); }
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

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : Resource(device, SHADER,
                   device->create_shader(function)),
          _shader_tag(tag) {}

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }

    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }

    Shader &operator()(Args &&...args) noexcept {
        return *this;
    }
};

}// namespace ocarina