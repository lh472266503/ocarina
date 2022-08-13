//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "rhi/device.h"
#include "buffer.h"
#include "stream.h"
#include "rhi/command.h"
#include "core/concepts.h"
#include "core/util.h"

namespace ocarina {

namespace detail {
template<typename T>
struct prototype_to_shader_invocation {
    using type = const T &;
};

template<typename T>
struct prototype_to_shader_invocation<Buffer<T>> {
    using type = const Buffer<T> &;
};

template<typename T>
struct prototype_to_shader_invocation<Image<T>> {
    using type = const Image<T> &;
};
}// namespace detail

template<typename T>
using prototype_to_shader_invocation_t = typename detail::prototype_to_shader_invocation<T>::type;

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
        oc_memcpy(dst_ptr, &arg, sizeof(T));
        _args.push_back(dst_ptr);
    }

    template<typename T>
    void _encode_buffer(const Buffer<T> &buffer) noexcept {
        push_handle_address(const_cast<handle_ty *>(buffer.handle_address()));
    }

    template<typename T>
    void _encode_texture(const Image<T> &texture) noexcept {
        push_handle_address(const_cast<handle_ty *>(texture.tex_handle_address()));
    }

public:
    ArgumentList() = default;
    [[nodiscard]] span<void *> ptr() noexcept { return _args; }
    [[nodiscard]] size_t num() const noexcept { return _args.size(); }
    void clear() noexcept { _cursor = 0; }

    void push_handle_address(handle_ty *address) noexcept {
        _cursor = mem_offset(_cursor, alignof(handle_ty));
        _args.push_back(address);
        _cursor += sizeof(handle_ty);
        OC_ASSERT(_cursor < Size);
    }

    template<typename T>
    ArgumentList &operator<<(T &&arg) {
        if constexpr (concepts::basic<T>) {
            _encode_basic(OC_FORWARD(arg));
        } else if constexpr (is_buffer_v<T>) {
            _encode_buffer(OC_FORWARD(arg));
        } else if constexpr (is_texture_v<T>) {
            _encode_texture(OC_FORWARD(arg));
        }
        return *this;
    }
};

template<typename T = int>
class Shader {
    static_assert(std::is_same_v<T, int>);

public:
    class Impl {
    public:
        virtual void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept = 0;
        virtual void compute_fit_size() noexcept = 0;
    };
};

template<typename... Args>
class Shader<void(Args...)> final : public RHIResource {
public:
    using signature = typename detail::canonical_signature_t<void(Args...)>;
    using Impl = typename Shader<>::Impl;

private:
    ShaderTag _shader_tag;
    const Function &_function;
    ArgumentList _argument_list;

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : RHIResource(device, SHADER,
                      device->create_shader(function)),
          _shader_tag(tag), _function(function) {}

    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) {
        _argument_list << make_uint3(x, y, z);
        return ShaderDispatchCommand::create(_function, handle(), _argument_list.ptr(), make_uint3(x, y, z));
    }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) { return dispatch(dim.x, dim.y, dim.z); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] ShaderTag shader_tag() const noexcept { return _shader_tag; }
    void compute_fit_size() noexcept { impl()->compute_fit_size(); }

    Shader &operator()(prototype_to_shader_invocation_t<Args> &&...args) noexcept {
        _argument_list.clear();
        (_argument_list << ... << OC_FORWARD(args));
#if CUDA_ARGUMENT_PUSH
        for (const auto &uniform : _function.uniform_vars()) {
            _argument_list.push_handle_address(const_cast<handle_ty *>(uniform.handle_address()));
        }
#endif
        return *this;
    }
};

}// namespace ocarina