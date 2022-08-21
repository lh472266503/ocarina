//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "rhi/device.h"
#include "buffer.h"
#include "rhi/rtx/accel.h"
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

template<>
struct prototype_to_shader_invocation<Image> {
    using type = const Image &;
};
}// namespace detail

template<typename T>
using prototype_to_shader_invocation_t = typename detail::prototype_to_shader_invocation<T>::type;

class ArgumentList {
private:
    static constexpr auto Size = 200;
    ocarina::vector<void *> _args;
    ocarina::vector<handle_ty> _params;
    ocarina::array<std::byte, Size> _argument_data{};
    const Function &_function;
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
        push_handle_ptr(const_cast<void *>(buffer.handle_ptr()));
    }

    void _encode_image(const Image &image) noexcept;

    void _encode_accel(const Accel &accel) noexcept {
        push_handle_ptr(const_cast<void *>(accel.handle_ptr()));
    }

public:
    explicit ArgumentList(const Function &f) : _function(f){}
    [[nodiscard]] span<void *> ptr() noexcept { return _args; }
    [[nodiscard]] size_t num() const noexcept { return _args.size(); }
    void clear() noexcept { _cursor = 0; }

    void push_handle_ptr(void *address) noexcept {
        _cursor = mem_offset(_cursor, alignof(handle_ty));
        _args.push_back(address);
        OC_ASSERT(_cursor < Size);
        if (_function.is_raytracing()) {
            handle_ty handle = *(reinterpret_cast<const handle_ty *>(address));
            add_param(handle);
        }
    }

    void add_param(handle_ty handle) noexcept {
        _params.push_back(handle);
    }

    [[nodiscard]] span<handle_ty> params() noexcept { return _params; }

    template<typename T>
    ArgumentList &operator<<(T &&arg) {
        if constexpr (concepts::basic<T>) {
            _encode_basic(OC_FORWARD(arg));
        } else if constexpr (is_buffer_v<T>) {
            _encode_buffer(OC_FORWARD(arg));
        } else if constexpr (is_image_v<T>) {
            _encode_image(OC_FORWARD(arg));
        } else if constexpr (is_accel_v<T>) {
            _encode_accel(OC_FORWARD(arg));
        } else {
            static_assert(always_false_v<T>);
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
        virtual void compute_fit_size() noexcept {};
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
    ArgumentList _argument_list{_function};

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : RHIResource(device, SHADER,
                      device->create_shader(function)),
          _shader_tag(tag), _function(function) {}

    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) {
        if (_function.is_raytracing()) {
            return ShaderDispatchCommand::create(_function, handle(), _argument_list.params(), make_uint3(x, y, z));
        } else {
            _argument_list << make_uint3(x, y, z);
            return ShaderDispatchCommand::create(_function, handle(), _argument_list.ptr(), make_uint3(x, y, z));
        }
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
        if (_function.is_raytracing()) {
            for (const auto &uniform : _function.uniform_vars()) {
                handle_ty handle = *(reinterpret_cast<const handle_ty *>(uniform.handle_ptr()));
                _argument_list.add_param(handle);
            }
        } else {
            for (const auto &uniform : _function.uniform_vars()) {
                _argument_list.push_handle_ptr(const_cast<void *>(uniform.handle_ptr()));
            }
        }
        return *this;
    }
};

}// namespace ocarina