//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "rhi/device.h"
#include "buffer.h"
#include "rhi/rtx/accel.h"
#include "rhi/resources/bindless_array.h"
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
struct prototype_to_shader_invocation<Texture> {
    using type = const Texture &;
};

template<>
struct prototype_to_shader_invocation<BindlessArray> {
    using type = const BindlessArray &;
};

}// namespace detail

template<typename T>
using prototype_to_shader_invocation_t = typename detail::prototype_to_shader_invocation<T>::type;

class ArgumentList {
private:
    static constexpr auto Size = 200;
    ocarina::vector<void *> _args;
    ocarina::array<std::byte, Size> _pod_data{};
    const Function *_function{};
    size_t _cursor{};
    ocarina::vector<MemoryBlock> _blocks;
    ocarina::vector<std::byte> _argument_data;

private:
    template<typename T>
    requires std::is_trivially_destructible_v<T>
    void _encode_pod_type(T &&arg) noexcept {
        _cursor = mem_offset(_cursor, alignof(T));
        auto dst_ptr = _pod_data.data() + _cursor;
        _cursor += sizeof(T);
        OC_ASSERT(_cursor < Size);
        oc_memcpy(dst_ptr, &arg, sizeof(T));
        push_memory_block({dst_ptr, sizeof(T), alignof(T), Type::of<T>()->max_member_size()});
    }

    template<typename TBuffer>
    void _encode_buffer(const TBuffer &buffer) noexcept {
        push_memory_block(buffer.memory_block());
    }

    void _encode_texture(const Texture &texture) noexcept;
    void _encode_bindless_array(const BindlessArray &bindless_array) noexcept;
    void _encode_accel(const Accel &accel) noexcept {
        push_memory_block(accel.memory_block());
    }

public:
    explicit ArgumentList(const Function *f = nullptr) : _function(f) {}
    [[nodiscard]] span<void *> ptr() noexcept { return _args; }
    [[nodiscard]] size_t num() const noexcept { return _args.size(); }
    void clear() noexcept {
        _cursor = 0;
        _args.clear();
        _blocks.clear();
    }

    [[nodiscard]] const Function &function() const noexcept {
        return *_function;
    }

    void push_memory_block(const MemoryBlock &block) noexcept {
        if (!_function->is_raytracing()) {
            _args.push_back(const_cast<void *>(block.address));
        }
        add_block(block);
    }

    void add_block(const MemoryBlock &block) noexcept {
        _blocks.push_back(block);
    }

    /// move exterior data of temporary variable to argument data
    void move_argument_data() noexcept {
        uint size = structure_size(_blocks);
        if (!_function->is_raytracing()) {
            size = mem_offset(size, alignof(uint3)) + sizeof(uint3);
        }
        _argument_data.resize(size);
        size_t offset = 0;
        _args.clear();
        auto head = reinterpret_cast<std::byte *>(_argument_data.data());
        for (const MemoryBlock &block : _blocks) {
            offset = mem_offset(offset, block.alignment);

            oc_memcpy(head + offset,
                      block.address, block.size);
            if (!_function->is_raytracing()) {
                _args.push_back(reinterpret_cast<void *>(head + offset));
            }
            offset += block.size;
        }
    }

    [[nodiscard]] span<const std::byte> argument_data() const noexcept { return _argument_data; }

    [[nodiscard]] span<const MemoryBlock> blocks() noexcept { return _blocks; }

    template<typename T>
    ArgumentList &operator<<(T &&arg) {
        if constexpr (is_param_struct_v<T>) {
            using arg_type = std::remove_cvref_t<T>;
            using Tuple = ocarina::struct_member_tuple_t<arg_type>;
            const std::byte *head = reinterpret_cast<const std::byte *>(addressof(arg));
            traverse_tuple(Tuple{}, [&]<typename Elm>(const Elm &_, uint index) {
                constexpr auto offset_array = ocarina::struct_member_tuple<arg_type>::offset_array;
                uint offset = offset_array[index];
                const std::byte *addr = head + offset;
                const Elm &elm = reinterpret_cast<const Elm &>(*addr);
                *this << elm;
            });
        } else if constexpr (is_buffer_v<T>) {
            _encode_buffer(OC_FORWARD(arg));
        } else if constexpr (is_texture_v<T>) {
            _encode_texture(OC_FORWARD(arg));
        } else if constexpr (is_accel_v<T>) {
            _encode_accel(OC_FORWARD(arg));
        } else if constexpr (is_bindless_array_v<T>) {
            _encode_bindless_array(OC_FORWARD(arg));
        } else {
            _encode_pod_type(OC_FORWARD(arg));
        }
        return *this;
    }
};

class ShaderInvoke {
private:
    SP<ArgumentList> _argument_list;
    handle_ty _shader_entry;

public:
    ShaderInvoke(handle_ty entry, SP<ArgumentList> argument_list)
        : _shader_entry(entry),
          _argument_list(ocarina::move(argument_list)) {}

    template<typename T>
    ShaderInvoke &operator<<(T &&arg) {
        *_argument_list << OC_FORWARD(arg);
        return *this;
    }
    [[nodiscard]] const Function &function() const noexcept { return _argument_list->function(); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint x, uint y = 1, uint z = 1) const noexcept {
        if (!function().is_raytracing()) {
            *_argument_list << make_uint3(x, y, z);
        }
        return ShaderDispatchCommand::create(_shader_entry, _argument_list, make_uint3(x, y, z));
    }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint2 dim) const noexcept { return dispatch(dim.x, dim.y, 1); }
    [[nodiscard]] ShaderDispatchCommand *dispatch(uint3 dim) const noexcept { return dispatch(dim.x, dim.y, dim.z); }
};

template<typename T = int>
class Shader {
    static_assert(std::is_same_v<T, int>);

public:
    class Impl {
    public:
        virtual ~Impl() = default;
        virtual void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept = 0;
        virtual void compute_fit_size() noexcept {};
    };
};

template<typename... Args>
class Shader<void(Args...)> final : public RHIResource {
public:
    using signature = canonical_signature_t<void(Args...)>;
    using Impl = typename Shader<>::Impl;

private:
    ShaderTag _shader_tag{};
    ocarina::shared_ptr<Function> _function{};

public:
    Shader() = default;
    Shader(Device::Impl *device, ocarina::shared_ptr<Function> function, ShaderTag tag) noexcept
        : RHIResource(device, SHADER,
                      device->create_shader(*function)),
          _shader_tag(tag), _function(ocarina::move(function)) {}

    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] ShaderTag shader_tag() const noexcept { return _shader_tag; }
    void compute_fit_size() noexcept { impl()->compute_fit_size(); }
    [[nodiscard]] bool has_function() const noexcept { return _function != nullptr; }
    [[nodiscard]] ShaderInvoke operator()(prototype_to_shader_invocation_t<Args> &&...args) const noexcept {
        auto argument_list = make_shared<ArgumentList>(_function.get());
        (*argument_list << ... << OC_FORWARD(args));
        for (const auto &var : _function->captured_resources()) {
            argument_list->push_memory_block(var.block());
        }
        argument_list->move_argument_data();
        ShaderInvoke shader_invoke{handle(), ocarina::move(argument_list)};
        return shader_invoke;
    }
};

}// namespace ocarina