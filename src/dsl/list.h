//
// Created by Zero on 2024/9/24.
//

#pragma once

#include "soa.h"

namespace ocarina {

template<typename TBuffer, typename T, AccessMode mode = AOS>
class List {
private:
    template<typename U>
    struct buffer_impl {
        using type = BufferStorage<TBuffer>;
    };

    template<>
    struct buffer_impl<ByteBuffer> {
        using type = ByteBuffer;
    };
    template<typename U>
    using buffer_t = typename buffer_impl<U>::type;

public:
    using element_type = T;
    static constexpr bool is_host = std::is_same_v<TBuffer, ByteBuffer>;
    static constexpr AccessMode access_mode = mode;
    static constexpr uint stride = sizeof(T);

private:
    buffer_t<TBuffer> buffer_;

public:
    template<typename U>
    requires std::is_same_v<U, ByteBuffer>
    explicit List(U buffer) : buffer_(std::move(buffer)) {}

    template<typename U>
    requires(!std::is_same_v<U, ByteBuffer>)
    explicit List(const U &u) : buffer_(u) {}

    [[nodiscard]] auto &buffer() noexcept {
        if constexpr (is_host) {
            return buffer_;
        } else {
            return *buffer_.get();
        }
    }

    [[nodiscard]] const auto &buffer() const noexcept {
        if constexpr (is_host) {
            return buffer_;
        } else {
            return *buffer_.get();
        }
    }

    [[nodiscard]] auto view() noexcept {
        if constexpr (access_mode == AOS) {
            return buffer().template aos_view<element_type>(storage_size_in_byte());
        } else {
            return buffer().template soa_view<element_type>(storage_size_in_byte());
        }
    }

    [[nodiscard]] auto view() const noexcept {
        if constexpr (access_mode == AOS) {
            return buffer().template aos_view<element_type>(storage_size_in_byte());
        } else {
            return buffer().template soa_view<element_type>(storage_size_in_byte());
        }
    }

    /// for dsl start
    template<typename Size = uint>
    [[nodiscard]] Var<Size> size_in_byte() const noexcept {
        if constexpr (is_host) {
            return buffer().expr().size_in_byte();
        } else {
            return buffer().size_in_byte();
        }
    }

    template<typename Index = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<Index> advance_index() noexcept {
        Var<Index> old_index = atomic_add(count<Index>(), 1);
        return old_index;
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> storage_size_in_byte() const noexcept {
        return size_in_byte() - sizeof(uint);
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> &count() noexcept {
        return buffer().template load_as<Size>(size_in_byte() - sizeof(uint));
    }

    template<typename Index, typename Arg, typename Size = uint>
    requires std::is_same_v<T, remove_device_t<Arg>> && is_integral_expr_v<Index>
    void write(const Index &index, const Arg &arg) noexcept {
        view().write(index, arg);
    }

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> read(const Index &index) const noexcept {
        return view().read(index);
    }

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> at(const Index &index) const noexcept {
        return read(index);
    }

    template<typename Index, typename Size = uint>
    requires is_integral_expr_v<Index>
    [[nodiscard]] Var<T> &at(const Index &index) noexcept {
        static_assert(access_mode == AOS, "must be AOS!");
        return view().at(index);
    }
    /// for dsl end
};

}// namespace ocarina