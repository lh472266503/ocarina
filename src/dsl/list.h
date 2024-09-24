//
// Created by Zero on 2024/9/24.
//

#pragma once

#include "soa.h"

namespace ocarina {

template<typename TBuffer, typename T, AccessMode mode = AOS>
class List {
public:
    using element_type = T;
    static constexpr bool is_host = std::is_same_v<TBuffer, ByteBuffer>;
    static constexpr AccessMode access_mode = mode;
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

    [[nodiscard]] auto view() const noexcept {
        if constexpr (access_mode == AOS) {
            return buffer().aos_view();
        } else {
            return buffer().soa_view();
        }
    }
};

}// namespace ocarina