//
// Created by Zero on 10/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"

namespace ocarina {

template<typename T>
class Pool : public concepts::Noncopyable {
public:
    static constexpr size_t element_count = 64;

private:
    ocarina::vector<T *> _blocks;
    ocarina::vector<T *> _available_ptr;
public:
    template<typename ...Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {

    }

    void recycle(T *ptr) noexcept {

    }
};

}// namespace ocarina