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
    static constexpr size_t element_count = 32;
    using element_type = std::remove_cvref_t<T>;
    static constexpr bool trivially = std::is_trivially_destructible_v<element_type>;

private:
    ocarina::vector<T *> _blocks;
    ocarina::vector<T *> _available_objects;

    void _enlarge() {
        T *ptr = ocarina::allocate<T>(element_count, !trivially);
        _blocks.push_back(ptr);
        _available_objects.reserve(element_count);
        for (int i = 0; i < element_count; ++i) {
            _available_objects.push_back(&(ptr[i]));
        }
    }

public:
    ~Pool() {
        if constexpr (!trivially) {
            for (auto &ptr : _available_objects) {
                ptr->~T();
            }
        }
        for (auto &ptr : _blocks) {
            ocarina::delete_with_allocator(ptr, !trivially);
        }
    }

    template<typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        if (_available_objects.empty()) {
            _enlarge();
        }
        T *ptr = _available_objects.back();
        _available_objects.pop_back();
        std::construct_at(ptr, OC_FORWARD(args)...);
        return ptr;
    }

    void recycle(T *ptr) noexcept {
        if (!trivially) {
            ptr->~T();
        }
        _available_objects.push_back(ptr);
    }
};

}// namespace ocarina