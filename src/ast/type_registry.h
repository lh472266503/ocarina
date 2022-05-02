//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"

namespace sycamore {
namespace ast {
template<typename T>
class Buffer;

template<typename T>
class BufferView;

template<typename T>
class Image;

template<typename T>
class ImageView;

namespace detail {

template<typename T>
struct TypeDesc {
    static_assert(always_false_v<T>, "Invalid type.");
};

#define SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, tag)     \
    template<>                                                          \
    struct TypeDesc<S> {                                                \
        static constexpr sycamore::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return #S##sv;                                              \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 2>> {                                     \
        static constexpr sycamore::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",2>"sv;                                \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 3>> {                                     \
        static constexpr sycamore::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",3>"sv;                                \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 4>> {                                     \
        static constexpr sycamore::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",4>"sv;                                \
        }                                                               \
    };

SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool, BOOL)
SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float, FLOAT)
SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, INT32)
SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, UINT32)

#undef SCM_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

}// namespace detail

}
}// namespace sycamore::ast