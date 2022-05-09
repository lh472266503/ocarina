//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"

namespace katana::ast {
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

#define KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, tag)     \
    template<>                                                          \
    struct TypeDesc<S> {                                                \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return #S##sv;                                              \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 2>> {                                     \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",2>"sv;                                \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 3>> {                                     \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",3>"sv;                                \
        }                                                               \
    };                                                                  \
    template<>                                                          \
    struct TypeDesc<Vector<S, 4>> {                                     \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                  \
            return "vector<" #S ",4>"sv;                                \
        }                                                               \
    };

KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool, BOOL)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float, FLOAT)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, INT32)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, UINT32)

#undef KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

// matrices
template<>
struct TypeDesc<float2x2> {
    static constexpr katana::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<2>"sv;
    }
};

template<>
struct TypeDesc<float3x3> {
    static constexpr katana::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<3>"sv;
    }
};

template<>
struct TypeDesc<float4x4> {
    static constexpr katana::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<4>"sv;
    }
};

//template<typename... T>
//struct TypeDesc<std::tuple<T...>> {
//    static constexpr katana::string_view description() noexcept {
//
//    }
//}

};// namespace detail

}// namespace katana::ast