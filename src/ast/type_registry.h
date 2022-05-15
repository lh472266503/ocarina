//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/util.h"
#include "core/string_util.h"

namespace katana {
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

#define KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S, tag)   \
    template<>                                                        \
    struct TypeDesc<S> {                                              \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                \
            return #S##sv;                                            \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 2>> {                                   \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",2>"sv;                              \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 3>> {                                   \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",3>"sv;                              \
        }                                                             \
    };                                                                \
    template<>                                                        \
    struct TypeDesc<Vector<S, 4>> {                                   \
        static constexpr katana::string_view description() noexcept { \
            using namespace std::string_view_literals;                \
            return "vector<" #S ",4>"sv;                              \
        }                                                             \
    };

KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool, BOOL)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float, FLOAT)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int, INT32)
KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint, UINT32)

#undef KTN_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

/// matrices
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

template<typename... T>
struct TypeDesc<std::tuple<T...>> {
    static katana::string_view description() noexcept {
        static thread_local katana::string str = []() -> katana::string {
            auto ret = katana::format("struct<{}", alignof(std::tuple<T...>));
            (ret.append(",").append(TypeDesc<T>::description()), ...);
            ret.append(">");
            return ret;
        }();
        return string_view(str);
    }
};

};// namespace detail

/// make struct type description
#define KTN_MAKE_STRUCT_MEMBER_FMT(member) ",{}"

#define KTN_MAKE_STRUCT_MEMBER_DESC(member) \
    katana::detail::TypeDesc<std::remove_cvref_t<decltype(this_type::member)>>::description()

#define KTN_MAKE_STRUCT_DESC(S, ...)                                                        \
    template<>                                                                              \
    struct katana::detail::TypeDesc<S> {                                                    \
        using this_type = S;                                                                \
        static katana::string_view description() noexcept {                                 \
            static thread_local katana::string s = katana::format(                          \
                FMT_STRING("struct<{}" MAP(KTN_MAKE_STRUCT_MEMBER_FMT, ##__VA_ARGS__) ">"), \
                alignof(this_type),                                                         \
                MAP_LIST(KTN_MAKE_STRUCT_MEMBER_DESC, ##__VA_ARGS__));                      \
            return s;                                                                       \
        }                                                                                   \
    };

namespace detail {
template<typename S, typename Members, typename offsets>
struct is_valid_reflection : std::false_type {};

template<typename S, typename... M, typename I, I... os>
struct is_valid_reflection<S, std::tuple<M...>, std::integer_sequence<I, os...>> {
    static_assert(((!is_struct_v<M>)&&...));
    static_assert((!is_bool_vector_v<M> && ...),
                  "Boolean vectors are not allowed in DSL "
                  "structures since their may have different "
                  "layouts on different platforms.");

private:
    KTN_NODISCARD static constexpr bool _check() noexcept {
        constexpr auto count = sizeof...(M);
        static_assert(sizeof...(os) == count);
        constexpr std::array<size_t, count> sizes{sizeof(M)...};
        constexpr std::array<size_t, count> alignments{alignof(M)...};
        constexpr std::array<size_t, count> offsets{os...};
        size_t cur_offset = 0u;
        for (auto i = 0u; i < count; ++i) {
            auto offset = offsets[i];
            auto size = sizes[i];
            auto alignment = alignments[i];
            cur_offset = mem_offset(cur_offset, alignment);
            if (cur_offset != offset) {
                return false;
            }
            cur_offset += size;
        }
        constexpr auto struct_size = sizeof(S);
        constexpr auto struct_alignment = alignof(S);
        cur_offset = mem_offset(cur_offset, struct_alignment);
        return cur_offset == struct_size;
    };

public:
    static constexpr bool value = _check();
};
}// namespace detail

template<typename S, typename M, typename I>
static constexpr bool is_valid_reflection_v = detail::is_valid_reflection<S, M, I>::value;

}// namespace katana