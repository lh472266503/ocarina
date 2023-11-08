//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include <mutex>
#include "core/util.h"
#include "core/string_util.h"

namespace ocarina {
template<typename T, int... dims>
class Buffer;

template<typename T, int... dims>
class BufferView;

class Texture;

class Accel;

template<typename T>
struct TypeDesc {
    static_assert(always_false_v<T>, "Invalid type.");
};

#define OC_MAKE_VECTOR_DESC_NAME(S, N)                                 \
    template<>                                                         \
    struct TypeDesc<Vector<S, N>> {                                    \
        static constexpr ocarina::string_view description() noexcept { \
            return ocarina::string_view("vector<" #S "," #N ">");      \
        }                                                              \
        static constexpr ocarina::string_view name() noexcept {        \
            return ocarina::string_view(#S #N);                        \
        }                                                              \
    };

#define OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(S)          \
    template<>                                                         \
    struct TypeDesc<S> {                                               \
        static constexpr ocarina::string_view description() noexcept { \
            using namespace std::string_view_literals;                 \
            return #S##sv;                                             \
        }                                                              \
        static constexpr ocarina::string_view name() noexcept {        \
            return description();                                      \
        }                                                              \
    };                                                                 \
    OC_MAKE_VECTOR_DESC_NAME(S, 2)                                     \
    OC_MAKE_VECTOR_DESC_NAME(S, 3)                                     \
    OC_MAKE_VECTOR_DESC_NAME(S, 4)

OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(bool)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(float)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(int)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uint)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(uchar)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(char)
OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION(ushort)

#undef OC_MAKE_VECTOR_DESC_NAME
#undef OC_MAKE_SCALAR_AND_VECTOR_TYPE_DESC_SPECIALIZATION

template<>
struct TypeDesc<void> {
    static constexpr ocarina::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "void"sv;
    }

    static constexpr ocarina::string_view name() noexcept {
        return description();
    }
};

/// matrices
template<>
struct TypeDesc<float2x2> {
    static constexpr ocarina::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<2>"sv;
    }

    static constexpr ocarina::string_view name() noexcept {
        return "float2x2";
    }
};

template<>
struct TypeDesc<float3x3> {
    static constexpr ocarina::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<3>"sv;
    }

    static constexpr ocarina::string_view name() noexcept {
        return "float3x3";
    }
};

template<>
struct TypeDesc<float4x4> {
    static constexpr ocarina::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "matrix<4>"sv;
    }
    static constexpr ocarina::string_view name() noexcept {
        return "float4x4";
    }
};

template<typename T>
struct TypeDesc<vector<T>> {
    static_assert(alignof(T) >= 4u);

    static ocarina::string_view description() noexcept {
        static thread_local auto s = ocarina::format(
            "d_array<{},0>",
            TypeDesc<T>::description());
        return s;
    }

    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T, size_t N>
struct TypeDesc<std::array<T, N>> {
    static_assert(alignof(T) >= 4u);
    static ocarina::string description() noexcept {
        static thread_local auto s = ocarina::format(
            "array<{},{}>",
            TypeDesc<T>::description(), N);
        return s;
    }

    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T, int... Dims>
struct TypeDesc<Buffer<T, Dims...>> {
    static_assert(alignof(T) >= 4u);
    static ocarina::string description() noexcept {
        static thread_local string str = []() -> string {
            auto ret = ocarina::format("buffer<{}", TypeDesc<T>::description());
            (ret.append(",").append(to_string(Dims)), ...);
            ret.append(">");
            return ret;
        }();
        return str;
    }
    static ocarina::string name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<Texture> {
    static ocarina::string_view description() noexcept {
        return "texture";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<Accel> {
    static ocarina::string_view description() noexcept {
        return "accel";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T, size_t N>
struct TypeDesc<T[N]> : public TypeDesc<std::array<T, N>> {};

template<typename... T>
struct TypeDesc<ocarina::tuple<T...>> {
    static ocarina::string_view description() noexcept {
        static thread_local ocarina::string str = []() -> ocarina::string {
            auto ret = ocarina::format("struct<{}", alignof(ocarina::tuple<T...>));
            (ret.append(",").append(TypeDesc<T>::description()), ...);
            ret.append(">");
            return ret;
        }();
        return string_view(str);
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<>
struct TypeDesc<ResourceArray> {
    static ocarina::string_view description() noexcept {
        return "resourceArray";
    }
    static ocarina::string_view name() noexcept {
        return description();
    }
};

template<typename T>
const Type *Type::of() noexcept {
    auto ret = Type::from(TypeDesc<std::remove_cvref_t<T>>::description());
    if constexpr (ocarina::is_struct_v<T>) {
        if constexpr (requires() {
                          Var<T>::cname;
                      }) {
            ret->set_cname(Var<T>::cname);
        }
    }
    return ret;
}

/// make struct type description
#define OC_MAKE_STRUCT_MEMBER_FMT(member) ",{}"

#define OC_MAKE_STRUCT_MEMBER_DESC(member) \
    ocarina::TypeDesc<std::remove_cvref_t<decltype(this_type::member)>>::description()

#define OC_MAKE_STRUCT_DESC(S, ...)                                                        \
    template<>                                                                             \
    struct ocarina::TypeDesc<S> {                                                          \
        using this_type = S;                                                               \
        static ocarina::string_view description() noexcept {                               \
            static thread_local ocarina::string s = ocarina::format(                       \
                FMT_STRING("struct<{}" MAP(OC_MAKE_STRUCT_MEMBER_FMT, ##__VA_ARGS__) ">"), \
                alignof(this_type),                                                        \
                MAP_LIST(OC_MAKE_STRUCT_MEMBER_DESC, ##__VA_ARGS__));                      \
            return s;                                                                      \
        }                                                                                  \
    };

namespace detail {
template<typename S, typename Members, typename offsets>
struct is_valid_reflection : std::false_type {};

template<typename S, typename... M, typename I, I... os>
struct is_valid_reflection<S, ocarina::tuple<M...>, std::integer_sequence<I, os...>> {
    //    static_assert(((!is_struct_v<M>)&&...));
    static_assert((!is_bool_vector_v<M> && ...),
                  "Boolean vectors are not allowed in DSL "
                  "structures since their may have different "
                  "layouts on different platforms.");

private:
    [[nodiscard]] static constexpr bool _check() noexcept {
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

class OC_AST_API TypeRegistry {
public:
    struct TypePtrHash {
        using is_transparent = void;
        [[nodiscard]] uint64_t operator()(const Type *type) const noexcept { return type->hash(); }
        [[nodiscard]] uint64_t operator()(uint64_t hash) const noexcept { return hash; }
    };

    struct TypePtrEqual {
        using is_transparent = void;
        template<typename Lhs, typename Rhs>
        [[nodiscard]] bool operator()(Lhs &&lhs, Rhs &&rhs) const noexcept {
            constexpr TypePtrHash hash;
            return hash(std::forward<Lhs>(lhs)) == hash(std::forward<Rhs>(rhs));
        }
    };

    ocarina::vector<ocarina::unique_ptr<Type>> _types;
    ocarina::unordered_set<Type *, TypePtrHash, TypePtrEqual> _type_set;
    mutable std::mutex _mutex;
    TypeRegistry() = default;

private:
    [[nodiscard]] static uint64_t _hash(ocarina::string_view desc) noexcept;
    void parse_vector(Type *type, ocarina::string_view desc) noexcept;
    void parse_matrix(Type *type, ocarina::string_view desc) noexcept;
    void parse_array(Type *type, ocarina::string_view desc) noexcept;
    void parse_dynamic_array(Type *type, ocarina::string_view desc) noexcept;
    void parse_buffer(Type *type, ocarina::string_view desc) noexcept;
    void parse_texture(Type *type, ocarina::string_view desc) noexcept;
    void parse_accel(Type *type, ocarina::string_view desc) noexcept;
    void parse_struct(Type *type, ocarina::string_view desc) noexcept;
    void parse_resource_array(Type *type, ocarina::string_view desc) noexcept;

public:
    TypeRegistry &operator=(const TypeRegistry &) = delete;
    TypeRegistry &operator=(TypeRegistry &&) = delete;
    [[nodiscard]] static TypeRegistry &instance() noexcept;
    [[nodiscard]] const Type *parse_type(ocarina::string_view desc) noexcept;
    [[nodiscard]] bool is_exist(ocarina::string_view desc) const noexcept;
    [[nodiscard]] bool is_exist(uint64_t hash) const noexcept;
    [[nodiscard]] const Type *type_from(ocarina::string_view desc) noexcept;
    [[nodiscard]] const Type *type_at(uint i) const noexcept;
    [[nodiscard]] size_t type_count() const noexcept;
    void add_type(ocarina::unique_ptr<Type> type);
    static void try_add_to_current_function(const Type *type) noexcept;
    void for_each(TypeVisitor *visitor) const noexcept;
};

};// namespace ocarina