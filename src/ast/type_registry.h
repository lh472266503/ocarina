//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include <mutex>
#include "core/util.h"
#include "type_desc.h"

namespace ocarina {


template<typename T>
const Type *Type::of() noexcept {
    using raw_type = std::remove_cvref_t<T>;
    constexpr bool is_builtin = is_builtin_struct_v<raw_type>;
    auto ret = Type::from(TypeDesc<raw_type>::description());
    if constexpr (ocarina::is_struct_v<T>) {
        if constexpr (requires {
                          Var<T>::cname;
                      }) {
            ret->set_cname(Var<T>::cname);
        } else {
            ret->set_cname(string(ret->description()));
        }
        if constexpr (requires {
                          ocarina::struct_member_tuple<raw_type>::members;
                      }) {
            constexpr auto arr = ocarina::struct_member_tuple<raw_type>::members;
            constexpr int num = sizeof(ocarina::struct_member_tuple<raw_type>::members) / sizeof(arr[0]);
            const_cast<Type *>(ret)->update_member_name(arr, num);
        }
        using member_tuple = typename ocarina::struct_member_tuple<raw_type>::type;
        traverse_tuple(member_tuple{}, [&](auto elm) {
            using elm_t = decltype(elm);
            auto t = Type::of<elm_t>();
        });
    }
    return ret;
}


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
        constexpr ocarina::array<size_t, count> sizes{sizeof(M)...};
        constexpr ocarina::array<size_t, count> alignments{alignof(M)...};
        constexpr ocarina::array<size_t, count> offsets{os...};
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
    void parse_byte_buffer(Type *type, ocarina::string_view desc) noexcept;
    void parse_struct(Type *type, ocarina::string_view desc) noexcept;
    void parse_bindless_array(Type *type, ocarina::string_view desc) noexcept;

public:
    TypeRegistry &operator=(const TypeRegistry &) = delete;
    TypeRegistry &operator=(TypeRegistry &&) = delete;
    [[nodiscard]] static TypeRegistry &instance() noexcept;
    [[nodiscard]] const Type *parse_type(ocarina::string_view desc,
                                         uint64_t ext_hash = 0u) noexcept;
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