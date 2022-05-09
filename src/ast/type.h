//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/header.h"
#include "core/basic_types.h"
#include "core/stl.h"
#include <cassert>

namespace katana {

template<typename T>
struct array_dimension {
    static constexpr size_t value = 0u;
};

template<typename T, size_t N>
struct array_dimension<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct array_dimension<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
constexpr auto array_dimension_v = array_dimension<T>::value;

template<typename T>
struct array_element {
    using type = T;
};

template<typename T, size_t N>
struct array_element<T[N]> {
    using type = T;
};

template<typename T, size_t N>
struct array_element<std::array<T, N>> {
    using type = T;
};

template<typename T>
using array_element_t = typename array_element<T>::type;

template<typename T>
class is_array : public std::false_type {};

template<typename T, size_t N>
class is_array<T[N]> : public std::true_type {};

template<typename T, size_t N>
class is_array<std::array<T, N>> : public std::true_type {};

template<typename T>
constexpr auto is_array_v = is_array<T>::value;

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename T>
struct is_struct : std::false_type {};

template<typename... T>
struct is_struct<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_struct_v = is_struct<T>::value;

namespace detail {

template<typename T, size_t>
using array_to_tuple_element_t = T;

template<typename T, size_t N, size_t... i>
KTN_NODISCARD constexpr auto array_to_tuple_impl(std::array<T, N> array, std::index_sequence<i...>) noexcept {
    return static_cast<std::tuple<array_to_tuple_element_t<T, i>...>>(std::tuple(array[i]...));
}

template<typename T, size_t N>
KTN_NODISCARD constexpr auto array_to_tuple_impl(std::array<T, N> array = {}) noexcept {
    return array_to_tuple_impl(array, std::make_index_sequence<N>());
}

}// namespace detail

template<typename T>
struct struct_member_tuple {
    using type = std::tuple<T>;
};

template<typename... T>
struct struct_member_tuple<std::tuple<T...>> {
    using type = std::tuple<T...>;
};

template<typename T, size_t N>
struct struct_member_tuple<std::array<T, N>> {
    using type = std::remove_pointer_t<
        decltype(detail::array_to_tuple_impl<T, N>())>;
};

template<typename T, size_t N>
struct struct_member_tuple<T[N]> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<typename T, size_t N>
struct struct_member_tuple<Vector<T, N>> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<size_t N>
struct struct_member_tuple<Matrix<N>> {
    using type = typename struct_member_tuple<std::array<Vector<float, N>, N>>::type;
};

template<typename T>
using struct_member_tuple_t = typename struct_member_tuple<T>::type;

template<typename T>
struct canonical_layout {
    using type = struct_member_tuple_t<T>;
};

template<>
struct canonical_layout<float> {
    using type = std::tuple<float>;
};

template<>
struct canonical_layout<bool> {
    using type = std::tuple<bool>;
};

template<>
struct canonical_layout<int> {
    using type = std::tuple<int>;
};

template<>
struct canonical_layout<uint> {
    using type = std::tuple<uint>;
};

template<typename T>
struct canonical_layout<std::tuple<T>> {
    using type = typename canonical_layout<T>::type;
};

template<typename... T>
struct canonical_layout<std::tuple<T...>> {
    using type = std::tuple<typename canonical_layout<T>::type...>;
};

template<typename T>
using canonical_layout_t = typename canonical_layout<T>::type;

template<typename... T>
struct tuple_join {
    static_assert(always_false_v<T...>);
};

template<typename... T, typename... U>
struct tuple_join<std::tuple<T...>, U...> {
    using type = std::tuple<T..., U...>;
};

template<typename... A, typename... B, typename... C>
struct tuple_join<std::tuple<A...>, std::tuple<B...>, C...> {
    using type = typename tuple_join<std::tuple<A..., B...>, C...>::type;
};

template<typename... T>
using tuple_join_t = typename tuple_join<T...>::type;

namespace detail {

template<typename T>
struct dimension_impl {
    static constexpr auto value = dimension_impl<canonical_layout_t<T>>::value;
};

template<typename T, size_t N>
struct dimension_impl<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<size_t N>
struct dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename... T>
struct dimension_impl<std::tuple<T...>> {
    static constexpr auto value = sizeof...(T);
};

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

class Type;

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
};

class KTN_AST_API Type {
public:
    enum struct Tag : uint32_t {
        BOOL,
        FLOAT,
        INT,
        UINT,

        VECTOR,
        MATRIX,

        ARRAY,
        STRUCTURE,

        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        NONE
    };

private:
    uint64_t _hash{0};
    size_t _size{0};
    size_t _index{0};
    size_t _alignment{0};
    uint32_t _dimension{0};
    Tag _tag{Tag::NONE};
    katana::string _description;
    katana::vector<const Type *> _members;

public:
    template<typename T>
    KTN_NODISCARD static const Type *of() noexcept;

    template<typename T>
    KTN_NODISCARD static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }

    KTN_NODISCARD static const Type *from(std::string_view description) noexcept;

    KTN_NODISCARD static const Type *find(uint64_t hash) noexcept;

    KTN_NODISCARD static const Type *at(uint32_t uid) noexcept;

    KTN_NODISCARD static size_t count() noexcept;

    KTN_NODISCARD bool operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
    KTN_NODISCARD bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    KTN_NODISCARD bool operator<(const Type &rhs) const noexcept { return _index < rhs._index; }
    KTN_NODISCARD constexpr size_t index() const noexcept { return _index; }
    KTN_NODISCARD constexpr uint64_t hash() const noexcept { return _hash; }
    KTN_NODISCARD constexpr size_t size() const noexcept { return _size; }
    KTN_NODISCARD constexpr size_t alignment() const noexcept { return _alignment; }
    KTN_NODISCARD constexpr Tag tag() const noexcept { return _tag; }
    KTN_NODISCARD auto description() const noexcept { return katana::string_view{_description}; }

    KTN_NODISCARD constexpr size_t dimension() const noexcept {
        assert(is_array() || is_vector() || is_matrix() || is_texture());
        return _dimension;
    }

    KTN_NODISCARD katana::span<const Type *const> members() const noexcept;
    KTN_NODISCARD const Type *element() const noexcept;

    KTN_NODISCARD constexpr bool is_scalar() const noexcept {
        return _tag == Tag::BOOL || _tag == Tag::FLOAT || _tag == Tag::INT || _tag == Tag::UINT;
    }

    KTN_NODISCARD constexpr auto is_basic() const noexcept {
        return is_scalar() || is_vector() || is_matrix();
    }

    KTN_NODISCARD constexpr bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    KTN_NODISCARD constexpr bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    KTN_NODISCARD constexpr bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    KTN_NODISCARD constexpr bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    KTN_NODISCARD constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
    KTN_NODISCARD constexpr bool is_texture() const noexcept { return _tag == Tag::TEXTURE; }
    KTN_NODISCARD constexpr bool is_bindless_array() const noexcept { return _tag == Tag::BINDLESS_ARRAY; }
    KTN_NODISCARD constexpr bool is_accel() const noexcept { return _tag == Tag::ACCEL; }
};

}// namespace katana::ast