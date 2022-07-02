//
// Created by Zero on 15/05/2022.
//

#pragma once
#include "core/basic_traits.h"
#include "dsl/struct.h"

namespace ocarina {

struct alignas(16) Hit {
    uint inst_id{};
    uint prim_id{};
    float2 bary;
};

template<>
struct ocarina::is_struct<ocarina::Hit> : std::true_type {};
template<>
struct ocarina::struct_member_tuple<ocarina::Hit> {
    using this_type = Hit;
    using type = ocarina::tuple<std::remove_cvref_t<decltype(this_type::inst_id)>, std::remove_cvref_t<decltype(this_type::prim_id)>, std::remove_cvref_t<decltype(this_type::bary)>>;
    using offset = std::index_sequence<__builtin_offsetof(this_type, inst_id), __builtin_offsetof(this_type, prim_id), __builtin_offsetof(this_type, bary)>;
    static_assert(is_valid_reflection_v<this_type, type, offset>, "may be order of members is wrong!");
    static constexpr size_t member_index(ocarina::string_view name) {
        constexpr string_view members[] = {"inst_id", "prim_id", "bary"};
        return std::find(std::begin(members),std::end(members), name) - std::begin(members);
    }
};
template<>
struct ocarina::detail::TypeDesc<ocarina::Hit> {
    using this_type = ocarina::Hit;
    static ocarina::string_view description() noexcept {
        static thread_local ocarina::string s = ocarina::format([] {struct FMT_COMPILE_STRING:fmt::compile_string{using char_type=fmt::remove_cvref_t<decltype("struct<{}" ",{}" ",{}" ",{}" ">"[0])>;constexpr operator fmt::basic_string_view<char_type>()const{return fmt::detail_exported::compile_string_to_view<char_type>("struct<{}" ",{}" ",{}" ",{}" ">");}};return FMT_COMPILE_STRING(); }(), alignof(this_type), ocarina::detail::TypeDesc<std::remove_cvref_t<decltype(this_type::inst_id)>>::description(), ocarina::detail::TypeDesc<std::remove_cvref_t<decltype(this_type::prim_id)>>::description(), ocarina::detail::TypeDesc<std::remove_cvref_t<decltype(this_type::bary)>>::description());
        return s;
    }
};
namespace detail {
template<>
struct Computable<ocarina::Hit> {
    using this_type = ocarina::Hit;

private:
    const Expression *_expression{nullptr};

public:
    [[nodiscard]] const Expression *expression() const noexcept { return _expression; }

protected:
    explicit Computable(const Expression *e) noexcept : _expression{e} {}
    Computable(Computable &&) noexcept = default;
    Computable(const Computable &) noexcept = default;

public:
    Var<std::remove_cvref_t<decltype(this_type::inst_id)>> inst_id{Function::current()->member(Type::of<decltype(this_type::inst_id)>(), expression(), 0)};
    Var<std::remove_cvref_t<decltype(this_type::prim_id)>> prim_id{Function::current()->member(Type::of<decltype(this_type::prim_id)>(), expression(), 1)};
    Var<std::remove_cvref_t<decltype(this_type::bary)>> bary{Function::current()->member(Type::of<decltype(this_type::bary)>(), expression(), 2)};
};
}// namespace detail

//OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary)


}// namespace ocarina