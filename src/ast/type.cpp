//
// Created by Zero on 30/04/2022.
//

#include "type.h"
#include "type_registry.h"

namespace ocarina {


size_t Type::count() noexcept {
    return TypeRegistry::instance().type_count();
}

const Type *Type::from(std::string_view description) noexcept {
    return TypeRegistry::instance().type_from(description);
}

const Type *Type::at(uint32_t uid) noexcept {
    return TypeRegistry::instance().type_at(uid);
}

ocarina::span<const Type *const> Type::members() const noexcept {
    return {_members};
}

constexpr size_t Type::dimension() const noexcept {
    OC_ASSERT(is_array() || is_vector() || is_matrix() || is_texture());
    return _dimension;
}

const Type *Type::element() const noexcept {
    return _members.front();
}

constexpr auto Type::is_basic() const noexcept {
    return is_scalar() || is_vector() || is_matrix();
}

void Type::for_each(TypeVisitor *visitor) {
    TypeRegistry::instance().for_each(visitor);
}

constexpr bool Type::is_scalar() const noexcept {
    return _tag == Tag::BOOL || _tag == Tag::FLOAT || _tag == Tag::INT || _tag == Tag::UINT;
}

}// namespace ocarina
