//
// Created by Zero on 30/04/2022.
//

#include "type.h"
#include "type_registry.h"

namespace katana {


size_t Type::count() noexcept {
    return TypeRegistry::instance().type_count();
}

const Type *Type::from(std::string_view description) noexcept {
    return TypeRegistry::instance().type_from(description);
}

const Type *Type::at(uint32_t uid) noexcept {
    return TypeRegistry::instance().type_at(uid);
}

katana::span<const Type *const> Type::members() const noexcept {
    return {_members};
}

constexpr size_t Type::dimension() const noexcept {
    KTN_ASSERT(is_array() || is_vector() || is_matrix() || is_texture());
    return _dimension;
}

constexpr auto Type::is_basic() const noexcept {
    return is_scalar() || is_vector() || is_matrix();
}

constexpr bool Type::is_scalar() const noexcept {
    return _tag == Tag::BOOL || _tag == Tag::FLOAT || _tag == Tag::INT || _tag == Tag::UINT;
}

}// namespace katana
