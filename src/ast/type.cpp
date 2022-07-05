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

const Type *Type::element() const noexcept {
    return _members.front();
}

void Type::for_each(TypeVisitor *visitor) {
    TypeRegistry::instance().for_each(visitor);
}


}// namespace ocarina
