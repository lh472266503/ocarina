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

size_t Type::max_member_size() const noexcept {
    switch (_tag) {
        case Tag::BOOL:
        case Tag::FLOAT:
        case Tag::INT:
        case Tag::UINT:
        case Tag::UCHAR:
        case Tag::CHAR: return size();
        case Tag::VECTOR:
        case Tag::MATRIX:
        case Tag::ARRAY: return element()->max_member_size();
        case Tag::STRUCTURE: {
            size_t size = 0;
            for (const Type *member : _members) {
                if (member->size() > size) {
                    size = member->size();
                }
            }
            return size;
        }
        default:
            return 0;
    }
}

void Type::for_each(TypeVisitor *visitor) {
    TypeRegistry::instance().for_each(visitor);
}

void Type::update_name(ocarina::string_view desc) noexcept {
    switch (_tag) {
        case Tag::NONE:
            OC_ASSERT(0);
            break;
        case Tag::VECTOR:
            _name = ocarina::format("{}{}", element()->name(), dimension());
            break;
        case Tag::MATRIX:
            _name = ocarina::format("float{}x{}", dimension(), dimension());
            break;
        default:
            _name = desc;
            break;
    }
}

}// namespace ocarina
