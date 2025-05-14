//
// Created by Zero on 30/04/2022.
//

#include "type.h"

#include <utility>
#include "type_registry.h"

namespace ocarina {

size_t Type::count() noexcept {
    return TypeRegistry::instance().type_count();
}

const Type *Type::from(std::string_view description, ocarina::string cname) noexcept {
    return TypeRegistry::instance().type_from(description, "");
}

const Type *Type::at(uint32_t uid) noexcept {
    return TypeRegistry::instance().type_at(uid);
}

ocarina::span<const Type *const> Type::members() const noexcept {
    return {members_};
}

const Type *Type::element() const noexcept {
    return members_.front();
}

void Type::set_cname(std::string s) const noexcept {
    cname_ = ocarina::move(s);
}

ocarina::string Type::simple_cname() const noexcept {
    return cname_.substr(cname_.find_last_of("::") + 1);
}

bool Type::is_valid() const noexcept {
    switch (tag_) {
        case Tag::STRUCTURE: {
            bool ret = true;
            for (auto member : members_) {
                ret = ret && member->is_valid();
            }
            return ret;
        }
        case Tag::ARRAY: return dimension() > 0;
        default: return true;
    }
}

const Type *Type::get_member(ocarina::string_view name) const noexcept {
    for (int i = 0; i < member_name_.size(); ++i) {
        if (member_name_[i] == name) {
            return members_[i];
        }
    }
    return nullptr;
}

void Type::update_dynamic_member_length(ocarina::string_view member_name, uint length) const noexcept {
    Type *member = const_cast<Type *>(get_member(member_name));
    member->dimension_ = length;
    member->size_ = length * member->max_member_size();
    update_structure_alignment_and_size();
}

void Type::update_structure_alignment_and_size() const noexcept {
    vector<MemoryBlock> blocks;
    for (const Type *member : members_) {
        MemoryBlock block;
        block.max_member_size = member->max_member_size();
        block.size = member->size();
        block.alignment = member->alignment();
        blocks.push_back(block);
    }
    const_cast<Type *>(this)->alignment_ = structure_alignment(blocks);
    const_cast<Type *>(this)->size_ = structure_size(blocks);
}

size_t Type::max_member_size() const noexcept {
    switch (tag_) {
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
            for (const Type *member : members_) {
                if (member->max_member_size() > size) {
                    size = member->max_member_size();
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

uint64_t Type::compute_hash() const noexcept {
    return hash64(description_, cname_);
}

void Type::update_name(ocarina::string_view desc) noexcept {
    switch (tag_) {
        case Tag::NONE:
            OC_ASSERT(0);
            break;
        case Tag::VECTOR:
            name_ = ocarina::format("{}{}", element()->name(), dimension());
            break;
        case Tag::MATRIX:
            name_ = ocarina::format("float{}x{}", dimension(), element()->dimension());
            break;
        default:
            name_ = desc;
            break;
    }
}

}// namespace ocarina
