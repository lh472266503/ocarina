//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"
#include "core/hash.h"
#include "core/util.h"
#include "core/logging.h"

namespace katana {

TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry type_registry;
    return type_registry;
}

/*
 * TYPE: BASIC | ARRAY | VECTOR | MATRIX | STRUCT
 * BASIC: int | uint | bool | float
 * ARRAY: array<BASIC,N>
 * VECTOR: vector<BASIC,2> | vector<BASIC,3> | vector<BASIC,4>
 * MATRIX: matrix<2> | matrix<3> | matrix<4>
 * STRUCT: struct<4,TYPE...> | struct<8,TYPE...> | struct<16,TYPE...>
 */
const Type *TypeRegistry::parse_type(katana::string_view desc) noexcept {
    using namespace std::string_view_literals;
    int cursor = 0;
    auto find_identifier = [&cursor](katana::string_view desc) {
        uint i = 0u;
        for (; i < desc.size() && is_identifier(desc[i]); ++i)
            ;
        cursor = i;
        return desc.substr(0, i);
    };

    auto match = [&desc, &cursor](katana::string_view str) {
        if (!desc.substr(cursor).starts_with(str)) [[unlikely]] {
            KTN_ERROR("type error: expect '{}' from {}", str, desc);
        }
        cursor += static_cast<int>(str.size());
    };

    katana::string_view identifier = find_identifier(desc);
    auto type = katana::make_unique<Type>();
#define KTN_PARSE_BASIC_TYPE(T, TAG)   \
    if (identifier == #T##sv) {        \
        type->_size = sizeof(T);       \
        type->_alignment = alignof(T); \
        type->_description = #T;       \
        type->_tag = Type::Tag::TAG;   \
    } else

    KTN_PARSE_BASIC_TYPE(int, INT)
    KTN_PARSE_BASIC_TYPE(uint, UINT)
    KTN_PARSE_BASIC_TYPE(bool, BOOL)
    KTN_PARSE_BASIC_TYPE(float, FLOAT)

#undef KTN_PARSE_BASIC_TYPE

    if (identifier == "vector"sv) {
        type->_tag = Type::Tag::VECTOR;
        match("<");
    }

    return nullptr;
}

const Type *TypeRegistry::type_from(katana::string_view desc) noexcept {
    uint64_t hash = _hash(desc);
    if (auto iter = _type_set.find(hash); iter != _type_set.cend()) {
        return *iter;
    }
    const Type *type = parse_type(desc);
    return type;
}

size_t TypeRegistry::type_count() const noexcept {
    std::unique_lock lock{_mutex};
    return _types.size();
}

const Type *TypeRegistry::type_at(uint i) const noexcept {
    std::unique_lock lock{_mutex};
    return _types[i].get();
}

uint64_t TypeRegistry::_hash(katana::string_view desc) noexcept {
    using namespace std::string_view_literals;
    return hash64(desc, hash64("__hash_type"sv));
}
bool TypeRegistry::is_exist(katana::string_view desc) const noexcept {
    return is_exist(_hash(desc));
}

bool TypeRegistry::is_exist(uint64_t hash) const noexcept {
    return _type_set.find(hash) != _type_set.cend();
}

}// namespace katana