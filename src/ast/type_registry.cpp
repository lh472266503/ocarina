//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"
#include "core/hash.h"

namespace katana {
/*
 *
 */
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
const Type *TypeRegistry::type_from(katana::string_view desc) noexcept {
    uint64_t hash = _hash(desc);
    if (auto iter = _type_set.find(hash); iter != _type_set.cend()) {
        return *iter;
    }

    using namespace std::string_view_literals;


    return nullptr;
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