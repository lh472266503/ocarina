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

[[nodiscard]] std::pair<int, int> bracket_matching(katana::string_view str, char l, char r) {
    int start = 0;
    int end = 0;
    int pair_count = 0;
    for (int i = 0; i < str.size(); ++i) {
        char ch = str[i];
        if (ch == l) {
            if (pair_count == 0) {
                start = i;
            }
            pair_count += 1;
        } else if (ch == r) {
            pair_count -= 1;
            if (pair_count == 0) {
                end = i;
            }
        }
    }
    return std::make_pair(start, end);
}

/*
 * TYPE: BASIC | ARRAY | VECTOR | MATRIX | STRUCT
 * BASIC: int | uint | bool | float
 * ARRAY: array<TYPE, N>
 * VECTOR: vector<BASIC,2> | vector<BASIC,3> | vector<BASIC,4>
 * MATRIX: matrix<2> | matrix<3> | matrix<4>
 * STRUCT: struct<4,TYPE...> | struct<8,TYPE...> | struct<16,TYPE...>
 */
const Type *TypeRegistry::parse_type(katana::string_view desc) noexcept {
    uint64_t hash = _hash(desc);
    if (auto iter = _type_set.find(hash); iter != _type_set.cend()) {
        return *iter;
    }

    using namespace std::string_view_literals;
    auto find_identifier = [&desc]() -> katana::string_view {
        uint i = 0u;
        for (; i < desc.size() && is_identifier(desc[i]); ++i)
            ;
        auto ret = desc.substr(0, i);
        desc = desc.substr(i);
        return ret;
    };

    auto type = katana::make_unique<Type>();
    type->_description = desc;
    type->_hash = hash;
    katana::string_view identifier = find_identifier();
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
        auto [start, end] = bracket_matching(desc, '<', '>');
        auto content = desc.substr(start + 1, end - start - 1);
        auto lst = string_split(content, ',');
        KTN_ASSERT(lst.size() == 2);
        auto type_str = lst[0];
        auto dimension_str = lst[1];
        type->_dimension = std::stoi(string(dimension_str));
        type->_members.push_back(parse_type(string_view(type_str)));
    }
    const Type *ret = type.get();
    add_type(std::move(type));
    return ret;
}

void TypeRegistry::add_type(katana::unique_ptr<Type> type) {
    _type_set.insert(type.get());
    type->_index = _types.size();
    _types.push_back(std::move(type));
}

const Type *TypeRegistry::type_from(katana::string_view desc) noexcept {
    return parse_type(desc);
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