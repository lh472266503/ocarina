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

namespace detail {
[[nodiscard]] bool is_identifier(char ch) noexcept {
    return std::isalpha(ch) || ch == '_';
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

[[nodiscard]] auto find_content(katana::string_view &str, char l = '<', char r = '>') {
    katana::vector<katana::string_view> ret;
    auto prev_token = str.find_first_of(l);
    constexpr auto token = ',';
    for (int i = 0; i < str.size(); ++i) {
        auto ch = str[i];
        if (ch == token) {
        }
    }
    return ret;
}

[[nodiscard]] katana::string_view find_identifier(katana::string_view &desc) {
    uint i = 0u;
    for (; i < desc.size() && is_identifier(desc[i]); ++i)
        ;
    auto ret = desc.substr(0, i);
    desc = desc.substr(i);
    return ret;
}
}// namespace detail

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

    KTN_USING_SV

    auto type = katana::make_unique<Type>();
    type->_description = desc;
    type->_hash = hash;
    katana::string_view identifier = detail::find_identifier(desc);
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
        parse_vector(type.get(), desc);
    } else if (identifier == "matrix"sv) {
        parse_matrix(type.get(), desc);
    } else if (identifier == "array"sv) {
        parse_array(type.get(), desc);
    } else if (identifier == "struct"sv) {
        parse_struct(type.get(), desc);
    } else [[unlikely]] {
        KTN_ERROR("invalid data type {}", desc);
    }
    const Type *ret = type.get();
    add_type(std::move(type));
    return ret;
}

void TypeRegistry::parse_vector(Type *type, katana::string_view &desc) noexcept {
    type->_tag = Type::Tag::VECTOR;
    auto [start, end] = detail::bracket_matching(desc, '<', '>');
    auto content = desc.substr(start + 1, end - start - 1);
    auto lst = string_split(content, ',');
    KTN_ASSERT(lst.size() == 2);
    auto type_str = lst[0];
    auto dimension_str = lst[1];
    auto dimension = std::stoi(string(dimension_str));
    type->_dimension = dimension;
    type->_members.push_back(parse_type(type_str));
    auto member = type->_members[0];
    if (!member->is_scalar()) [[unlikely]] {
        KTN_ERROR("invalid vector element: {}!", member->description());
    }
    type->_size = member->size() * (dimension == 3 ? 4 : dimension);
    type->_alignment = type->size();
}

void TypeRegistry::parse_matrix(Type *type, katana::string_view &desc) noexcept {
    type->_tag = Type::Tag::MATRIX;
    auto [start, end] = detail::bracket_matching(desc, '<', '>');
    auto dimension_str = desc.substr(start + 1, end - start - 1);
    auto dimension = std::stoi(string(dimension_str));
    type->_dimension = dimension;
    auto tmp_desc = katana::format("vector<float,{}>", dimension);
    type->_members.push_back(parse_type((tmp_desc)));

#define KTN_SIZE_ALIGN(dim)                      \
    if (dimension == dim) {                      \
        type->_size = sizeof(Matrix<dim>);       \
        type->_alignment = alignof(Matrix<dim>); \
    } else
    KTN_SIZE_ALIGN(2)
    KTN_SIZE_ALIGN(3)
    KTN_SIZE_ALIGN(4) {
        KTN_ERROR("invalid matrix dimension {}!", dimension)
    }
#undef KTN_SIZE_ALIGN
}

void TypeRegistry::parse_struct(Type *type, string_view &desc) noexcept {
    type->_tag = Type::Tag::STRUCTURE;
    auto [start, end] = detail::bracket_matching(desc, '<', '>');
    auto content = desc.substr(start + 1, end - start - 1);
    auto lst = string_split(content, ',');
    auto alignment_str = lst[0];
    auto alignment = std::stoi(string(alignment_str));
    type->_alignment = alignment;
    auto size = 0u;
    for (int i = 1; i < lst.size(); ++i) {
        auto type_str = lst[i];
        type->_members.push_back(parse_type(type_str));
        auto member = type->_members[i - 1];
        size = mem_offset(size, member->alignment());
        size += member->size();
    }
    type->_size = mem_offset(size, type->alignment());
}

void TypeRegistry::parse_array(Type *type, katana::string_view &desc) noexcept {
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