//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"
#include "core/hash.h"
#include "core/util.h"
#include "core/logging.h"
#include "dsl/rtx_type.h"

namespace ocarina {

TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry type_registry;
    return type_registry;
}

namespace detail {
[[nodiscard]] bool is_letter(char ch) noexcept {
    return std::isalpha(ch) || ch == '_';
}

[[nodiscard]] bool is_letter_or_num(char ch) noexcept {
    return std::isalnum(ch) || ch == '_';
}

[[nodiscard]] bool is_num(char ch) noexcept {
    return ch >= '0' && ch <= '9';
}

[[nodiscard]] std::pair<int, int> bracket_matching_far(ocarina::string_view str, char l = '<', char r = '>') {
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

[[nodiscard]] std::pair<int, int> bracket_matching_near(ocarina::string_view str, char l = '<', char r = '>') {
    int start = -1;
    int end = -1;
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
        if (pair_count == 0 && start != -1 && end != -1) {
            break;
        }
    }
    return std::make_pair(start, end);
}

[[nodiscard]] ocarina::string_view find_identifier(ocarina::string_view &str,
                                                   bool check_start_with_num = false) {
    OC_USING_SV
    uint i = 0u;
    for (; i < str.size() && is_letter_or_num(str[i]); ++i)
        ;
    auto ret = str.substr(0, i);
    if (ret == "vector"sv ||
        ret == "matrix"sv ||
        ret == "struct"sv ||
        ret == "array"sv) {
        auto [start, end] = bracket_matching_near(str);
        ret = str.substr(0, end + 1);
        str = str.substr(end + 1);
    } else {
        str = str.substr(i);
    }
    if (is_num(ret[0]) && check_start_with_num) [[unlikely]] {
        OC_ERROR_FORMAT("invalid identifier {} !", ret)
    }
    return ret;
}

[[nodiscard]] auto find_content(ocarina::string_view &str, char l = '<', char r = '>') {
    ocarina::vector<ocarina::string_view> ret;
    OC_USING_SV
    auto prev_token = str.find_first_of(l);
    constexpr auto token = ',';
    str = str.substr(prev_token + 1);
    while (true) {
        auto content = find_identifier(str);
        auto new_cursor = str.find_first_of(token) + 1;
        str = str.substr(new_cursor);
        ret.push_back(content);
        if (str[0] == r) {
            break;
        }
    }
    return ret;
}

}// namespace detail

/*
 * TYPE: BASIC | ARRAY | VECTOR | MATRIX | STRUCT | BUFFER
 * BASIC: int | uint | bool | float
 * ARRAY: array<BASIC | STRUCTURE, N>
 * BUFFER : buffer<BASIC | STRUCTURE>
 * VECTOR: vector<BASIC,2> | vector<BASIC,3> | vector<BASIC,4>
 * MATRIX: matrix<2> | matrix<3> | matrix<4>
 * STRUCT: struct<4,TYPE...> | struct<8,TYPE...> | struct<16,TYPE...>
 */
const Type *TypeRegistry::parse_type(ocarina::string_view desc) noexcept {
    if (desc == "void") {
        return nullptr;
    }
    uint64_t hash = _hash(desc);
    if (auto iter = _type_set.find(hash); iter != _type_set.cend()) {
        return *iter;
    }

    OC_USING_SV

    auto type = ocarina::make_unique<Type>();

#define OC_PARSE_BASIC_TYPE(T, TAG)    \
    if (desc == #T##sv) {              \
        type->_size = sizeof(T);       \
        type->_tag = Type::Tag::TAG;   \
        type->_alignment = alignof(T); \
        type->set_description(#T);     \
        type->_dimension = 1;          \
    } else

    OC_PARSE_BASIC_TYPE(int, INT)
    OC_PARSE_BASIC_TYPE(uint, UINT)
    OC_PARSE_BASIC_TYPE(bool, BOOL)
    OC_PARSE_BASIC_TYPE(float, FLOAT)
    OC_PARSE_BASIC_TYPE(uchar, UCHAR)

#undef OC_PARSE_BASIC_TYPE

    if (desc.starts_with("vector")) {
        parse_vector(type.get(), desc);
    } else if (desc.starts_with("matrix")) {
        parse_matrix(type.get(), desc);
    } else if (desc.starts_with("array")) {
        parse_array(type.get(), desc);
    } else if (desc.starts_with("struct")) {
        parse_struct(type.get(), desc);
    } else if (desc.starts_with("buffer")) {
        parse_buffer(type.get(), desc);
    } else if (desc.starts_with("image")) {
        parse_image(type.get(), desc);
    } else if (desc.starts_with("accel")) {
        parse_accel(type.get(), desc);
    } else {
        OC_ERROR("invalid data type ", desc);
    }
    type->set_description(desc);
    const Type *ret = type.get();
    add_type(std::move(type));
    return ret;
}

void TypeRegistry::parse_vector(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::VECTOR;
    auto [start, end] = detail::bracket_matching_far(desc, '<', '>');
    auto content = desc.substr(start + 1, end - start - 1);
    auto lst = string_split(content, ',');
    OC_ASSERT(lst.size() == 2);
    auto type_str = lst[0];
    auto dimension_str = lst[1];
    auto dimension = std::stoi(string(dimension_str));
    type->_dimension = dimension;
    type->_members.push_back(parse_type(type_str));
    auto member = type->_members.front();
    if (!member->is_scalar()) [[unlikely]] {
        OC_ERROR("invalid vector element: {}!", member->description());
    }
    type->_size = member->size() * (dimension == 3 ? 4 : dimension);
    type->_alignment = type->size();
}

void TypeRegistry::parse_matrix(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::MATRIX;
    auto [start, end] = detail::bracket_matching_far(desc, '<', '>');
    auto dimension_str = desc.substr(start + 1, end - start - 1);
    auto dimension = std::stoi(string(dimension_str));
    type->_dimension = dimension;
    auto tmp_desc = ocarina::format("vector<float,{}>", dimension);
    type->_members.push_back(parse_type((tmp_desc)));

#define OC_SIZE_ALIGN(dim)                       \
    if (dimension == (dim)) {                    \
        type->_size = sizeof(Matrix<dim>);       \
        type->_alignment = alignof(Matrix<dim>); \
    } else
    OC_SIZE_ALIGN(2)
    OC_SIZE_ALIGN(3)
    OC_SIZE_ALIGN(4) {
        OC_ERROR("invalid matrix dimension {}!", dimension)
    }
#undef OC_SIZE_ALIGN
}

void TypeRegistry::parse_struct(Type *type, string_view desc) noexcept {
    type->_tag = Type::Tag::STRUCTURE;
    auto lst = detail::find_content(desc);
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

void TypeRegistry::parse_buffer(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::BUFFER;
    auto lst = detail::find_content(desc);
    auto type_str = lst[0];
    const Type *element_type = parse_type(type_str);
    type->_members.push_back(element_type);
    auto alignment = element_type->alignment();
    type->_alignment = alignment;
}

void TypeRegistry::parse_image(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::IMAGE;
    type->_alignment = alignof(ImageData);
}

void TypeRegistry::parse_accel(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::ACCEL;
}

void TypeRegistry::parse_array(Type *type, ocarina::string_view desc) noexcept {
    type->_tag = Type::Tag::ARRAY;
    auto lst = detail::find_content(desc);
    auto type_str = lst[0];
    auto len = std::stoi(string(lst[1]));
    const Type *element_type = parse_type(type_str);
    type->_members.push_back(element_type);
    auto alignment = element_type->alignment();
    auto size = element_type->size() * len;
    type->_alignment = alignment;
    type->_dimension = len;
    type->_size = size;
}

void TypeRegistry::add_type(ocarina::unique_ptr<Type> type) {
    _type_set.insert(type.get());
    type->_index = _types.size();
    if (auto f = Function::current(); f != nullptr && type->is_structure() &&
                                      type->description() != detail::TypeDesc<Ray>::description() &&
                                      type->description() != detail::TypeDesc<Hit>::description()) {
        f->add_used_structure(type.get());
    }
    _types.push_back(std::move(type));
}

const Type *TypeRegistry::type_from(ocarina::string_view desc) noexcept {
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

uint64_t TypeRegistry::_hash(ocarina::string_view desc) noexcept {
    return Hashable::compute_hash<Type>(hash64(desc));
}
bool TypeRegistry::is_exist(ocarina::string_view desc) const noexcept {
    return is_exist(_hash(desc));
}

bool TypeRegistry::is_exist(uint64_t hash) const noexcept {
    return _type_set.find(hash) != _type_set.cend();
}

void TypeRegistry::for_each(TypeVisitor *visitor) const noexcept {
    for (const auto &t : _types) {
        visitor->visit(t.get());
    }
}

}// namespace ocarina