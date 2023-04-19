//
// Created by Zero on 21/09/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "dsl/polymorphic.h"

using namespace ocarina;

namespace ocarina::detail {
template<typename T, typename Elm>
requires(sizeof(T) == sizeof(float))
void encode(vector<T> &data, Elm elm) noexcept {
    if constexpr (is_scalar_v<Elm>) {
        data.push_back(bit_cast<T>(elm));
    } else if constexpr (is_vector_v<Elm>) {
        for (int i = 0; i < vector_dimension_v<Elm>; ++i) {
            data.push_back(bit_cast<T>(elm[i]));
        }
    } else if constexpr (is_matrix_v<Elm>) {
        for (int i = 0; i < matrix_dimension_v<Elm>; ++i) {
            for (int j = 0; j < matrix_dimension_v<Elm>; ++j) {
                data.push_back(bit_cast<T>(elm[i][j]));
            }
        }
    } else {
        static_assert(always_false_v<Elm>);
    }
}

template<typename T>
[[nodiscard]] uint size_of(T t) noexcept {
    if constexpr (is_scalar_v<T>) {
        static_assert(sizeof(T) <= sizeof(float));
        return 1;
    } else if constexpr (is_vector_v<T>) {
        return vector_dimension_v<T>;
    } else if constexpr (is_matrix_v<T>) {
        return sqr(matrix_dimension_v<T>);
    } else {
        static_assert(always_false_v<T>);
    }
}

template<typename Ret, typename T>
[[nodiscard]] Var<Ret> decode(const Array<T> &array, uint offset) noexcept {
    if constexpr (is_scalar_v<Ret>) {
        return as<Ret>(array[offset]);
    } else if constexpr (is_vector_v<Ret>) {
        Var<Ret> ret;
        using element_ty = vector_element_t<Ret>;
        for (int i = 0; i < vector_dimension_v<Ret>; ++i) {
            ret[i] = as<element_ty>(array[offset + i]);
        }
        return ret;
    } else if constexpr (is_matrix_v<Ret>) {
        Var<Ret> ret;
        uint cursor = 0u;
        for (int i = 0; i < matrix_dimension_v<Ret>; ++i) {
            for (int j = 0; j < matrix_dimension_v<Ret>; ++j) {
                ret[i][j] = as<float>(array[cursor + offset]);
                ++cursor;
            }
        }
        return ret;
    } else {
        static_assert(always_false_v<Ret>);
    }
}

}// namespace ocarina::detail

#define OC_SERIALIZE_MEMBER(type, name) \
    type name{};                        \
    optional<Var<type>> _device_##name; \
    uchar _offset_##name;

#define OC_ENCODE_ELEMENT(name)             \
    _offset_##name = datas.size() - offset; \
    detail::encode(datas, name);

#define OC_DECODE_ELEMENT(name) _device_##name = detail::decode<decltype(name)>(values, _offset_##name);

#define OC_ENCODE_DECODE(...)                                            \
    uint _data_size{0u};                                                 \
                                                                         \
public:                                                                  \
    void encode(vector<float> &datas) noexcept {                         \
        uint offset = datas.size();                                      \
        MAP(OC_ENCODE_ELEMENT, __VA_ARGS__)                              \
        _data_size = datas.size() - offset;                              \
    }                                                                    \
    void decode(const DataAccessor<float> *da) noexcept {                \
        Array<float> values = da->read_dynamic_array<float>(_data_size); \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                              \
    }

struct Test {
    OC_SERIALIZE_MEMBER(float, a);
    OC_SERIALIZE_MEMBER(float, b);

    OC_ENCODE_DECODE(a, b)
};

int main() {

    Test t;

    //    cout << is_vector_v<float2>;
    //    cout << (!is_dsl_v<Float3>) && is_vector_v<Float3> ;

    return 0;

    //    cout << typeid(scalar_t<int3>).name() << endl;
    //    cout << typeid(scalar_t<float>).name() << endl;
    //    cout << typeid(scalar_t<float4x4>).name() << endl;
    //    cout << typeid(scalar_t<Float3>).name() << endl;
    //    cout << typeid(scalar_t<Float>).name() << endl;
    //    cout << typeid(scalar_t<Float4x4>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(vec_t<float3, 2>).name() << endl;
    //    cout << typeid(vec_t<float, 2>).name() << endl;
    //    cout << typeid(vec_t<float4x4, 2>).name() << endl;
    //    cout << typeid(vec_t<Float3, 2>).name() << endl;
    //    cout << typeid(vec_t<Float, 2>).name() << endl;
    //    cout << typeid(vec_t<Float4x4, 2>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(matrix_t<float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<float, 2>).name() << endl;
    //    cout << typeid(matrix_t<float4x4, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float4x4, 2>).name() << endl;
    //
    //    cout << typeid(boolean_t<Float>).name() << endl;
    //    cout << typeid(boolean_t<float>).name() << endl;

    return 0;
}